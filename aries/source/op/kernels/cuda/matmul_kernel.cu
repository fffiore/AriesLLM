#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"
#include "base/cuda_config.h"
#include <cublas_v2.h>

namespace kernel {

  __global__ void matmul_kernel_cu_fp32(const float* input,  const float* weight, 
                                              float* output, int M, // hidden dim (e.g. 896)
                                              int K  // output dim (e.g. 151936)
                                              ) {
    // blockIdx.y 负责 sequence 维度
    int seq_idx = blockIdx.y;
    // 偏移输入输出指针
    const float* curr_input = input + seq_idx * M;
    float* curr_output = output + seq_idx * K;
    // blockIdx.x 负责 output channel (vocab) 维度
    // 每个 block 处理 128 个输出元素 (假设 blockDim.x=128)
    int row_start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int p = row_start; p < K; p += stride) {
      // 计算第 p 个输出 (即第 p 个词的 logit)
      // 对应 Weight 矩阵的第 p 行 (转置存储)
      float sum = 0.0f;
      int weight_offset = p * M;
      // 标量循环，编译器会自动进行 SIMD 优化
      for (int i = 0; i < M; ++i) {
          sum += curr_input[i] * weight[weight_offset + i];
      }
      curr_output[p] = sum;
    }
  }

  __global__ void matmul_kernel_cu_int8(const float* input, const int8_t* weight,
                                              const float* scales, const int32_t group_size,
                                              float* output, int M, int K) {
    int seq_idx = blockIdx.y;
    const float* curr_input = input + seq_idx * M;
    float* curr_output = output + seq_idx * K;
    int row_start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int p = row_start; p < K; p += stride) {
      float sum = 0.0f;
      int weight_idx_base = p * M;
      
      for (int i = 0; i < M; ++i) {
        int w_idx = weight_idx_base + i;
        int group_idx = w_idx / group_size;
        float w_dequant = static_cast<float>(weight[w_idx]) * scales[group_idx];
        sum += curr_input[i] * w_dequant;
      }
      curr_output[p] = sum;
    }
  }

  void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, const float scale, const CudaConfig* config) {
    // weight 形状: [output_dim, hidden_dim] (例如 [vocab_size, hidden])
    // weight.get_dim(0) = K (rows/vocab), get_dim(1) = M (cols/hidden)
    const int32_t output_dim = weight.get_dim(0); 
    const int32_t hidden_dim = weight.get_dim(1);

    int32_t seq_len = 1;
    if (input.dims_size() == 2) {
        seq_len = input.get_dim(0);
    }
    const float* d_input = input.ptr<float>();
    const float* d_weight = weight.ptr<float>();
    float* d_output = const_cast<float*>(output.ptr<float>());

    // 3. cuBLAS 调用
    // output(RowMajor) = input(RowMajor) * weight^T(RowMajor)
    // 维度: [seq, hidden] * [hidden, output] -> [seq, output]
    
    // cuBLAS 是列优先 (Column Major)，需要转置，C(Row) = A(Row) * B(Row)  <=>  C^T(Col) = B^T(Col) * A^T(Col)
    // d_weight 是 [output, hidden] 的连续内存。
    // 在列优先视角下，是一个 [hidden, output] 的矩阵，用 CUBLAS_OP_T 对其转置后 [output, hidden]。
    // d_input 是 [seq, hidden] 的连续内存。
    // 在列优先视角下，是一个 [hidden, seq] 的矩阵，用 CUBLAS_OP_N (不转置)。
    // 结果 d_output 是 [seq, output] 的连续内存，列优先视角下，它是一个 [output, seq] 的矩阵。
    
    // 公式: C(mxn) = A(mxk) * B(kxn)
    // 映射到 cuBLAS 参数:
    // m = output_dim
    // n = seq_len
    // k = hidden_dim
    // A = d_weight (配合 OP_T)
    // B = d_input  (配合 OP_N)
    // lda = hidden_dim (d_weight 的 leading dimension，即列优先下的行数)
    // ldb = hidden_dim (d_input 的 leading dimension)
    // ldc = output_dim (d_output 的 leading dimension)
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasHandle_t handle = config->cublas_handle;
    if (!handle) {
        printf("CRITICAL ERROR: cuBLAS handle is null. Call init() first.\n");
        return;
    }
    cublasSgemm(handle,
                CUBLAS_OP_T,  // 对权重进行转置 (逻辑上的 [output, hidden])
                CUBLAS_OP_N,  // 输入不转置 (逻辑上的 [hidden, seq])
                output_dim,   // m: 结果矩阵的行数 (output_dim)
                seq_len,      // n: 结果矩阵的列数 (seq_len)
                hidden_dim,   // k: 中间维数 (hidden_im)
                &alpha,
                d_weight, hidden_dim, // A, lda
                d_input, hidden_dim,  // B, ldb
                &beta,
                d_output, output_dim  // C, ldc
    );
  }

  void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                              const tensor::Tensor& output, int32_t group_size,
                              const tensor::Tensor& scale, const CudaConfig* config) {
    const int32_t K = weight.get_dim(0);
    const int32_t M = weight.get_dim(1);

    int32_t seq_len = 1;
    if (input.dims_size() == 2) seq_len = input.get_dim(0);

    int thread_per_block = 256;
    int block_x = (K + thread_per_block - 1) / thread_per_block;
    dim3 grid(block_x, seq_len);
    dim3 block(thread_per_block);
    
    cudaStream_t stream = config->stream;
    matmul_kernel_cu_int8<<<grid, block, 0, stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(), group_size,
        const_cast<float*>(output.ptr<float>()), M, K);
  }

}  // namespace kernel
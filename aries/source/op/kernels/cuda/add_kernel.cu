#include "add_kernel.cuh"

namespace kernel {

// 使用取模实现广播
__global__ void add_kernel_cu_fp32(int32_t size1, int32_t size2, 
                                   const float* in1, const float* in2, float* out) {
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size1) {
    return;
  }
  
  float in_val1 = in1[tid];
  // 如果 size1 == size2 (残差连接)，tid % size2 == tid，行为不变。
  // 如果 size1 > size2 (Bias加法)，tid % size2 会循环读取 Bias，实现每行加上相同的 Bias。
  float in_val2 = in2[tid % size2];
  out[tid] = in_val1 + in_val2;
}

void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);
  
  int32_t size1 = static_cast<int32_t>(input1.size());
  int32_t size2 = static_cast<int32_t>(input2.size());
  
  // 允许 size1 是 size2 的整数倍
  CHECK_LE(size2, size1) << "Input2 size must be <= Input1 size";
  CHECK_EQ(size1 % size2, 0) << "Input1 size must be a multiple of Input2 size for broadcasting";
  CHECK_EQ(size1, output.size()) << "Output size must match Input1 size";

  int32_t thread_num = 512;
  int32_t block_num = (size1 + thread_num - 1) / thread_num;
  
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    // 传入 size1 和 size2
    add_kernel_cu_fp32<<<block_num, thread_num, 0, stream_>>>(
        size1, size2, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  } else {
    add_kernel_cu_fp32<<<block_num, thread_num>>>(
        size1, size2, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  }
}

}  // namespace kernel
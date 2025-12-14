#include <cuda_runtime.h>
#include <cmath>
#include <cassert>
#include "rope_kernel.cuh"

namespace kernel {

// sin/cos 预计算 kernel
__global__ void sin_cos_calc(int rotary_dim, int max_seq_len, float base,
                             float* sin_cache, float* cos_cache) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;   // dimension index
  int pos = blockIdx.y * blockDim.y + threadIdx.y;   // token position
  if (idx >= rotary_dim || pos >= max_seq_len) return;

  // rotary_dim 必须为偶数（pairs）
  // pair_id 对应频率级别：pair 0 -> lowest freq
  int pair_id = idx / 2;
  
  float inv_freq = powf(base, -2.0f * (float)pair_id / (float)rotary_dim);
  float angle = (float)pos * inv_freq;
  long long cache_idx = (long long)pos * rotary_dim + idx; // safe 64-bit index

  sin_cache[cache_idx] = sinf(angle);
  cos_cache[cache_idx] = cosf(angle);
}

__global__ void rope_kernel_cu_fp32(const int* __restrict__ input_pos,
                                    int dim, int kv_dim, int head_size, int rotary_dim,
                                    const float* __restrict__ input_q,
                                    const float* __restrict__ input_k,
                                    const float* __restrict__ input_v,
                                    float* __restrict__ out_q,
                                    float* __restrict__ out_k,
                                    const float* __restrict__ sin_cache,
                                    const float* __restrict__ cos_cache,
                                    float* __restrict__ k_cache,
                                    float* __restrict__ v_cache,
                                    int layer_idx,
                                    int max_seq_len_cache,
                                    int seq_len) {
  int token_idx = blockIdx.x;
  int head_idx  = blockIdx.y;

  // Qwen/Llama 的 rotary_dim 通常等于 head_size (64)
  // half_dim = 32
  int half_dim = rotary_dim / 2;

  if (token_idx >= seq_len) return;

  int pos = input_pos[token_idx];
  bool pos_valid_for_cache = (pos >= 0 && pos < max_seq_len_cache);

  // 指针计算
  long long q_base = (long long)token_idx * dim + (long long)head_idx * head_size;
  long long kv_src_base = (long long)token_idx * kv_dim + (long long)head_idx * head_size;
  
  // cache 偏移
  long long layer_offset = 0;
  if (k_cache != nullptr) {
    layer_offset = (long long)layer_idx * max_seq_len_cache * kv_dim;
  }

  // 要遍历前半部分 0..31，线程 i 处理位置 i 和 i + 32
  for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
    
    // 获取 sin/cos，sin_cos_calc 是按 adjacent 生成的 (idx 0,1 是 freq0; idx 2,3 是 freq1)
    // 第 i 组频率 (freq_i) 存储在 cache 的 index = 2 * i 位置，pos 计算保持不变
    long long cache_idx = (long long)pos * rotary_dim + (2 * i); 
    float s = sin_cache[cache_idx];
    float c = cos_cache[cache_idx];

    // 读取 q (Rotary Half)
    int idx0 = i;
    int idx1 = i + half_dim; // 跨越半个 Head Dim

    float q0 = input_q[q_base + idx0];
    float q1 = input_q[q_base + idx1];

    // 旋转
    // out[i] = x[i]*cos - x[i+half]*sin
    // out[i+half] = x[i+half]*cos + x[i]*sin
    out_q[q_base + idx0] = q0 * c - q1 * s;
    out_q[q_base + idx1] = q1 * c + q0 * s;

    // 处理 k
    int num_kv_heads = kv_dim / head_size;
    if (head_idx < num_kv_heads) {
      float k0 = input_k[kv_src_base + idx0];
      float k1 = input_k[kv_src_base + idx1];

      float k0_rot = k0 * c - k1 * s;
      float k1_rot = k1 * c + k0 * s;

      // 直接把旋转后的值写进去，不改变相对位置
      if (k_cache != nullptr && pos_valid_for_cache) {
        long long token_offset = (long long)pos * kv_dim;
        long long head_offset  = (long long)head_idx * head_size;
        long long final_offset = layer_offset + token_offset + head_offset;

        k_cache[final_offset + idx0] = k0_rot;
        k_cache[final_offset + idx1] = k1_rot;

        // 搬运 v (v 不需要旋转，直接搬运)
        if (v_cache != nullptr && input_v != nullptr) {
          v_cache[final_offset + idx0] = input_v[kv_src_base + idx0];
          v_cache[final_offset + idx1] = input_v[kv_src_base + idx1];
        }
      } 
      // 处理非 cache 模式 (prefill 临时输出)
      else if (out_k != nullptr) {
          out_k[kv_src_base + idx0] = k0_rot;
          out_k[kv_src_base + idx1] = k1_rot;
      }
    }
  }
}

void rope_kernel_cu(int dim, int kv_dim, int head_size, int rotary_dim,
                    tensor::Tensor& input_q,
                    tensor::Tensor& input_k,
                    tensor::Tensor& input_v,
                    const tensor::Tensor& input_pos,
                    const tensor::Tensor& sin_cache,
                    const tensor::Tensor& cos_cache,
                    tensor::Tensor& k_cache,
                    tensor::Tensor& v_cache,
                    int layer_idx,
                    cudaStream_t stream) {
  // layout assumption: input_q/input_k/input_v are [seq_len, dim] / [seq_len, kv_dim]
  int seq_len = input_q.get_dim(0);
  int num_q_heads = dim / head_size;
  int pairs_per_head = rotary_dim / 2;

  assert(rotary_dim % 2 == 0 && "rotary_dim must be even (pairs)");

  // grid/block 设置：block.x 使用固定合理线程数，kernel 内循环遍历 pairs
  int threads_x = 128;
  dim3 block(threads_x);
  dim3 grid(seq_len, num_q_heads);

  // cache 指针与尺寸
  float* k_cache_ptr = nullptr;
  float* v_cache_ptr = nullptr;
  int max_seq_len_cache = 0;
  if (!k_cache.is_empty()) {
    k_cache_ptr = const_cast<float*>(k_cache.ptr<float>());
    v_cache_ptr = const_cast<float*>(v_cache.ptr<float>());
    max_seq_len_cache = k_cache.get_dim(1); // [layers, max_seq_len, kv_dim]
  }
  // 权重交错对应，内存布局是 [head_0, head_32, head_1, head_33, ...]
  const float* input_v_ptr = nullptr;
  if (!input_v.is_empty()) input_v_ptr = input_v.ptr<float>();
  rope_kernel_cu_fp32<<<grid, block, 0, stream>>>(
      input_pos.ptr<int32_t>(),
      dim, kv_dim, head_size, rotary_dim,
      input_q.ptr<float>(),
      input_k.ptr<float>(),
      input_v_ptr,
      input_q.ptr<float>(), // out_q (in-place)
      input_k.ptr<float>(), // out_k (in-place or writeback)
      sin_cache.ptr<float>(),
      cos_cache.ptr<float>(),
      k_cache_ptr,
      v_cache_ptr,
      layer_idx,
      max_seq_len_cache,
      seq_len);

  // launch error check (non-blocking)
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("rope_kernel_cu_fp32 launch failed: %s\n", cudaGetErrorString(err));
  }
}

void sin_cos_cache_calc_cu(int rotary_dim, int max_seq_len, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, cudaStream_t stream) {
  float rope_theta = 1000000.0f;
  dim3 threads(32, 32);
  dim3 blocks((rotary_dim + threads.x - 1) / threads.x,
              (max_seq_len + threads.y - 1) / threads.y);

  sin_cos_calc<<<blocks, threads, 0, stream>>>(
      rotary_dim, max_seq_len, rope_theta,
      const_cast<float*>(sin_cache.ptr<float>()),
      const_cast<float*>(cos_cache.ptr<float>()));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("sin_cos_calc launch failed: %s\n", cudaGetErrorString(err));
  }
}

} // namespace kernel
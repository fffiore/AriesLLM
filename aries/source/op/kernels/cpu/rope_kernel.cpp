#include "rope_kernel.h"

namespace kernel {

// Qwen2.5 默认 RoPE base frequency（与 Qwen 系列一致）
// freq = 10000^( -2*i/rotary_dim )
static inline float rope_inv_freq(int i, int rotary_dim) {
    return std::pow(10000.0f, -2.0f * i / rotary_dim);
}

void sin_cos_cache_calc_cpu(
    int rotary_dim,
    int max_seq_len,
    tensor::Tensor& sin_cache,
    tensor::Tensor& cos_cache,
    cudaStream_t stream) {
    int half_dim = rotary_dim / 2;

    float* sin_ptr = sin_cache.ptr<float>();
    float* cos_ptr = cos_cache.ptr<float>();

    std::vector<float> inv_freq(half_dim);
    for (int i = 0; i < half_dim; i++) {
        inv_freq[i] = rope_inv_freq(i, rotary_dim);
    }

    for (int pos = 0; pos < max_seq_len; pos++) {
        float* sin_row = sin_ptr + pos * half_dim;
        float* cos_row = cos_ptr + pos * half_dim;

        for (int j = 0; j < half_dim; j++) {
            float v = pos * inv_freq[j];
            sin_row[j] = std::sin(v);
            cos_row[j] = std::cos(v);
        }
    }
}

void apply_rope_pairwise(float* base_ptr, int rotary_dim, const float* sinp, const float* cosp) {
  // 假定 rotary_dim 为偶数；按 (i,i+1) 对旋转（interleaved/pairwise）
  for (int i = 0; i < rotary_dim; i += 2) {
    float x0 = base_ptr[i];
    float x1 = base_ptr[i + 1];
    float s = sinp[i];   // 注意：sin_cache layout 必须与这里一致
    float c = cosp[i];
    base_ptr[i]     = x0 * c - x1 * s;
    base_ptr[i + 1] = x0 * s + x1 * c;
  }
}

void rope_kernel_cpu(int num_q_heads, int num_kv_heads, int head_size, int rotary_dim,
                     tensor::Tensor& q, tensor::Tensor& k, const tensor::Tensor& pos,
                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                     cudaStream_t /*stream*/) {
  // q: [seq_len, num_q_heads * head_size] 或 [seq_len, num_q_heads, head_size] 取决于你的 tensor 实现
  // k: [seq_len, num_kv_heads * head_size] 或类似
  // pos: [seq_len] (int)
  // sin_cache/cos_cache: [seq_len, rotary_dim] recommended

  int q_dims = q.dims_size();
  int seq_len = q.get_dim(0);

  // pointers
  float* qptr = q.ptr<float>();
  float* kptr = k.ptr<float>();
  const float* sin_ptr = sin_cache.ptr<float>();
  const float* cos_ptr = cos_cache.ptr<float>();

  // strides (假设 q 存 layout: [seq_len, num_q_heads * head_size])
  int q_stride = num_q_heads * head_size;
  int k_stride = num_kv_heads * head_size;

  for (int t = 0; t < seq_len; ++t) {
    int pos_i = pos.index<int32_t>(t);
    const float* sinp = sin_ptr + pos_i * rotary_dim;
    const float* cosp = cos_ptr + pos_i * rotary_dim;

    // Q heads
    float* q_time = qptr + t * q_stride;
    for (int h = 0; h < num_q_heads; ++h) {
      float* q_head = q_time + h * head_size;
      apply_rope_pairwise(q_head, rotary_dim, sinp, cosp);
    }

    // K heads: map q heads -> kv heads by modulo (常用映射)
    float* k_time = kptr + t * k_stride;
    for (int h = 0; h < num_q_heads; ++h) {
      int kv_h = h % num_kv_heads; // 如果 num_kv_heads < num_q_heads
      float* k_head = k_time + kv_h * head_size;
      apply_rope_pairwise(k_head, rotary_dim, sinp, cosp);
    }
  }
}  

}// namespace kernel
#ifndef LLAMA_INFER_ROPE_KERNEL_H
#define LLAMA_INFER_ROPE_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
void sin_cos_cache_calc_cpu(int rotary_dim, int max_seq_len, tensor::Tensor& sin_cache,
                            tensor::Tensor& cos_cache, cudaStream_t stream);

void rope_kernel_cpu(int num_q_heads, int num_kv_heads, int head_size, int rotary_dim,
                     tensor::Tensor& q, tensor::Tensor& k, const tensor::Tensor& pos,
                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                     cudaStream_t stream);
}  // namespace kernel
#endif  // LLAMA_INFER_ROPE_KERNEL_H

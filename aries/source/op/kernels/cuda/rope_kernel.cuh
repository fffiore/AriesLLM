#ifndef ROPE_KERNEL_CU_CUH
#define ROPE_KERNEL_CU_CUH
#include "tensor/tensor.h"
namespace kernel {
   void rope_kernel_cu(int dim, int kv_dim, int head_size, int rotary_dim, tensor::Tensor& input_q, 
                      tensor::Tensor& input_k, tensor::Tensor& input_v, const tensor::Tensor& input_pos, 
                      const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache, tensor::Tensor& k_cache,        
                      tensor::Tensor& v_cache, int layer_idx, cudaStream_t stream);

    void sin_cos_cache_calc_cu(int rotary_dim, int max_seq_len, const tensor::Tensor& sin_cache,
                                const tensor::Tensor& cos_cache, cudaStream_t stream);

}  // namespace kernel
#endif  // ROPE_KERNEL_CU_CUH

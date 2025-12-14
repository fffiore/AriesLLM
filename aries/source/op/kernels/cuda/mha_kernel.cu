#include "mha_kernel.cuh"
#include "flash_atten.cuh"
#include "base/cuda_config.h"
#include <cub/cub.cuh>
#include <cassert>
#include <cstdio>

namespace kernel {

void mha_kernel_cu(const tensor::Tensor& pos_tensor,
                   int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                   const tensor::Tensor& mha_out,
                   const tensor::Tensor& query_tensor,
                   const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor,
                   const tensor::Tensor& value_cache_tensor,
                   base::DeviceType device_type,
                   CudaConfig* config) {

    (void)score_tensor;
    (void)device_type;

    // runtime safety: ensure head_size fits kernel's compile-time HEAD_SIZE
    assert(head_size <= HEAD_SIZE);
    if (head_size > HEAD_SIZE) {
        fprintf(stderr, "Error: head_size (%d) > HEAD_SIZE (%d)\n", head_size, HEAD_SIZE);
        return;
    }
    float* Q = const_cast<float*>(query_tensor.ptr<float>());
    float* O = const_cast<float*>(mha_out.ptr<float>());
    float* K_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
    float* V_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

    // stride to this layer's KV cache region
    int32_t layer_offset = layer_index * seq_len * kv_dim;
    float* K_layer = K_cache + layer_offset;
    float* V_layer = V_cache + layer_offset;

    int32_t current_seq_len = query_tensor.get_dim(0);
    float sm_scale = 1.0f / sqrtf(float(head_size));
    cudaStream_t stream = config->stream;

    if (current_seq_len > 1) {
        // grid/block: grid.x unused, grid.y heads, grid.z q-blocks
        dim3 grid(1, head_num, (current_seq_len + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
        dim3 block(BLOCK_THREADS);

        // shared memory layout must match kernel's expectation.
        // we allocate using HEAD_SIZE to ensure alignment/safety:
        // sQ: BLOCK_SIZE_M * HEAD_SIZE
        // sK: BLOCK_SIZE_N * HEAD_SIZE
        // sV: BLOCK_SIZE_N * HEAD_SIZE
        size_t smem_size = (size_t)(BLOCK_SIZE_M * HEAD_SIZE + 2 * BLOCK_SIZE_N * HEAD_SIZE) * sizeof(float);
        flash_attn_prefill_kernel<<<grid, block, smem_size, stream>>>(
            Q, K_layer, V_layer, O, 0,
            head_num, kv_mul, head_size, current_seq_len, kv_dim, seq_len, sm_scale
        );

        // check kernel launch error
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "flash_attn_prefill_kernel launch failed: %s\n", cudaGetErrorString(err));
            return;
        }
    } else {
        const int32_t* pos_ptr = pos_tensor.ptr<int32_t>();
        // grid: one block per head; block threads must match BLOCK_THREADS
        dim3 grid(head_num);
        dim3 block(BLOCK_THREADS);
        // shared memory: need at least HEAD_SIZE floats (kernel reuses this buffer)
        size_t smem_size = (size_t)HEAD_SIZE * sizeof(float);

        standard_decode_kernel<<<grid, block, smem_size, stream>>>(
            Q, K_layer, V_layer, O, pos_ptr,
            head_num, kv_mul, head_size, kv_dim, sm_scale
        );

        // check kernel launch error
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "standard_decode_kernel launch failed: %s\n", cudaGetErrorString(err));
            return;
        }
    }
}

} // namespace kernel

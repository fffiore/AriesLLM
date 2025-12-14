#include "topk_kernel.cuh"
#include <cub/cub.cuh>

namespace kernel {
    // 初始化索引 [0, 1, 2, ... N]
    __global__ void fill_indices_kernel(int* indices, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            indices[idx] = idx;
        }
    }

    void topk_kernel_cu(const float* logits, 
                        int size, 
                        int k, 
                        float* output_vals, 
                        int* output_indices, 
                        void* stream,
                        int* temp_indices_buffer, // 用作 Output Values (Sorted Indices)
                        float* temp_logits_buffer // 用作 Output Keys (Sorted Logits)
    ) {
        cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
        
        // 分配 input indices [0, 1, ... N] 的临时内存
        // 因为 CUB SortPairs 需要独立的 Input 和 Output Buffer
        int* d_indices_in = nullptr;
        // 使用 cudaMallocAsync (CUDA 11.2+) 减少开销，或者用 cudaMalloc
        cudaMalloc(&d_indices_in, size * sizeof(int));
        
        // 填充索引
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        fill_indices_kernel<<<blocks, threads, 0, cu_stream>>>(d_indices_in, size);

        // CUB Radix Sort (降序)
        // Keys:   Logits (In) -> temp_logits_buffer (Out)
        // Values: d_indices_in (In) -> temp_indices_buffer (Out)
        
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        
        // 确定临时存储大小
        cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
            logits, temp_logits_buffer,
            d_indices_in, temp_indices_buffer,
            size, 0, sizeof(float)*8, cu_stream);
            
        // 分配 CUB 临时存储
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        
        // 执行排序
        cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
            logits, temp_logits_buffer,
            d_indices_in, temp_indices_buffer,
            size, 0, sizeof(float)*8, cu_stream);
            
        // 拷贝 Top K 结果回 CPU
        // 此时 temp_logits_buffer 和 temp_indices_buffer 已经是排好序的
        cudaMemcpyAsync(output_vals, temp_logits_buffer, k * sizeof(float), cudaMemcpyDeviceToHost, cu_stream);
        cudaMemcpyAsync(output_indices, temp_indices_buffer, k * sizeof(int), cudaMemcpyDeviceToHost, cu_stream);
        
        // 在释放临时内存前同步，或者等待 Stream 完成
        // 因为 output_vals 需要在函数返回后立即可用，这里必须同步
        cudaStreamSynchronize(cu_stream);
        
        // 释放临时内存
        cudaFree(d_indices_in);
        cudaFree(d_temp_storage);
    }

} // namespace kernel
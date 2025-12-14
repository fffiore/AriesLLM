#ifndef TOPK_KERNEL_CUH
#define TOPK_KERNEL_CUH
#include <cuda_runtime.h>
#include <cstdint>

namespace kernel {
    /*
        @brief GPU TopK 实现
        @param logits 模型输出的原始 Logits 指针 (Device Memory)
        @param size 词表大小 (vocab_size)
        @param k 需要选取的 Top K
        @param output_vals 输出的 Top K 分数 (Host Memory)
        @param output_indices 输出的 Top K 索引 (Host Memory)
        @param stream CUDA 流
        @param temp_indices_buffer 临时的索引缓存 (Device Memory, 大小需为 size * sizeof(int))
        @param temp_logits_buffer 临时的Logits缓存 (Device Memory, 大小需为 size * sizeof(float))
    */
    void topk_kernel_cu(const float* logits, 
                    int size, 
                    int k, 
                    float* output_vals, 
                    int* output_indices, 
                    void* stream,
                    int* temp_indices_buffer,
                    float* temp_logits_buffer);

}  // namespace kernel
#endif
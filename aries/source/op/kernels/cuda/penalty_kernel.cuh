#ifndef PENALTY_KERNEL_CUH
#define PENALTY_KERNEL_CUH
#include <cuda_runtime.h>
#include <cstdint>

namespace kernel {
    void penalty_kernel_cu(float* logits, const int* d_history, int history_len, 
                            float penalty, int vocab_size, cudaStream_t stream);
}
#endif
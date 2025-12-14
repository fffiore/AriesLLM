#include <cuda_runtime.h>

namespace kernel {

__global__ void repetition_penalty_kernel(float* logits, 
                                          const int* history, 
                                          int history_len, 
                                          float penalty, 
                                          int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= history_len) return;

    // 获取历史中的 token id
    int token_id = history[idx];
    // 防止越界
    if (token_id < 0 || token_id >= vocab_size) return;
    // 修改对应的 logit
    float score = logits[token_id];
    // apply penalty
    if (score < 0) {
        logits[token_id] = score * penalty;
    } else {
        logits[token_id] = score / penalty;
    }
}

void penalty_kernel_cu(float* logits, 
                           const int* d_history, 
                           int history_len, 
                           float penalty, 
                           int vocab_size, 
                           cudaStream_t stream) {
    // 每个线程处理一个历史 token，history_len 通常远小于 vocab_size (几千 vs 15万)
    // 直接修改对应的 logits 位置
    int threads = 256;
    int blocks = (history_len + threads - 1) / threads;
    repetition_penalty_kernel<<<blocks, threads, 0, stream>>>(
        logits, d_history, history_len, penalty, vocab_size
    );
}

} // namespace kernel
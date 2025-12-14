#include "sampler/sampling_sampler.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <glog/logging.h>
#include "../op/kernels/cuda/topk_kernel.cuh"
#include "../op/kernels/cuda/penalty_kernel.cuh"

namespace sampler {

SamplingSampler::SamplingSampler(base::DeviceType device_type, SamplerConfig config)
    : Sampler(device_type), config_(config), mt_(config.seed) {}

SamplingSampler::~SamplingSampler() {
    free_cuda_memory();
    if (d_history_) cudaFree(d_history_);
}

void SamplingSampler::free_cuda_memory() {
    if (d_temp_logits_) {
        cudaFree(d_temp_logits_);
        d_temp_logits_ = nullptr;
    }
    if (d_temp_indices_) {
        cudaFree(d_temp_indices_);
        d_temp_indices_ = nullptr;
    }
    // 释放 top-k 专用的临时 value buffer
    if (d_topk_val_buf_) {
        cudaFree(d_topk_val_buf_);
        d_topk_val_buf_ = nullptr;
    }
}

size_t SamplingSampler::sample(const float* logits, size_t vocab_size, 
                               const std::vector<int32_t>& history, 
                               void* stream) {
    std::vector<std::pair<float, int>> candidates;
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);

    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        if (vocab_size > vocab_size_allocated_) {
            free_cuda_memory();
            // 存储可修改的 logits (input to top-k)
            cudaMalloc(&d_temp_logits_, vocab_size * sizeof(float));
            // 存储 top-p 排序后的 indices
            cudaMalloc(&d_temp_indices_, vocab_size * sizeof(int));
            // 存储 topk-k 排序后的 values (避免 nullptr crash)
            cudaMalloc(&d_topk_val_buf_, vocab_size * sizeof(float));
            vocab_size_allocated_ = vocab_size;
        }

        // 拷贝 logits 到 GPU (用于 in-place 修改)
        cudaMemcpyAsync(d_temp_logits_, logits, vocab_size * sizeof(float), cudaMemcpyDeviceToDevice, cu_stream);
        float* mutable_logits = d_temp_logits_; 

        // GPU repetition penalty
        if (config_.repetition_penalty > 1.0f + 1e-6f && !history.empty()) {
            if (history.size() > history_allocated_) {
                if (d_history_) cudaFree(d_history_);
                size_t alloc_size = std::max(history.size() * 2, (size_t)1024);
                cudaMalloc(&d_history_, alloc_size * sizeof(int));
                history_allocated_ = alloc_size;
            }
            
            cudaMemcpyAsync(d_history_, history.data(), history.size() * sizeof(int), cudaMemcpyHostToDevice, cu_stream);
            kernel::penalty_kernel_cu(mutable_logits, d_history_, history.size(), config_.repetition_penalty, 
                                        vocab_size, cu_stream);
        }

        // GPU top-k
        int gpu_fetch_k = config_.top_k > 0 ? std::max((int)config_.top_k, 128) : 128;
        if (gpu_fetch_k > vocab_size) gpu_fetch_k = vocab_size;

        std::vector<float> top_vals(gpu_fetch_k);
        std::vector<int> top_indices(gpu_fetch_k);
        
        // 传入 d_topk_val_buf_ 作为临时存储，防止 nullptr 导致 kernel 内部 crash
        kernel::topk_kernel_cu(mutable_logits, vocab_size, gpu_fetch_k, 
                               top_vals.data(), top_indices.data(), 
                               cu_stream, 
                               d_temp_indices_, d_topk_val_buf_); 

        // 同步并回传 CPU
        cudaStreamSynchronize(cu_stream);
        candidates.reserve(gpu_fetch_k);
        for (int i = 0; i < gpu_fetch_k; ++i) {
            candidates.push_back({top_vals[i], top_indices[i]});
        }
    }
    else {
        std::vector<float> cpu_logits(vocab_size);
        std::copy(logits, logits + vocab_size, cpu_logits.begin());

        // CPU repetition penalty
        if (config_.repetition_penalty > 1.0f + 1e-6f) {
             for (int token_id : history) {
                 if (token_id >= 0 && token_id < vocab_size) {
                     float& val = cpu_logits[token_id];
                     val = (val < 0) ? (val * config_.repetition_penalty) : (val / config_.repetition_penalty);
                 }
             }
        }

        candidates.reserve(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            candidates.push_back({cpu_logits[i], (int)i});
        }
    }
    
    // temperature 处理，先 apply temp 再找 max
    float temp = config_.temperature > 0 ? config_.temperature : 1.0f;
    float max_logit = -1e9f;
    
    for (auto& pair : candidates) {
        pair.first /= temp; // apply temperature
        if (pair.first > max_logit) max_logit = pair.first;
    }

    // softmax (exp & sum)
    float sum_prob = 0.0f;
    for (auto& pair : candidates) {
        pair.first = std::exp(pair.first - max_logit);
        sum_prob += pair.first;
    }

    // normalize
    for (auto& pair : candidates) {
        pair.first /= sum_prob;
    }

    // sort (probability descending)
    // GPU top-k 出来通常是有序的，但为了保险起见，CPU 还是排一下
    // 数据量只有 128 ~ 256，耗时忽略不计
    std::sort(candidates.begin(), candidates.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b){
        return a.first > b.first;
    });

    // top-p sampling
    if (config_.top_p > 0.0f && config_.top_p < 1.0f) {
        float cum_prob = 0.0f;
        size_t cutoff = 0;
        for (size_t i = 0; i < candidates.size(); ++i) {
            cum_prob += candidates[i].first;
            cutoff = i;
            if (cum_prob >= config_.top_p) {
                break;
            }
        }
        candidates.resize(cutoff + 1);
    }

    // random selection
    std::vector<float> final_probs;
    final_probs.reserve(candidates.size());
    for (const auto& p : candidates) final_probs.push_back(p.first);
    
    std::discrete_distribution<> dist(final_probs.begin(), final_probs.end());
    int idx = dist(mt_);

    return candidates[idx].second;
}

} // namespace sampler
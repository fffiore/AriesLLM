#ifndef LLAMA_INFER_SAMPLING_SAMPLER_H
#define LLAMA_INFER_SAMPLING_SAMPLER_H
#include <base/base.h>
#include "sampler.h"
#include <cuda_runtime.h>

namespace sampler {
    class SamplingSampler: public Sampler {
        public:
            explicit SamplingSampler(base::DeviceType device_type, SamplerConfig config);
            ~SamplingSampler();
            void free_cuda_memory();
            /*
                @brief 采样函数
                @param logits 模型输出的原始分数（未经过 softmax）
                @param size 词表大小（vocab_size）
                @param history 已经生成的 token 历史（用于重复惩罚）
                @param stream CUDA 流（如果需要）
                @return 选中的 token id
            */
            size_t sample(const float* logits, size_t size, const std::vector<int32_t>& history, void* stream = nullptr) override;
        private:
            SamplerConfig config_;
            std::mt19937 mt_;       // 随机数生成器
            // GPU 临时缓存
            float* d_temp_logits_ = nullptr;
            int* d_temp_indices_ = nullptr;
            float* d_topk_val_buf_ = nullptr;
            size_t vocab_size_allocated_ = 0;
            int* d_history_ = nullptr;        // GPU 上的历史记录 buffer
            size_t history_allocated_ = 0;    // 记录已分配的大小
    };
}
#endif
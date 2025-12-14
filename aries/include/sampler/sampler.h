#ifndef LLAMA_INFER_SAMPLER_H
#define LLAMA_INFER_SAMPLER_H
#include <cstddef>
#include <cstdint>
#include <random>
namespace sampler {
  // 采样配置
  struct SamplerConfig {
    float temperature = 1.0f;         // 温度
    float repetition_penalty = 1.0f;  // 重复惩罚 (> 1.0 代表惩罚)
    size_t top_k = 0;                 // Top-K (0 代表不使用)
    float top_p = 0.0f;               // Top-P (0.0 代表不使用)
    // 随即种子，为了复现
    unsigned long long seed = std::random_device{}();
  };
  class Sampler {
    public:
      explicit Sampler(base::DeviceType device_type) : device_type_(device_type) {}
      virtual ~Sampler() = default;
      virtual size_t sample(const float* logits, size_t size, const std::vector<int32_t>& history, void* stream = nullptr)  = 0;

    protected:
      base::DeviceType device_type_;
  };
}  // namespace sampler
#endif  // LLAMA_INFER_SAMPLER_H

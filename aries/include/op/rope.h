#ifndef KUIPER_INCLUDE_OP_ROPE_H_
#define KUIPER_INCLUDE_OP_ROPE_H_
#include "layer.h"
#pragma once

namespace op {

class RoPELayer : public Layer {
  public:
    RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t head_size,
              int32_t rotary_dim, int32_t num_kv_heads);
    virtual ~RoPELayer() = default;
    base::Status forward() override;
    base::Status check() const override;
    base::Status forward(tensor::Tensor& input_q, tensor::Tensor& input_k, tensor::Tensor& input_v, 
                       const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache, 
                       const tensor::Tensor& cos_cache, tensor::Tensor& k_cache, 
                       tensor::Tensor& v_cache, int32_t layer_idx);

  private:
    int32_t dim_;
    int32_t kv_dim_;
    int32_t head_size_;
    int32_t rotary_dim_;
    int32_t num_kv_heads_;
};

}  // namespace op

#endif


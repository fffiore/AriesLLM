#include "op/rope.h"
#include <cmath>
#include "kernels/cpu/rope_kernel.h"
#include "kernels/kernels_interface.h"

namespace op {

RoPELayer::RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t head_size,
                     int32_t rotary_dim, int32_t num_kv_heads)
    : Layer(device_type, LayerType::kLayerRoPe, "RoPe"),
      dim_(dim),
      kv_dim_(kv_dim),
      head_size_(head_size),
      rotary_dim_(rotary_dim),
      num_kv_heads_(num_kv_heads) {
  reset_input_size(5);
  reset_output_size(1);
}

base::Status RoPELayer::check() const {
  // 获取当前输入的 seq_len
  const auto& input_q = get_input(0);
  int32_t seq_len = input_q.get_dim(0); // 获取行数

  // 长度必须等于 seq_len (prefill 时 > 1)
  // 设备必须与 layer 一致 (CUDA kernel 只能读 GPU 内存)
  auto status = check_tensor_with_dim(get_input(2), device_type_, 
                                      base::DataType::kDataTypeInt32, seq_len);
  if (!status) {
    LOG(ERROR) << "The pos tensor error in RoPE. Expected len: " << seq_len;
    return status;
  }

  // q 检查
  int q_dims = input_q.dims_size();
  if (q_dims < 1) return base::error::InvalidArgument("Q tensor dims invalid");
  int q_last = input_q.get_dim(q_dims - 1);
  if (q_last % head_size_ != 0) {
    return base::error::InvalidArgument("Q shape not match head_size");
  }

  // k 检查
  const auto& k = get_input(1);
  // 确保 k 的行数也等于 seq_len
  if (k.get_dim(0) != seq_len) {
      return base::error::InvalidArgument("K tensor seq_len mismatch");
  }
  
  int k_dims = k.dims_size();
  int k_last = k.get_dim(k_dims - 1);
  if (k_last != kv_dim_ && k_last % head_size_ != 0) {
    return base::error::InvalidArgument("K shape mismatch");
  }

  // sin/cos 检查，prefill 时只要 cache 足够大即可
  const auto& sin_cache = get_input(3);
  int sin_dims = sin_cache.dims_size();
  int sin_last = sin_cache.get_dim(sin_dims - 1);
  if (sin_last < rotary_dim_) {
    return base::error::InvalidArgument("sin_cache layout mismatch rotary_dim");
  }
  return base::error::Success();
}

base::Status RoPELayer::forward() {
  return base::error::InternalError("RoPELayer requires arguments for forward. Use the overloaded version.");
}

base::Status RoPELayer::forward(tensor::Tensor& input_q, tensor::Tensor& input_k, tensor::Tensor& input_v,
                                const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                                const tensor::Tensor& cos_cache, tensor::Tensor& k_cache,
                                tensor::Tensor& v_cache, int32_t layer_idx) {
  
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }

  kernel::get_rope_kernel(device_type_)(
      dim_, kv_dim_, head_size_, rotary_dim_,   
      input_q, input_k, input_v,     
      input_pos, sin_cache, cos_cache,
      k_cache, v_cache, layer_idx,   
      cuda_config_ ? cuda_config_->stream : nullptr);

  return base::error::Success();
}

}  // namespace op
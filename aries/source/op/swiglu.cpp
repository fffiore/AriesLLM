#include "op/swiglu.h"
#include "kernels/cpu/swiglu_kernel.h"
#include "kernels/kernels_interface.h"
#include "op/layer.h"

namespace op {
SwiGLULayer::SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim)
    : Layer(device_type, op::LayerType::kLayerSwiGLU, "SwiGLU"), hidden_dim_(hidden_dim) {
  reset_input_size(2);
  reset_output_size(1);
}

base::Status SwiGLULayer::check() const {
  base::Status status;
  status = check_tensor(get_input(0), device_type_, data_type_);
  if (!status) {
    LOG(ERROR) << "The input tensor 1 error in the swiglu layer.";
    return status;
  }

  status = check_tensor(get_input(1), device_type_, data_type_);
  if (!status) {
    LOG(ERROR) << "The input tensor 2 error in the swiglu layer.";
    return status;
  }

  status = check_tensor(get_output(0), device_type_, data_type_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the swiglu layer.";
    return status;
  }

  // 检查最后一维 (Hidden Dim)
  // SwiGLU 通常作用于 hidden_dim 维度
  const tensor::Tensor& input1 = get_input(0);
  int32_t last_dim = input1.get_dim(input1.dims_size() - 1);
  if (last_dim != hidden_dim_) {
      LOG(ERROR) << "The input tensor dim mismatch in swiglu layer. Expected last dim: " 
                 << hidden_dim_ << ", but got: " << last_dim;
      return base::error::InvalidArgument("SwiGLU dimension mismatch");
  }

  // 检查大小一致性 (Input1 == Input2 == Output)
  if (get_input(0).size() != get_input(1).size()) {
      LOG(ERROR) << "The input tensors size mismatch in swiglu layer.";
      return base::error::InvalidArgument("SwiGLU inputs size mismatch");
  }
  if (get_input(0).size() != get_output(0).size()) {
      LOG(ERROR) << "The output tensor size mismatch in swiglu layer.";
      return base::error::InvalidArgument("SwiGLU output size mismatch");
  }

  return base::error::Success();
}

base::Status SwiGLULayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  auto input1 = this->get_input(0);
  auto input2 = this->get_input(1);
  auto output = this->get_output(0);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  kernel::get_swiglu_kernel(device_type_)(input1, input2, output,
                                          cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

}  // namespace op
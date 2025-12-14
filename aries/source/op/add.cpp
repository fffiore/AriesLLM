#include "op/add.h"
#include "kernels/kernels_interface.h"

namespace op {
VecAddLayer::VecAddLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerAdd, "Add") {
  reset_input_size(2);
  reset_output_size(1);
}

base::Status VecAddLayer::check() const {
  tensor::Tensor input1 = this->get_input(0);
  tensor::Tensor input2 = this->get_input(1);
  tensor::Tensor output = this->get_output(0);
  base::Status status;
  status = check_tensor(input1, device_type_, data_type_);
  if (!status) {
    LOG(ERROR) << "The input tensor 1 error in the add layer.";
    return status;
  }

  status = check_tensor(input2, device_type_, data_type_);
  if (!status) {
    LOG(ERROR) << "The input tensor 2 error in the add layer.";
    return status;
  }

  status = check_tensor(output, device_type_, data_type_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the add layer.";
    return status;
  }

  // 检查 Size 兼容性 (支持广播)
  // 逻辑必须与 kernels/cuda/add_kernel.cu 中的检查保持一致
  int32_t size1 = input1.size();
  int32_t size2 = input2.size();
  int32_t size_out = output.size();

  if (size1 % size2 != 0) {
      LOG(ERROR) << "Input tensor size mismatch for AddLayer broadcasting: " 
                 << "input1 size: " << size1 << ", input2 size: " << size2;
      return base::error::InvalidArgument("AddLayer input size mismatch");
  }

  if (size1 != size_out) {
      LOG(ERROR) << "Output tensor size mismatch in AddLayer: "
                 << "input1 size: " << size1 << ", output size: " << size_out;
      return base::error::InvalidArgument("AddLayer output size mismatch");
  }

  return base::error::Success();
}

base::Status VecAddLayer::forward() {
  auto status = this->check();
  if (!status) {
    return status;
  }
  auto input1 = this->get_input(0);
  auto input2 = this->get_input(1);
  auto output = this->get_output(0);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  kernel::get_add_kernel(device_type_)(input1, input2, output,
                                       cuda_config_ ? cuda_config_->stream : nullptr);
  return base::error::Success();
}

}  // namespace op
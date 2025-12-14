#include "op/matmul.h"
#include "kernels/cpu/matmul_kernel.h"
#include "kernels/kernels_interface.h"

namespace op {
  MatmulLayer::MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1,
                          bool is_quant_layer, bool has_bias)
      : LayerParam(device_type, LayerType::kLayerMatmul, is_quant_layer, "Matmul"),
        dim0_(dim0),
        dim1_(dim1),
        has_bias_(has_bias) {
    reset_input_size(1);
    reset_output_size(1);
    reset_weight_size(1);
    if (has_bias_) {
      bias_.resize(1);
    }
  }

  base::Status MatmulLayer::check() const {
    // check input
    auto status = check_tensor(get_input(0), device_type_, data_type_);
    if (!status) {
      LOG(ERROR) << "The input tensor error in the matmul layer.";
      return status;
    }

    // check input dim
    const tensor::Tensor& input_tensor = get_input(0);
    // 兼容 [batch, seq, dim] 和 [seq, dim]
    int32_t input_last_dim = input_tensor.get_dim(input_tensor.dims_size() - 1);
    if (input_last_dim != dim1_) {
        LOG(ERROR) << "The input tensor dim error. Expect last dim: " << dim1_ 
                  << ", but got: " << input_last_dim;
        return base::error::InvalidArgument("Input tensor dim mismatch");
    }

    // check weight
    if (!is_quant_layer_) {
      status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim0_, dim1_);
    } else {
      status = check_tensor_with_dim(get_weight(0), device_type_, base::DataType::kDataTypeInt8,
                                    dim0_, dim1_);
    }
    if (!status) {
      LOG(ERROR) << "The weight tensor error in the matmul layer.";
      return status;
    }

    // check scales (quant only)
    if (is_quant_layer_) {
      // 检查 scales 是否为空且设备类型正确
      if (scales_.is_empty()) {
          return base::error::InvalidArgument("The scale tensor is empty in quant matmul layer.");
      }
      if (scales_.device_type() != device_type_) {
          return base::error::InvalidArgument("The scale tensor device type mismatch.");
      }
    }

    // check output
    status = check_tensor(get_output(0), device_type_, data_type_);
    if (!status) {
      LOG(ERROR) << "The output tensor error in the matmul layer.";
      return status;
    }
    
    // 简单检查 output 最后一维，避免复杂的 shape 检查影响性能
    const tensor::Tensor& output_tensor = get_output(0);
    int32_t output_last_dim = output_tensor.get_dim(output_tensor.dims_size() - 1);
    if (output_last_dim != dim0_) {
      return base::error::InvalidArgument("Output tensor dim mismatch");
    }

    return base::error::Success();
  }

  base::Status MatmulLayer::forward() {
    // 在 CUDA graph 捕获阶段，CPU 开销不敏感
  #ifndef NDEBUG
    auto status = check();
    if (!status) {
      return status;
    }
  #endif

    if (device_type_ == base::DeviceType::kDeviceCUDA) {
      CHECK(cuda_config_ != nullptr);
    }

    // Matmul 计算，output 指针在 graph 模式下必须固定指向 temp buffer
    if (is_quant_layer_) {
      kernel::get_matmul_kernel_quant8(device_type_)(get_input(0), get_weight(0), get_output(0),
                                                    group_size_, scales_,
                                                    cuda_config_ ? cuda_config_.get() : nullptr);
    } else {
      kernel::get_matmul_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), 1.f,
                                              cuda_config_ ? cuda_config_.get() : nullptr);
    }

    // Bias Add
    if (has_bias_) {
      kernel::get_add_kernel(device_type_)(get_output(0), get_bias(0), get_output(0),
                                              cuda_config_ ? cuda_config_->stream : nullptr);
    }

    return base::error::Success();
  }

  base::Status MatmulLayer::set_bias(int32_t idx, int32_t& dim, const void* bias_ptr,
                                    base::DeviceType device_type) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, bias_.size());
    CHECK_NE(bias_ptr, nullptr);

    size_t size = dim * sizeof(float);
    std::shared_ptr<base::Buffer> buffer =
        std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(bias_ptr), true);
    
    // 允许在 CPU 初始化时指定为 GPU，buffer 不会立即分配 GPU 内存，但在 assign 时会处理
    if (device_type != base::DeviceType::kDeviceUnknown) {
      buffer->set_device_type(device_type);
    }

    if (!is_quant_layer_) {
      tensor::Tensor bias(base::DataType::kDataTypeFp32, dim);
      bias.set_device_type(device_type);
      CHECK(bias.assign(buffer));
      bias_.at(idx) = bias;
    } else {
      tensor::Tensor bias(base::DataType::kDataTypeInt8, dim);
      bias.set_device_type(device_type);
      CHECK(bias.assign(buffer)); // 拷贝 Bias 数据到 Device
      bias_.at(idx) = bias;

      const int32_t bias_size = static_cast<int32_t>(bias.size());
      CHECK(bias_size % group_size_ == 0);

      // 计算 scale 的位置和大小
      int32_t scale_nums = bias_size / group_size_;
      float* host_scale_ptr = reinterpret_cast<float*>((int8_t*)bias_ptr + bias_size);
      size_t scale_byte_size = scale_nums * sizeof(float);

      // 创建 Buffer 并 assign，让 tensor 内部处理内存分配和拷贝。
      std::shared_ptr<base::Buffer> scale_buffer = 
          std::make_shared<base::Buffer>(scale_byte_size, nullptr, host_scale_ptr, true);
      
      // 如果是 GPU 模式，buffer 标记为 GPU，assign 时会自动 cudaMemcpy
      if (device_type != base::DeviceType::kDeviceUnknown) {
          scale_buffer->set_device_type(device_type);
      }

      scales_ = tensor::Tensor(base::DataType::kDataTypeFp32, scale_nums);
      scales_.set_device_type(device_type);
      CHECK(scales_.assign(scale_buffer));
    }

    return base::error::Success();
  }

  tensor::Tensor& MatmulLayer::get_bias(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, bias_.size());
    return bias_.at(idx);
  }

  const tensor::Tensor& MatmulLayer::get_bias(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, bias_.size());
    return bias_.at(idx);
  }

  void MatmulLayer::to_cuda() {
    LayerParam::to_cuda();
    if (has_bias_) {
      for (auto& bias : bias_) {
        bias.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
      }
    }
    // scales 也可以转到 CUDA
    if (is_quant_layer_ && !scales_.is_empty()) {
        scales_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }

}  // namespace op
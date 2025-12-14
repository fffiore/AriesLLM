#ifndef KUIPER_INCLUDE_MODEL_LLAMA_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_H_
#include <base/cuda_config.h>
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"
namespace model {

struct Qwen2Layers {
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> rope_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::Layer> mha_layer_;

  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;

  std::vector<std::shared_ptr<op::Layer>> w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;
  std::shared_ptr<op::Layer> cls_layer_;

  std::shared_ptr<op::Layer> embedding_layer_;

  void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};

class Qwen2Model : public Model {
 public:
  explicit Qwen2Model(base::TokenizerType tokenizer_type, std::string token_path,
                      std::string model_path, bool is_quant_model);

  base::Status init(base::DeviceType device_type) override;

  base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       const std::vector<int32_t>& history, bool is_prompt, int& next) const override;

  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       int& next) const override;

  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;
  void set_cuda_graph(bool enable) {enable_cuda_graph_ = enable;  }

 private:
  void init_mem() override;
  base::Status create_layers() override;
  void create_param_layers() override;
  void create_nonparam_layers() override;
  void create_param_quant_layers() override;
  void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;
  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;
  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;
  // CPU pos 给 slice_kv_cache 使用，GPU pos 给 RoPE 使用
  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;
  void cls_logits(const tensor::Tensor& input) const;
  int32_t post_processing(const tensor::Tensor& pos, const std::vector<int32_t>& history, bool is_prompt) const override;

 private:
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
  std::unique_ptr<Qwen2Layers> qwen_layers_;
  int32_t max_position_embeddings_ = 0;
  int32_t rotary_dim_ = 0;
  mutable int prompt_len_ = 0;
  // CUDA graph 相关
  bool enable_cuda_graph_ = false;
  mutable bool graph_captured_ = false;    // mutable 允许在 const forward 中修改
  mutable cudaGraph_t graph_ = nullptr;
  mutable cudaGraphExec_t graph_exec_ = nullptr;
  // 用于 decode 阶段的持久化 GPU 位置 tensor，避免每次 forward 都申请内存，graph 地址固定
  mutable tensor::Tensor pos_tensor_cu_;
  // 临时 buffer，用于存放 mat_mul 的结果，不能直接写进 kv cache
  mutable tensor::Tensor temp_kv_storage_;
  mutable tensor::Tensor pos_ptr_;
};
}  // namespace model

#endif
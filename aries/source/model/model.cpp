#include "model/model.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
namespace model {
Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type, std::string token_path,
             std::string model_path, bool is_quant_model)
    : tokenizer_type_(tokenizer_type),
      model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)),
      is_quant_model_(is_quant_model) {}

base::ModelType Model::model_type() const { return model_type_; }

const std::string& Model::token_path() const { return token_path_; }

const std::string& Model::model_path() const { return model_path_; }

base::Status Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
  }
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
  }
  buffers_.insert({buffer_idx, tensor});
  return base::error::Success();
}

tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
  return buffers_.at(buffer_idx);
}

const tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) const {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}

base::Status Model::read_model_file() {
  using namespace base;
  if (model_path_.empty()) {
    return error::PathNotValid("Failed to open the weight file, the model path is empty!");
  }
  
  // 打开文件描述符 (用于 mmap)
  int32_t fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    return error::PathNotValid("Failed to open the weight file " + model_path_ +
                               " may be the path does not exist!");
  }

  // 打开 FILE 指针 (用于读取 header)
  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    close(fd);
    return error::PathNotValid("Failed to open the file. The path may be invalid.");
  }
  uint32_t magic = 0;
  ModelConfig config = {};
  size_t header_offset = 0; // 权重数据的起始位置

  // 尝试读取前 4 字节
  if (fread(&magic, sizeof(uint32_t), 1, file) != 1) {
      fclose(file);
      close(fd);
      return error::ModelParseError("Failed to read magic number.");
  }

  if (magic == 0x616b3432) {
    // 格式：[Magic(4)] [Version(4)] [Config(7*4)] [Flags...] ... [Padding to 256] [Weights...]
    int32_t version = 0;
    fread(&version, sizeof(int32_t), 1, file);
    
    // 读取 config (7 个 int)
    // 对应 export.py 中的 struct.pack('iiiiiii', ...)
    if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
        fclose(file);
        close(fd);
        return error::ModelParseError("Failed to read config from new format header.");
    }

    // header 固定填充到 256 字节，权重数据从 256 字节处开始
    header_offset = 256;
    // 如果是量化模型，version 2 会在 header 后半部分写入 group_size，所有 header 信息都在 256 字节内。
    // 如果需要读取 group_size，可以通过 fseek 读取。
    if (is_quant_model_) {
        // config 已读完，剩下的在 256 字节内
    }
  } 
  else {
    // Magic 不匹配，说明文件开头直接就是 config (dim)
    // 回退文件指针到开头
    fseek(file, 0, SEEK_SET);
    
    if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
        fclose(file);
        close(fd);
        return error::ModelParseError("Failed to retrieve the configuration from legacy file.");
    }
    
    // 偏移量计算
    header_offset = sizeof(ModelConfig);
    if (is_quant_model_) {
      if (fread(&group_size_, sizeof(int32_t), 1, file) != 1) {
        fclose(file);
        close(fd);
        return error::ModelParseError("Failed to retrieve group size from legacy file.");
      }
      header_offset += sizeof(int32_t);
    }
  }

  // 根据 Config 初始化模型参数
  auto gen_status = generate_model_infos(config);
  if (!gen_status) {
    fclose(file);
    close(fd);
    return gen_status;
  }

  // 准备 ModelData 对象
  if (!is_quant_model_) {
    raw_model_data_ = std::make_shared<RawModelDataFp32>();
  } else {
    raw_model_data_ = std::make_shared<RawModelDataInt8>();
  }

  // 获取文件大小
  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    fclose(file);
    close(fd);
    return error::ModelParseError("Failed to retrieve file size.");
  }
  raw_model_data_->file_size = sb.st_size;
  raw_model_data_->fd = fd;

  // 执行 mmap 映射，将整个文件映射到内存
  raw_model_data_->data =
      mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0);

  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    fclose(file);
    // fd 由 RawModelData 析构或这里手动关闭
    return error::ModelParseError("Failed to map the weight file " + model_path_ + " into memory.");
  }

  // 设置权重数据的指针偏移，使用 calculate 出来的 header_offset (256 或 sizeof(Config))
  raw_model_data_->weight_data = static_cast<int8_t*>(raw_model_data_->data) + header_offset;

  if (raw_model_data_ == nullptr) {
    return error::ModelParseError("Raw model data is null.");
  }
  
  // 关闭 FILE*，但保留 fd (mmap 需要 fd 有效，或者 mmap 后 fd 可以关闭取决于 OS，通常保留在 struct 中)
  fclose(file); 
  
  return error::Success();
}

base::Status Model::generate_model_infos(const ModelConfig& config) const {
  config_->dim_ = config.dim;
  config_->hidden_dim_ = config.hidden_dim;
  config_->layer_num_ = config.layer_num;
  config_->head_num_ = config.head_num;
  config_->kv_head_num_ = config.kv_head_num;
  config_->seq_len_ = config.seq_len;

  config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
  config_->kv_mul_ = config.head_num / config.kv_head_num;
  config_->head_size_ = config.dim / config.head_num;
#if defined(QWEN3_SUPPORT)
  config_->immediate_dim_ = config.immediate_dim_;
#endif
  if (config.vocab_size > 0) {
    config_->is_shared_weight_ = true;
  } else {
    config_->is_shared_weight_ = false;
  }

  // Qwen tokenizer size and embedding size is mismatched
  // refer: https://github.com/QwenLM/Qwen2.5/issues/29
  // if (std::abs(config.vocab_size) != config_->vocab_size_) {
  //   return base::error::ModelParseError(
  //       "Vocabulary size mismatch between the model file and the token list.");
  // }
  config_->vocab_size_ = std::abs(config.vocab_size);
  return base::error::Success();
}

base::Status Model::create_encode_layer() {
  using namespace base;

  // create token encode decode layer
  if (tokenizer_type_ == TokenizerType::kEncodeSpe) {
    encode_layer_ = std::make_unique<op::SpeEncodeLayer>(this->token_path_, true, false);
  } else {
#ifdef LLAMA3_SUPPORT
    encode_layer_ = std::make_unique<op::BpeEncodeLayer>(this->token_path_, true, false);
#endif

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    encode_layer_ = std::make_unique<op::QwenEncodeLayer>(this->token_path_, false, false);
#endif
  }
  if (!encode_layer_) {
    return error::InternalError("Create the encode layer failed.");
  }

  config_->vocab_size_ = encode_layer_->vocab_size();
  if (config_->vocab_size_ <= 0) {
    return error::InternalError("The vocab size param read error from the model file!");
  }
  return error::Success();
}

base::Status Model::gen_model_from_file() {
  using namespace base;
  config_ = std::make_unique<TransformerConfig>();

  // init sentence piece processor
  // google sentence piece
  auto create_encode_status = create_encode_layer();
  if (!create_encode_status) {
    LOG(ERROR) << "Create the encode layer failed!";
    return create_encode_status;
  }
  // mmap
  auto mmap_status = read_model_file();
  if (!mmap_status) {
    LOG(ERROR) << "Handle model file " << model_path_ << " failed!";
    return mmap_status;
  }
  auto layer_create_status = create_layers();
  if (!layer_create_status) {
    LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!";
    return layer_create_status;
  }

  return error::Success();
}

std::vector<int32_t> Model::encode(const std::string& sentence) const {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->encode(sentence);
}

bool Model::is_sentence_ending(int32_t token_idx) const {
    // Qwen2.5 的特殊 token
    // <|im_end|> = 151645
    // <|endoftext|> = 151643
    if (token_idx == 151645 || token_idx == 151643) {
        return true;
    }
    return false;
}

std::string Model::decode(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idx);
}

std::string Model::decode(std::vector<int32_t> token_idxs) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idxs);
}

// 读 kv cahce，每一层存储 seq_len * kv_dim 个 fp32 数值，连续一维度数组
std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(
    int32_t layer_idx, int32_t token_pos, int32_t len) const {
    
  // 计算起始偏移量 (layer offset + pos offset)
  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  // 指向某个 layer 某个 token_pos 的开始位置，对应一整条 key 或 value 向量
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;
  // 获取指向该位置的指针
  float* key_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
  float* val_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

  // 创建包含 len 行的 tensor view
  tensor::Tensor key(base::DataType::kDataTypeFp32, len, config_->kv_dim_, false, nullptr,
                     key_cache_ptr);
  tensor::Tensor val(base::DataType::kDataTypeFp32, len, config_->kv_dim_, false, nullptr,
                     val_cache_ptr);
  key.set_device_type(device_type_);
  val.set_device_type(device_type_);
  
  // [seq_len, kv_dim]
  return {key, val};
}

// 为 decode 阶段设计，用于逐个 token 处理的场景
tensor::Tensor Model::fill_input(const tensor::Tensor& pos_tensor,
                                 const op::EmbeddingOutput& embedding_output,
                                 bool is_prompt) const {
  // 获取当前位置索引的第一个元素
  const int32_t pos = pos_tensor.index<int32_t>(0);
  auto [input_tokens, input_embeddings, input_token_num] = embedding_output;
  // 只有一个 token，所以 index 永远是 0
  int32_t index = 0;
  if (is_prompt) {
    // prompt 阶段取 pos 作为偏移量
    index = pos;
  }
#if defined(QWEN3_SUPPORT)
  std::shared_ptr<base::Buffer> input_emb_buffer = std::make_shared<base::Buffer>(
      config_->hidden_dim_ * sizeof(float), nullptr,
      input_embeddings.ptr<float>(index * config_->hidden_dim_), true);
  tensor::Tensor input(base::DataType::kDataTypeFp32, config_->hidden_dim_);

#else
  // 创建一个指向特定 embedding 数据的 buffer（零拷贝），config_->dim_ * sizeof(float) 表示第一个 token 的长度
  // input_embedding.ptr<float>(index * config_->dim_) 计算内存偏移地址
  std::shared_ptr<base::Buffer> input_emb_buffer =
      std::make_shared<base::Buffer>(config_->dim_ * sizeof(float), nullptr,
                                     input_embeddings.ptr<float>(index * config_->dim_), true);
  // 创建 tensor 对象并赋值，一个维度为 [dim] 即单个 token 的 tensor，数据指向 input_embedding 中第 index 个位置
  tensor::Tensor input(base::DataType::kDataTypeFp32, config_->dim_);
#endif
  input.assign(input_emb_buffer);
  input.set_device_type(device_type_);
  return input;
}

}  // namespace model
#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "model/qwen2.h"
#include <chrono>
#include <cuda_runtime.h>


int32_t generate(const model::Qwen2Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false) {
    // 1. tokenizer
    auto tokens = model.encode(sentence);
    // encode 经测试正常
    int32_t prompt_len = tokens.size();
    LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

    bool is_prompt = true;
    int32_t pos = 0;
    int32_t next = tokens.at(pos);
    std::vector<int32_t> history_tokens = tokens;

    // 预分配 pos tensor
    tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
    // 计时变量
    double prefill_time_s = 0.0;
    double decode_time_s = 0.0;
    int decode_tokens_count = 0;

    // 2. prefill
    pos_tensor.reshape({prompt_len});
    for (int i = 0; i < prompt_len; i++) {
        pos_tensor.index<int32_t>(i) = i;
    }
    auto t_prefill_start = std::chrono::steady_clock::now();
    const auto& prompt_embedding = model.embedding(tokens);
    // embedding 经测试无误
    model.predict(prompt_embedding.input_embeddings, pos_tensor, history_tokens, is_prompt, next);
    cudaDeviceSynchronize(); // 确保 GPU 计算完成
    auto t_prefill_end = std::chrono::steady_clock::now();
    prefill_time_s = std::chrono::duration<double>(t_prefill_end - t_prefill_start).count();

    // 3. decode
    pos = prompt_len;
    is_prompt = false;
    std::vector<int32_t> current_token(1);
    
    // 恢复 pos_tensor 为单步形状 [1]
    pos_tensor.reshape({1});
    auto t_decode_start = std::chrono::steady_clock::now();
    while (pos < total_steps) {
        // 设置当前 step 的位置
        pos_tensor.index<int32_t>(0) = pos;
        current_token[0] = next;
        const auto& token_embedding = model.embedding(current_token);
        model.predict(token_embedding.input_embeddings, pos_tensor, history_tokens, is_prompt, next);
        decode_tokens_count++;
        if (model.is_sentence_ending(next)) 
            break;
        history_tokens.push_back(next);
        pos += 1;
        
        if (need_output) {
            std::string new_word = model.decode(current_token);
            printf("%s", new_word.c_str());
            fflush(stdout);
        }
    }
    cudaDeviceSynchronize(); // 确保 GPU 计算完成
    auto t_decode_end = std::chrono::steady_clock::now();
    decode_time_s = std::chrono::duration<double>(t_decode_end - t_decode_start).count();
    // 4. 统计输出
    if (need_output) {
        printf("\n\n[Performance Statistics]\n");
        printf("------------------------------------------------\n");
        printf("Prompt Len:   %4d tokens\n", prompt_len);
        printf("Gen Len:      %4d tokens\n", decode_tokens_count);
        printf("------------------------------------------------\n");
        // ttft
        printf("TTFT (Prefill): %6.4f s  | Speed: %8.2f tokens/s\n", 
               prefill_time_s, prompt_len / prefill_time_s);
        // tpot
        double tpot = decode_time_s * 1000.0 / decode_tokens_count; // ms/token
        printf("TPOT (Decode):  %6.4f ms | Speed: %8.2f tokens/s\n", 
               tpot, decode_tokens_count / decode_time_s);
        
        printf("Total Latency:  %6.4f s\n", prefill_time_s + decode_time_s);
        printf("------------------------------------------------\n");
    }

    return std::min(pos, total_steps);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        LOG(INFO) << "Usage: ./demo checkpoint_path tokenizer_path";
        return -1;
    }
    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];

    model::Qwen2Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path, false);
    model.set_cuda_graph(true);
    auto init_status = model.init(base::DeviceType::kDeviceCUDA);
    if (!init_status) {
        LOG(FATAL) << "The model init failed, code: " << init_status.get_err_code();
    }
    std::string sentence = "Beijing is the capital of";
    // GPU 热身 (warm-up)
    // 防止第一次 CUDA context 初始化影响 prefill 计时
    {
      std::vector<int32_t> dummy_history;
      int dummy_next;
      tensor::Tensor dummy_pos = model.get_buffer(model::ModelBufferType::kInputPos);
      dummy_pos.reshape({1});
      dummy_pos.index<int32_t>(0) = 0;
      std::vector<int32_t> dummy_token = {0}; // 随便一个 token ID
      const auto& emb = model.embedding(dummy_token);
      // 跑一次 decode
      model.predict(emb.input_embeddings, dummy_pos, dummy_history, false, dummy_next);
      cudaDeviceSynchronize();
    }
    // 正式 generate
    int steps = generate(model, sentence, 128, true);

    return 0;
}
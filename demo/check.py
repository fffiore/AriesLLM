import torch
from transformers import AutoModelForCausalLM

model_path = "/root/workspace/AriesLLM/Qwen/Qwen2.5-0.5B/"

model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    torch_dtype=torch.float32,
    device_map="cpu"
)

input_ids = torch.tensor([[3430, 23649, 374, 279, 6722, 315]])

results = {}

# 使用 pre-hook 获取进入 O_proj 之前的数据
def get_o_proj_input(name):
    def hook(module, args):
        # args[0] 是输入 tensor
        results[name] = args[0].detach().clone()
    return hook

# 注册到 Layer 0 的 o_proj
model.model.layers[0].self_attn.o_proj.register_forward_pre_hook(get_o_proj_input('layer0_mha_out'))

print("Running inference...")
with torch.no_grad():
    model(input_ids)

mha_out = results['layer0_mha_out'][0] # [Seq, Hidden]

print(f"\n=== [Python Ground Truth] Layer 0 MHA Output (Pre-O_proj) ===")
print(f"Shape: {mha_out.shape}")

# 打印 Token 0 (Head)
print(f"Token 0 (First 10): {mha_out[0, :10].tolist()}")
# 打印 Token -1 (Head)
print(f"Token -1 (First 10): {mha_out[-1, :10].tolist()}")
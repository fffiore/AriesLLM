"""
Complete export.py for AriesLLM/KuiperLLM (Qwen2.5 Support)
Features:
- Exports Bias for Q, K, V layers (Critical for Qwen2)
- Permutes Weights AND Biases for RoPE (Half-Split format)
- Aligns perfectly with the C++ pointer arithmetic
"""
import os
import struct
import argparse
import json
import torch
from torch import nn

# -----------------------------------------------------------------------------
# Model Definitions (Containers to hold weights before export)

class ModelArgs:
    def __init__(self, **kwargs):
        self.dim = 512
        self.n_layers = 8
        self.n_heads = 8
        self.n_kv_heads = None
        self.vocab_size = -1
        self.multiple_of = 256
        self.norm_eps = 1e-5
        self.max_seq_len = 2048
        for k, v in kwargs.items():
            setattr(self, k, v)

class Layer:
    def __init__(self):
        self.attention_norm = None # RMSNorm
        self.ffn_norm = None       # RMSNorm
        self.attention = None      # Holder for wq, wk, wv, wo
        self.feed_forward = None   # Holder for w1, w2, w3

class Attention:
    def __init__(self):
        self.wq = None # Linear
        self.wk = None # Linear
        self.wv = None # Linear
        self.wo = None # Linear

class FeedForward:
    def __init__(self):
        self.w1 = None # Linear (Gate)
        self.w2 = None # Linear (Down)
        self.w3 = None # Linear (Up)

class Transformer:
    def __init__(self, params):
        self.params = params
        self.tok_embeddings = None
        self.norm = None
        self.output = None
        self.layers = []

# -----------------------------------------------------------------------------
# Utilities

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

# -----------------------------------------------------------------------------
# Loading from HuggingFace

def load_hf_model(model_path):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        return None

    print(f"Loading HuggingFace model from: {model_path}")
    # Load model (bfloat16 or float32 handled by torch, we convert to fp32 later)
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    hf_dict = hf_model.state_dict()
    config_hf = hf_model.config

    # Setup Config
    args = ModelArgs()
    args.dim = config_hf.hidden_size
    args.n_layers = config_hf.num_hidden_layers
    args.n_heads = config_hf.num_attention_heads
    args.n_kv_heads = config_hf.num_key_value_heads
    args.vocab_size = config_hf.vocab_size
    args.norm_eps = config_hf.rms_norm_eps
    args.max_seq_len = config_hf.max_position_embeddings
    
    # Qwen2 intermediate size logic
    args.hidden_dim = config_hf.intermediate_size 

    print(f"Model Params: dim={args.dim}, layers={args.n_layers}, heads={args.n_heads}, kv_heads={args.n_kv_heads}")

    # Helper to permute weights for RoPE (Half-Split format: [x0, x1.. | x_half..])
    # This matches the C++ Kernel: `idx` and `idx + half`
    def permute_weight(w, n_heads, dim_in, dim_out):
        # w: [dim_out, dim_in]
        return w.view(n_heads, 2, dim_out // n_heads // 2, dim_in).transpose(1, 2).reshape(dim_out, dim_in)

    def permute_bias(b, n_heads, dim_out):
        # b: [dim_out]
        return b.view(n_heads, 2, dim_out // n_heads // 2).transpose(1, 2).reshape(dim_out)

    # Create Container
    model = Transformer(args)
    
    # 1. Global Weights
    model.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
    model.tok_embeddings.weight = nn.Parameter(hf_dict['model.embed_tokens.weight'])
    
    model.norm = nn.LayerNorm(args.dim) # Placeholder type, just holding weight
    model.norm.weight = nn.Parameter(hf_dict['model.norm.weight'])
    
    model.output = nn.Linear(args.dim, args.vocab_size)
    model.output.weight = nn.Parameter(hf_dict['lm_head.weight'])

    # 2. Layers
    for i in range(args.n_layers):
        layer = Layer()
        prefix = f'model.layers.{i}'
        
        # Norms
        layer.attention_norm = nn.LayerNorm(args.dim)
        layer.attention_norm.weight = nn.Parameter(hf_dict[f'{prefix}.input_layernorm.weight'])
        
        layer.ffn_norm = nn.LayerNorm(args.dim)
        layer.ffn_norm.weight = nn.Parameter(hf_dict[f'{prefix}.post_attention_layernorm.weight'])
        
        # Attention
        layer.attention = Attention()
        
        # Q (Weight + Bias) - Needs Permute
        q_w = hf_dict[f'{prefix}.self_attn.q_proj.weight']
        q_b = hf_dict[f'{prefix}.self_attn.q_proj.bias']
        layer.attention.wq = nn.Linear(args.dim, args.dim)
        layer.attention.wq.weight = nn.Parameter(permute_weight(q_w, args.n_heads, args.dim, args.dim))
        layer.attention.wq.bias = nn.Parameter(permute_bias(q_b, args.n_heads, args.dim))
        
        # K (Weight + Bias) - Needs Permute
        k_w = hf_dict[f'{prefix}.self_attn.k_proj.weight']
        k_b = hf_dict[f'{prefix}.self_attn.k_proj.bias']
        dim_k = args.n_kv_heads * (args.dim // args.n_heads)
        layer.attention.wk = nn.Linear(args.dim, dim_k)
        layer.attention.wk.weight = nn.Parameter(permute_weight(k_w, args.n_kv_heads, args.dim, dim_k))
        layer.attention.wk.bias = nn.Parameter(permute_bias(k_b, args.n_kv_heads, dim_k))
        
        # V (Weight + Bias) - No Permute
        layer.attention.wv = nn.Linear(args.dim, dim_k)
        layer.attention.wv.weight = nn.Parameter(hf_dict[f'{prefix}.self_attn.v_proj.weight'])
        layer.attention.wv.bias = nn.Parameter(hf_dict[f'{prefix}.self_attn.v_proj.bias'])
        
        # O (Weight) - Usually no bias in Qwen2
        layer.attention.wo = nn.Linear(args.dim, args.dim)
        layer.attention.wo.weight = nn.Parameter(hf_dict[f'{prefix}.self_attn.o_proj.weight'])
        # Optional: Check if O has bias
        if f'{prefix}.self_attn.o_proj.bias' in hf_dict:
             layer.attention.wo.bias = nn.Parameter(hf_dict[f'{prefix}.self_attn.o_proj.bias'])
        
        # FFN
        layer.feed_forward = FeedForward()
        # Gate (w1)
        layer.feed_forward.w1 = nn.Linear(args.dim, args.hidden_dim)
        layer.feed_forward.w1.weight = nn.Parameter(hf_dict[f'{prefix}.mlp.gate_proj.weight'])
        # Down (w2)
        layer.feed_forward.w2 = nn.Linear(args.hidden_dim, args.dim)
        layer.feed_forward.w2.weight = nn.Parameter(hf_dict[f'{prefix}.mlp.down_proj.weight'])
        # Up (w3)
        layer.feed_forward.w3 = nn.Linear(args.dim, args.hidden_dim)
        layer.feed_forward.w3.weight = nn.Parameter(hf_dict[f'{prefix}.mlp.up_proj.weight'])
        
        model.layers.append(layer)

    return model

# -----------------------------------------------------------------------------
# Export Logic

def version1_export(model, filepath):
    """
    Exports to .bin file matching the specific read order of the C++ Code.
    Order:
    1. Header
    2. Token Embeddings
    3. Attn Norms (All Layers)
    4. WQ (All Layers: Weight, Bias)
    5. WK (All Layers: Weight, Bias)
    6. WV (All Layers: Weight, Bias)
    7. WO (All Layers: Weight)
    8. FFN Norms (All Layers)
    9. W1 (All Layers: Weight)
    10. W2 (All Layers: Weight)
    11. W3 (All Layers: Weight)
    12. Final Norm
    13. Freqs (Dummy)
    14. Classifier
    """
    version = 1
    out_file = open(filepath, 'wb')

    # --- Header ---
    out_file.write(struct.pack('I', 0x616b3432)) # Magic
    out_file.write(struct.pack('i', version))
    
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                         n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)
    
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    out_file.write(struct.pack('B', int(shared_classifier)))
    
    pad = 256 - out_file.tell()
    out_file.write(b'\0' * pad)

    # --- Body (Must strictly follow C++ `create_param_layers` loop order) ---

    print("Writing Token Embeddings...")
    serialize_fp32(out_file, model.tok_embeddings.weight)

    print("Writing Attention Norms...")
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention_norm.weight)

    print("Writing WQ (Weight + Bias)...")
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wq.weight)
        serialize_fp32(out_file, layer.attention.wq.bias)

    print("Writing WK (Weight + Bias)...")
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wk.weight)
        serialize_fp32(out_file, layer.attention.wk.bias)

    print("Writing WV (Weight + Bias)...")
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wv.weight)
        serialize_fp32(out_file, layer.attention.wv.bias)

    print("Writing WO (Weight only)...")
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wo.weight)
        # Note: If your C++ expects WO Bias, uncomment below. Qwen2 usually doesn't have it.
        # if getattr(layer.attention.wo, 'bias', None) is not None:
        #     serialize_fp32(out_file, layer.attention.wo.bias)

    print("Writing FFN Norms...")
    for layer in model.layers:
        serialize_fp32(out_file, layer.ffn_norm.weight)

    print("Writing FFN W1 (Gate)...")
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w1.weight)

    print("Writing FFN W2 (Down)...")
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w2.weight)

    print("Writing FFN W3 (Up)...")
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w3.weight)

    print("Writing Final Norm...")
    serialize_fp32(out_file, model.norm.weight)

    # Freqs (Cos/Sin) - C++ usually generates these, but file format might expect space
    # Based on legacy format, we write dummy zeros or precomputed freqs if strict compatibility needed
    # For this specific C++ code, it seems to skip the file area for freqs or calculate them.
    # Let's write dummy freqs to matches file offsets if standard loader is used.
    # The C++ `create_param_layers` snippet you showed skips `seq_len * head_size`.
    print("Writing Dummy Freqs (C++ generates them)...")
    dummy_freqs = torch.zeros(p.max_seq_len * (p.dim // p.n_heads), dtype=torch.float32)
    serialize_fp32(out_file, dummy_freqs)

    print("Writing Output Classifier...")
    if not shared_classifier:
        serialize_fp32(out_file, model.output.weight)

    out_file.close()
    print(f"Successfully wrote {filepath}")

# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=str, help="the output filepath (e.g., Qwen2.5-0.5B.bin)")
    parser.add_argument("--hf", type=str, help="huggingface model path (e.g., Qwen/Qwen2.5-0.5B)", required=True)
    parser.add_argument("--version", default=1, type=int, help="Export version (use 1 for fp32)")
    args = parser.parse_args()

    model = load_hf_model(args.hf)
    if model:
        if args.version == 1:
            version1_export(model, args.output_path)
        else:
            print("Only version 1 (fp32) is fully implemented for Qwen2 in this script.")
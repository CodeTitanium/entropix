import os
import tyro
import torch
import ml_dtypes
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from transformers import AutoModelForCausalLM
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

def translate_key(in_key: str):
    out_key = in_key.replace('.weight', '')
    if out_key.startswith('model.'):
        out_key = out_key.replace('model.', '')
        if out_key.endswith('input_layernorm'):
            out_key = out_key.replace('input_layernorm', 'attention_norm')
        elif out_key.endswith('mlp.down_proj'):
            out_key = out_key.replace('mlp.down_proj', 'feed_forward.w2')
        elif out_key.endswith('mlp.gate_proj'):
            out_key = out_key.replace('mlp.gate_proj', 'feed_forward.w1')
        elif out_key.endswith('mlp.up_proj'):
            out_key = out_key.replace('mlp.up_proj', 'feed_forward.w3')
        elif out_key.endswith('post_attention_layernorm'):
            out_key = out_key.replace('post_attention_layernorm', 'ffn_norm')
        elif out_key.endswith('self_attn.k_proj'):
            out_key = out_key.replace('self_attn.k_proj', 'attention.wk')
        elif out_key.endswith('self_attn.o_proj'):
            out_key = out_key.replace('self_attn.o_proj', 'attention.wo')
        elif out_key.endswith('self_attn.q_proj'):
            out_key = out_key.replace('self_attn.q_proj', 'attention.wq')
        elif out_key.endswith('self_attn.v_proj'):
            out_key = out_key.replace('self_attn.v_proj', 'attention.wv')
        elif out_key.endswith('down_proj'):
            out_key = out_key.replace('down_proj', 'w2')
        elif out_key.endswith('gate_proj'):
            out_key = out_key.replace('gate_proj', 'w1')
        elif out_key.endswith('up_proj'):
            out_key = out_key.replace('up_proj', 'w3')
        elif out_key == 'embed_tokens':
            out_key = 'tok_embeddings'
        elif out_key == 'norm':
            out_key = 'norm'
        else:
            print(f"Don't know how to handle {in_key=}")
            return None
    elif out_key == 'lm_head':
        out_key = 'output'
    else:
        print(f"Don't know how to handle {in_key=}")
        return None
    return f'{out_key}.weight'


def reverse_permute(tensor: torch.Tensor, n_heads: int = 32, dim1: int = 4096, dim2: int = 4096) -> torch.Tensor:
    # Convert to numpy array first
    param = tensor.detach().cpu().numpy()
    # Calculate the size for each head dimension
    head_dim = dim1 // n_heads // 2
    # Reshape using numpy reshape
    reshaped = param.reshape(n_heads, 2, head_dim, dim2)
    # Transpose and reshape back
    transposed = np.transpose(reshaped, (0, 2, 1, 3))
    return torch.tensor(transposed.reshape(dim1, dim2))


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_deepseek.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


def main(model_id: str, out_dir: Path):
    with patch('transformers.dynamic_module_utils.get_imports', fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map=None)
        params = dict(model.named_parameters())
        out_dir.mkdir(parents=True, exist_ok=True)

        # First pass: Check if we have all required keys
        required_keys = {
            'tok_embeddings.weight': False,
            'norm.weight': False,
            'output.weight': False
        }
        
        for key, param in params.items():
            print(f" {key}: param.shape={param.shape}")
            out_key = translate_key(key)
            if not out_key:
                print(f"Skipping {key} - no translation available")
                continue
                
            # Track if we found required keys
            base_key = out_key.replace('.weight', '')
            if base_key in ['tok_embeddings', 'norm', 'output']:
                required_keys[out_key] = True

            # Handle special permutation cases for attention weights
            if key == 'model.layers.0.self_attn.q_proj.weight':
                param = reverse_permute(param, n_heads=32, dim1=2048, dim2=2048)   # 1B model
            elif key == 'model.layers.0.self_attn.k_proj.weight':
                param = reverse_permute(param, n_heads=8, dim1=512, dim2=2048)   # 1B model
            elif key == 'model.layers.0.self_attn.v_proj.weight':
                param = reverse_permute(param, n_heads=8, dim1=512, dim2=2048)   # 1B model

            # Convert to numpy and save
            param_np = param.detach().cpu().numpy()
            out_path = out_dir / out_key
            print(f"Writing {key} as {out_key} to {out_path}")
            jnp.save(str(out_path), param_np.astype(ml_dtypes.bfloat16))
        
        # Check if we're missing any required keys
        missing_keys = [k for k, found in required_keys.items() if not found]
        if missing_keys:
            raise ValueError(f"Missing required weight files: {missing_keys}")


if __name__ == "__main__":
    tyro.cli(main)

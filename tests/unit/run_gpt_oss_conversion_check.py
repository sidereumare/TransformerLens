import os
import sys

import torch

# Ensure local repository version (not site-packages) is imported
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.pretrained.weight_conversions import convert_gpt_oss_weights


# Reuse minimal mocks (duplicated to avoid pytest dependency)
class MockLinear(torch.nn.Module):
    def __init__(self, in_f: int, out_f: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_f, in_f))

class MockExpert(torch.nn.Module):
    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.w3 = MockLinear(d_model, d_mlp)
        self.w1 = MockLinear(d_model, d_mlp)
        self.w2 = MockLinear(d_mlp, d_model)

class MockMoE(torch.nn.Module):
    def __init__(self, d_model: int, d_mlp: int, num_experts: int) -> None:
        super().__init__()
        self.router = MockLinear(d_model, num_experts)
        self.experts = torch.nn.ModuleList([MockExpert(d_model, d_mlp) for _ in range(num_experts)])

class MockSelfAttn(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, n_kv_heads: int) -> None:
        super().__init__()
        total_q = n_heads * d_head
        total_kv = n_kv_heads * d_head
        self.q_proj = MockLinear(d_model, total_q)
        self.k_proj = MockLinear(d_model, total_kv)
        self.v_proj = MockLinear(d_model, total_kv)
        self.o_proj = MockLinear(total_q, d_model)
        self.sinks = torch.nn.Parameter(torch.zeros(n_heads))

import types


class MockLayer(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, n_kv_heads: int, d_mlp: int, num_experts: int) -> None:
        super().__init__()
        self.input_layernorm = types.SimpleNamespace(weight=torch.nn.Parameter(torch.ones(d_model)))
        self.self_attn = MockSelfAttn(d_model, n_heads, d_head, n_kv_heads)
        self.post_attention_layernorm = types.SimpleNamespace(weight=torch.nn.Parameter(torch.ones(d_model)))
        self.mlp = types.SimpleNamespace(router=None, experts=None)
        if num_experts:
            moe = MockMoE(d_model, d_mlp, num_experts)
            self.mlp.router = moe.router
            self.mlp.experts = moe.experts
        else:
            dense = types.SimpleNamespace(
                up_proj=MockLinear(d_model, d_mlp),
                gate_proj=MockLinear(d_model, d_mlp),
                down_proj=MockLinear(d_mlp, d_model),
            )
            self.mlp = dense

class MockModel(torch.nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_head: int, n_kv_heads: int, d_mlp: int, num_experts: int) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(100, d_model)
        self.layers = torch.nn.ModuleList([
            MockLayer(d_model, n_heads, d_head, n_kv_heads, d_mlp, num_experts) for _ in range(n_layers)
        ])
        self.norm = types.SimpleNamespace(weight=torch.nn.Parameter(torch.ones(d_model)))

class MockGPTOSS(torch.nn.Module):
    def __init__(self, cfg: HookedTransformerConfig) -> None:
        super().__init__()
        n_kv = cfg.n_key_value_heads or cfg.n_heads
        num_experts: int = cfg.num_experts or 0
        self.model = MockModel(cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.d_head, n_kv, cfg.d_mlp, num_experts)
        self.lm_head = MockLinear(cfg.d_model, cfg.d_vocab)


def main():
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=64,
        n_ctx=128,
        d_head=16,
        n_heads=4,
        d_mlp=128,
        d_vocab=100,
        act_fn="relu",
        positional_embedding_type="rotary",
        n_key_value_heads=2,
        num_experts=3,
        experts_per_token=2,
        use_attention_sinks=True,
    )
    model = MockGPTOSS(cfg)
    sd = convert_gpt_oss_weights(model, cfg)
    assert "embed.W_E" in sd, "Missing embedding weight"
    for l in range(cfg.n_layers):
        assert f"blocks.{l}.attn.W_Q" in sd
        assert f"blocks.{l}.attn._W_K" in sd
        assert f"blocks.{l}.attn._W_V" in sd
        assert f"blocks.{l}.attn.attn_sinks" in sd
        assert cfg.num_experts is not None
        for e in range(cfg.num_experts):
            assert f"blocks.{l}.mlp.experts.{e}.W_in.weight" in sd
            assert f"blocks.{l}.mlp.experts.{e}.W_gate.weight" in sd
            assert f"blocks.{l}.mlp.experts.{e}.W_out.weight" in sd
    assert "ln_final.w" in sd
    assert "unembed.W_U" in sd
    wq = sd["blocks.0.attn.W_Q"]
    assert wq.shape == (cfg.n_heads, cfg.d_model, cfg.d_head)
    sinks = sd["blocks.0.attn.attn_sinks"]
    assert sinks.shape == (cfg.n_heads,)
    print("GPT-OSS conversion check: OK")

if __name__ == "__main__":
    main()

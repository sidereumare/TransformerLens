import types

import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.pretrained.weight_conversions import convert_gpt_oss_weights


# Simulate provided GptOssExperts structure
class GptOssExperts(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.expert_dim = intermediate_size
        self.gate_up_proj = torch.nn.Parameter(torch.randn(num_experts, hidden_size, 2 * intermediate_size))
        self.gate_up_proj_bias = torch.nn.Parameter(torch.randn(num_experts, 2 * intermediate_size))
        self.down_proj = torch.nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size))
        self.down_proj_bias = torch.nn.Parameter(torch.randn(num_experts, hidden_size))

class MockSelfAttn(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_head, n_kv):
        super().__init__()
        self.q_proj = torch.nn.Parameter(torch.randn(n_heads * d_head, d_model))
        self.k_proj = torch.nn.Parameter(torch.randn(n_kv * d_head, d_model))
        self.v_proj = torch.nn.Parameter(torch.randn(n_kv * d_head, d_model))
        self.o_proj = torch.nn.Parameter(torch.randn(d_model, n_heads * d_head))
        self.sinks = torch.nn.Parameter(torch.zeros(n_heads))

class MockLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_head, n_kv, d_mlp, num_experts):
        super().__init__()
        self.input_layernorm = types.SimpleNamespace(weight=torch.nn.Parameter(torch.ones(d_model)))
        self.self_attn = types.SimpleNamespace(
            q_proj=MockSelfAttn(d_model, n_heads, d_head, n_kv).q_proj,
            k_proj=MockSelfAttn(d_model, n_heads, d_head, n_kv).k_proj,
            v_proj=MockSelfAttn(d_model, n_heads, d_head, n_kv).v_proj,
            o_proj=MockSelfAttn(d_model, n_heads, d_head, n_kv).o_proj,
            sinks=torch.nn.Parameter(torch.zeros(n_heads)),
        )
        self.post_attention_layernorm = types.SimpleNamespace(weight=torch.nn.Parameter(torch.ones(d_model)))
        # Router (gate)
        self.mlp = types.SimpleNamespace()
        self.mlp.router = types.SimpleNamespace(weight=torch.nn.Parameter(torch.randn(num_experts, d_model)))
        self.mlp.experts = GptOssExperts(d_model, d_mlp, num_experts)

class MockModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.layers = torch.nn.ModuleList([
            MockLayer(cfg.d_model, cfg.n_heads, cfg.d_head, cfg.n_key_value_heads or cfg.n_heads, cfg.d_mlp, cfg.num_experts) for _ in range(cfg.n_layers)
        ])
        self.norm = types.SimpleNamespace(weight=torch.nn.Parameter(torch.ones(cfg.d_model)))

class Wrapper(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = MockModel(cfg)
        self.lm_head = torch.nn.Parameter(torch.randn(cfg.d_vocab, cfg.d_model))

cfg = HookedTransformerConfig(
    n_layers=1,
    d_model=64,
    n_ctx=64,
    d_head=16,
    n_heads=4,
    d_mlp=32,
    d_vocab=128,
    act_fn="silu",
    positional_embedding_type="rotary",
    n_key_value_heads=2,
    num_experts=4,
    experts_per_token=2,
    use_attention_sinks=True,
)

model = Wrapper(cfg)
state = convert_gpt_oss_weights(model, cfg)
print(sorted([k for k in state.keys() if k.startswith("blocks.0.mlp.experts.0")]))
print('OK total keys', len(state))
state = convert_gpt_oss_weights(model, cfg)
print(sorted([k for k in state.keys() if k.startswith("blocks.0.mlp.experts.0")]))
print('OK total keys', len(state))

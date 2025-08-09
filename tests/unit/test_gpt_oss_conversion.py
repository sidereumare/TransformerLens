import types
from pathlib import Path

import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.pretrained.weight_conversions import convert_gpt_oss_weights


# Minimal mock classes mirroring attribute names accessed in converter
class MockLinear(torch.nn.Module):
    def __init__(self, in_f: int, out_f: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_f, in_f))

class MockExpert(torch.nn.Module):
    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        # GPT-OSS naming: w3->up (W_in), w1->gate, w2->down (W_out)
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
            # dense fallback
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
        self.model = MockModel(
            cfg.n_layers,
            cfg.d_model,
            cfg.n_heads,
            cfg.d_head,
            n_kv,
            cfg.d_mlp,
            num_experts,
        )
        self.lm_head = MockLinear(cfg.d_model, cfg.d_vocab)


def test_convert_gpt_oss_weights_moe_with_sinks():
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
    # Basic key checks
    assert "embed.W_E" in sd
    # Attention weights
    for l in range(cfg.n_layers):
        assert f"blocks.{l}.attn.W_Q" in sd
        assert f"blocks.{l}.attn._W_K" in sd
        assert f"blocks.{l}.attn._W_V" in sd
        assert f"blocks.{l}.attn.attn_sinks" in sd
        # MoE experts
        assert cfg.num_experts is not None
        for e in range(cfg.num_experts):
            assert f"blocks.{l}.mlp.experts.{e}.W_in.weight" in sd
            assert f"blocks.{l}.mlp.experts.{e}.W_gate.weight" in sd
            assert f"blocks.{l}.mlp.experts.{e}.W_out.weight" in sd
    # Final
    assert "ln_final.w" in sd
    assert "unembed.W_U" in sd
    # Shape sanity
    wq = sd["blocks.0.attn.W_Q"]
    assert wq.shape == (cfg.n_heads, cfg.d_model, cfg.d_head)
    sinks = sd["blocks.0.attn.attn_sinks"]
    assert sinks.shape == (cfg.n_heads,)


def test_convert_gpt_oss_original_checkpoint_minimal(tmp_path: 'Path'):
    """Create a minimal synthetic 'original' GPT-OSS style checkpoint (single layer) and
    ensure auto-detection + direct conversion path works via HookedTransformer.from_pretrained.
    We only fabricate the tensors actually read by convert_gpt_oss_original_checkpoint for layer 0.
    """
    import json

    from safetensors.torch import save_file

    from transformer_lens import HookedTransformer

    d_model = 32
    d_head = 8
    n_heads = 2
    n_kv = 1
    d_mlp = 32  # (SwiGLU style: mlp1 has 2*d_mlp rows per expert)
    num_experts = 2
    experts_per_token = 1
    n_layers = 1
    vocab = 101
    # Group count for synthetic FP4 blocks (G * 32 == d_model). If d_model < 32, just use 1.
    g_groups = d_model // 32 if d_model >= 32 else 1

    # Config JSON matching fields used in converter
    cfg_json = dict(
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        hidden_size=d_model,
        head_dim=d_head,
        vocab_size=vocab,
        intermediate_size=d_mlp,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        initial_context_length=128,
        rope_theta=10000,
    )
    (tmp_path / "config.json").write_text(json.dumps(cfg_json))
    (tmp_path / "dtypes.json").write_text(json.dumps({}))

    # Build required tensors
    tensors = {}
    # Embedding
    tensors["embedding.weight"] = torch.randn(vocab, d_model, dtype=torch.bfloat16)
    # Layer 0 norms
    tensors["block.0.attn.norm.scale"] = torch.ones(d_model, dtype=torch.bfloat16)
    tensors["block.0.mlp.norm.scale"] = torch.ones(d_model, dtype=torch.bfloat16)
    # qkv packed
    q_size = n_heads * d_head
    kv_size = n_kv * d_head
    qkv_w = torch.randn(q_size + 2 * kv_size, d_model, dtype=torch.bfloat16)
    qkv_b = torch.randn(q_size + 2 * kv_size, dtype=torch.bfloat16)
    tensors["block.0.attn.qkv.weight"] = qkv_w
    tensors["block.0.attn.qkv.bias"] = qkv_b
    tensors["block.0.attn.out.weight"] = torch.randn(d_model, q_size, dtype=torch.bfloat16)
    tensors["block.0.attn.out.bias"] = torch.zeros(d_model, dtype=torch.bfloat16)
    tensors["block.0.attn.sinks"] = torch.zeros(n_heads, dtype=torch.bfloat16)
    # Router (stored transposed vs TL expectation, converter transposes back)
    tensors["block.0.mlp.gate.weight"] = torch.randn(num_experts, d_model, dtype=torch.bfloat16)
    # Expert grouped weights (FP4 blocks & UE8 scales) - fabricate uint8 arrays with correct shapes
    # Shapes: (E, 2*d_mlp, G, 16) and (E, d_mlp, G, 16) with G such that G*32 == d_model
    G = g_groups
    tensors["block.0.mlp.mlp1_weight.blocks"] = torch.randint(
        0, 256, (num_experts, 2 * d_mlp, G, 16), dtype=torch.uint8
    )
    tensors["block.0.mlp.mlp1_weight.scales"] = torch.randint(
        0, 256, (num_experts, 2 * d_mlp, G), dtype=torch.uint8
    )
    tensors["block.0.mlp.mlp2_weight.blocks"] = torch.randint(
        0, 256, (num_experts, d_mlp, G, 16), dtype=torch.uint8
    )
    tensors["block.0.mlp.mlp2_weight.scales"] = torch.randint(
        0, 256, (num_experts, d_mlp, G), dtype=torch.uint8
    )
    tensors["block.0.mlp.mlp1_bias"] = torch.zeros(num_experts, 2 * d_mlp, dtype=torch.bfloat16)
    tensors["block.0.mlp.mlp2_bias"] = torch.zeros(num_experts, d_mlp, dtype=torch.bfloat16)
    # Final / unembedding
    tensors["norm.scale"] = torch.ones(d_model, dtype=torch.bfloat16)
    tensors["unembedding.weight"] = torch.randn(vocab, d_model, dtype=torch.bfloat16)

    save_file(tensors, str(tmp_path / "model.safetensors"))

    # Load via HookedTransformer and layer restriction flag
    model = HookedTransformer.from_pretrained(
        str(tmp_path), gpt_oss_restrict_layers=[0], fold_ln=False, center_writing_weights=False
    )
    # Ensure expected transformed keys exist
    assert hasattr(model, "W_E")  # embedding loaded
    # Inspect internal parameter names via state dict
    sd = model.state_dict()
    assert any(k.startswith("blocks.0.attn.W_Q") for k in sd.keys())
    assert any(k.startswith("blocks.0.mlp.experts.0.W_in.weight") for k in sd.keys())

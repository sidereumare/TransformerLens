import pytest
import torch

from transformer_lens import HookedTransformerConfig
from transformer_lens.components.attention import Attention


def test_attention_sinks_pattern_normalization():
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=32,
        n_ctx=16,
        d_head=8,
        n_heads=4,
        d_mlp=64,
        act_fn="relu",
        d_vocab=100,
        positional_embedding_type="rotary",
        use_attention_sinks=True,
    )
    attn = Attention(cfg)
    B, P = 2, 5
    x = torch.randn(B, P, cfg.d_model)
    out = attn(x, x, x)
    # Access the stored sink probability
    sink_prob = getattr(attn, "last_sink_prob", None)
    assert sink_prob is not None
    # Pattern sums should be 1 over real keys
    pattern = attn.hook_pattern.ctx["pattern"] if hasattr(attn.hook_pattern, "ctx") else None
    if pattern is not None:
        s = pattern.sum(dim=-1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    # sink prob within (0,1)
    assert (sink_prob >= 0).all() and (sink_prob <= 1).all()
    assert out.shape == (B, P, cfg.d_model)

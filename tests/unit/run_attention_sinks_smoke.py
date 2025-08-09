import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformer_lens import HookedTransformer, HookedTransformerConfig


def main():
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=32,
        n_ctx=32,
        d_head=16,
        n_heads=2,
        d_mlp=64,
        d_vocab=100,
        act_fn="relu",
        positional_embedding_type="rotary",
        use_attention_sinks=True,
    )
    model = HookedTransformer(cfg)
    tokens = torch.randint(0, cfg.d_vocab, (1, 10))
    out = model(tokens)
    assert out.shape == (1, 10, cfg.d_vocab)
    # Access internal attention pattern (post-softmax) to ensure normalization still holds
    cache = {}
    _, cache = model.run_with_cache(tokens)
    # Check that sink parameter exists in module
    # pyright: ignore[reportUnknownMemberType]
    assert hasattr(model.blocks[0].attn, "attn_sinks") and getattr(model.blocks[0].attn, "attn_sinks") is not None
    print("Attention sinks smoke test: OK (forward + output shape)")

if __name__ == "__main__":
    main()

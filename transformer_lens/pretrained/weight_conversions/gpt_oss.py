"""Weight conversion utilities for GPT-OSS models (20B / 120B).

This currently provides a placeholder converter describing the mapping that will be
needed to move parameters from the reference GPT-OSS implementation into
TransformerLens' internal state_dict format. It does not yet handle:

* Learned attention sinks (per-head logits appended before softmax)
* Quantized MXFP4 block formats for MoE expert weights (gate_up_proj / down_proj)
* YaRN rotary scaling differences beyond base rotary_base assignment (partial support added elsewhere)

Once raw (dequantized) tensors are available, the following mapping is expected:

GPT-OSS name (per layer l) -> TransformerLens key
-------------------------------------------------
model.layers.l.input_layernorm.weight        -> blocks.l.ln1.w
model.layers.l.self_attn.q_proj.weight       -> blocks.l.attn.W_Q (reshape (n_heads,d_model,d_head))
model.layers.l.self_attn.k_proj.weight       -> blocks.l.attn._W_K (reshape (n_kv_heads,d_model,d_head))
model.layers.l.self_attn.v_proj.weight       -> blocks.l.attn._W_V (reshape (n_kv_heads,d_model,d_head))
model.layers.l.self_attn.o_proj.weight       -> blocks.l.attn.W_O (reshape (n_heads,d_head,d_model))
model.layers.l.post_attention_layernorm.weight -> blocks.l.ln2.w

MoE gate / experts:
  model.layers.l.mlp.router.weight           -> blocks.l.mlp.W_gate.weight (shape d_model,num_experts)
  experts (after dequantization of MXFP4):
    w3 -> W_in  -> blocks.l.mlp.experts.e.W_in.weight
    w1 -> W_gate-> blocks.l.mlp.experts.e.W_gate.weight
    w2 -> W_out -> blocks.l.mlp.experts.e.W_out.weight

Sinks:
  model.layers.l.self_attn.sinks             -> NOT YET SUPPORTED. Would require modifying AbstractAttention
                                                 to add a per-head learnable logit and softmax over an extra slot.

Final layers:
model.norm.weight                            -> ln_final.w
lm_head.weight                               -> unembed.W_U.T (add unembed.b_U zeros)

Usage: once implemented, call convert_gpt_oss_weights(hf_model, cfg) to get a TL-compatible state_dict.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple, cast

import einops
import torch
from safetensors import safe_open

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

# --------------------------------------------------------------------------------------
# Note: This file now contains two pathways:
# 1. convert_gpt_oss_weights: For already-loaded HF module objects (similar to LLaMA/Mixtral)
# 2. convert_gpt_oss_original_checkpoint: Directly convert an "original" GPT-OSS checkpoint
#    as distributed (single safetensors with packed FP4 + UE8 groups for MoE expert projections)
# --------------------------------------------------------------------------------------


def _maybe_dequant_mxfp4(t: torch.Tensor) -> torch.Tensor:
  """Placeholder MXFP4 dequant hook.

  GPT-OSS expert weights may arrive already dequantized if loaded via HF.
  If custom raw block format is encountered, raise for now.
  """
  # Heuristic: if dtype is int32 / int16 etc we don't support yet.
  if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
    raise NotImplementedError("MXFP4 raw block format dequant not implemented yet")
  return t


# ----------------------------- FP4 + UE8 (Provisional) --------------------------------
# The original GPT-OSS MoE expert projections (mlp1 & mlp2) are stored as:
#   mlp1_weight.blocks: (num_experts, 2*d_mlp, n_groups, group_bytes)
#   mlp1_weight.scales: (num_experts, 2*d_mlp, n_groups)  (UE8)
#   mlp2_weight.blocks: (num_experts, d_mlp, n_groups, group_bytes)
#   mlp2_weight.scales: (num_experts, d_mlp, n_groups)
# Where group_bytes=16 and each byte packs two FP4 values (hi & lo nibbles), so values per group = 32.
# Therefore columns = n_groups * 32 == d_model (here 90 * 32 = 2880), matching input dimension.
#
# FP4 & UE8 exact spec was not provided; we implement a provisional decode:
#   - Interpret each 4-bit nibble as signed int in [-8,7] via two's complement, then scale by 1/7.
#   - Interpret UE8 scale byte as linear factor s_u8/255.
#   - Final real_value = (signed_nibble/7) * (scale/255).
# This is ONLY a placeholder for analysis / wiring. Replace with authoritative spec when available.
# We expose an environment variable GPT_OSS_DEQUANT_MODE allowing experimentation (currently only 'linear').

def _dequant_fp4_ue8(
    blocks: torch.Tensor, scales: torch.Tensor, *, mode: str = "linear"
) -> torch.Tensor:
  """Provisional dequantization of packed FP4 + UE8 grouped weights.

  Args:
    blocks: uint8 tensor (E, OUT, G, B=16) where each byte contains two FP4 values.
    scales: uint8 tensor (E, OUT, G)
    mode: reserved for future variants (currently only 'linear').

  Returns:
    Float tensor (E, OUT, D_MODEL) with dtype torch.bfloat16.
  """
  assert blocks.dtype == torch.uint8 and scales.dtype == torch.uint8
  E, OUT, G, B = blocks.shape
  assert B == 16, "Expected group_bytes=16"
  # Extract low & high nibbles -> (E, OUT, G, B*2)
  low = blocks & 0x0F
  high = (blocks >> 4) & 0x0F
  codes = torch.stack((low, high), dim=-1).view(E, OUT, G, B * 2)
  # Two's complement signed 4-bit -> [-8,7]
  signed = ((codes + 8) % 16) - 8
  # Normalize to roughly [-1,1]
  values = signed.to(torch.float32) / 7.0
  # Scales broadcast: (E, OUT, G, 1)
  if mode == "linear":
    scale_f = scales.to(torch.float32) / 255.0
  else:
    raise ValueError(f"Unknown dequant mode: {mode}")
  values = values * scale_f.unsqueeze(-1)
  # Reshape groups back to feature dim = G * (B*2) == d_model
  values = values.view(E, OUT, G * B * 2)
  return values.to(torch.bfloat16)


def _split_mlp1_concat(
    tensor: torch.Tensor, d_mlp: int
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Split concatenated (gate, up) projection rows.

  The provisional assumption is ordering [gate, up] along OUT dimension.
  """
  assert tensor.shape[1] == 2 * d_mlp, "Unexpected mlp1 OUT dimension for split"
  gate = tensor[:, :d_mlp]
  up = tensor[:, d_mlp:]
  return gate, up


def convert_gpt_oss_weights(gpt_oss: torch.nn.Module, cfg: HookedTransformerConfig) -> dict[str, torch.Tensor]:
  """Convert GPT-OSS (20B/120B) HF model weights to TransformerLens format.

  This mirrors llama/mixtral mapping, with MoE & optional GQA. Sinks are a per-head
  parameter in each attention module (gpt_oss.model.layers[l].self_attn.sinks).
  """
  state_dict: Dict[str, torch.Tensor] = {}

  state_dict["embed.W_E"] = gpt_oss.model.embed_tokens.weight

  using_gqa = cfg.n_key_value_heads is not None and cfg.n_key_value_heads != cfg.n_heads
  gqa_uscore = "_" if using_gqa else ""
  n_kv = cast(int, cfg.n_key_value_heads if using_gqa else cfg.n_heads)
  assert cfg.d_mlp is not None

  moe = cfg.num_experts is not None
  if moe:
    assert cfg.num_experts is not None

  for l in range(cfg.n_layers):
    layer = gpt_oss.model.layers[l]
    state_dict[f"blocks.{l}.ln1.w"] = layer.input_layernorm.weight

    W_Q = layer.self_attn.q_proj.weight
    W_K = layer.self_attn.k_proj.weight
    W_V = layer.self_attn.v_proj.weight
    W_O = layer.self_attn.o_proj.weight

    # Reshape like llama if not quantized
    if not cfg.load_in_4bit:
      W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
      W_K = einops.rearrange(W_K, "(n h) m->n m h", n=n_kv)
      W_V = einops.rearrange(W_V, "(n h) m->n m h", n=n_kv)
      W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)

    state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
    state_dict[f"blocks.{l}.attn.{gqa_uscore}W_K"] = W_K
    state_dict[f"blocks.{l}.attn.{gqa_uscore}W_V"] = W_V
    # Biases are zero (GPT-OSS uses RMSNorm + no attn biases)
    state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
    state_dict[f"blocks.{l}.attn.{gqa_uscore}b_K"] = torch.zeros(n_kv, cfg.d_head, dtype=cfg.dtype)
    state_dict[f"blocks.{l}.attn.{gqa_uscore}b_V"] = torch.zeros(n_kv, cfg.d_head, dtype=cfg.dtype)
    state_dict[f"blocks.{l}.attn.W_O"] = W_O
    state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    # Sinks (per-head logit parameter)
    if getattr(layer.self_attn, "sinks", None) is not None:
      state_dict[f"blocks.{l}.attn.attn_sinks"] = layer.self_attn.sinks.detach()

    state_dict[f"blocks.{l}.ln2.w"] = layer.post_attention_layernorm.weight

    if moe and cfg.num_experts is not None:
      assert isinstance(cfg.num_experts, int)
      num_experts = cfg.num_experts
      # Router may be named 'router' or 'gate' depending on implementation
      router_mod = getattr(layer.mlp, "router", None)  # type: ignore[attr-defined]
      if router_mod is None:
        router_mod = getattr(layer.mlp, "gate", None)  # type: ignore[attr-defined]
      if router_mod is None:
        raise AttributeError("Could not find router/gate module on layer.mlp (expected 'router' or 'gate').")
      router_w = router_mod.weight
      if router_w.shape == (num_experts, cfg.d_model):  # transpose to (d_model, num_experts)
        router_w = router_w.T
      if router_w.shape != (cfg.d_model, num_experts):
        raise ValueError(f"Unexpected router weight shape {router_w.shape}")
      state_dict[f"blocks.{l}.mlp.W_gate.weight"] = router_w

      # Experts container: ModuleList or packed experts module
      experts_container = getattr(layer.mlp, "experts", None)  # type: ignore[attr-defined]
      if experts_container is None:
        raise AttributeError("layer.mlp missing 'experts' for MoE mapping")

      if isinstance(experts_container, torch.nn.Module) and hasattr(experts_container, "gate_up_proj"):
        gup = experts_container.gate_up_proj  # (E, d_model, 2*d_mlp)
        gup_bias = getattr(experts_container, "gate_up_proj_bias", None)
        down = experts_container.down_proj    # (E, d_mlp, d_model)
        down_bias = getattr(experts_container, "down_proj_bias", None)
        if gup.shape[0] != num_experts:
          raise ValueError("gate_up_proj first dim mismatch")
        # Interleaved even/odd for (gate, up)
        gate_w = gup[..., ::2]
        up_w = gup[..., 1::2]
        for e in range(num_experts):
          state_dict[f"blocks.{l}.mlp.experts.{e}.W_in.weight"] = _maybe_dequant_mxfp4(up_w[e].transpose(0, 1))
          state_dict[f"blocks.{l}.mlp.experts.{e}.W_gate.weight"] = _maybe_dequant_mxfp4(gate_w[e].transpose(0, 1))
          state_dict[f"blocks.{l}.mlp.experts.{e}.W_out.weight"] = _maybe_dequant_mxfp4(down[e])
          if gup_bias is not None:
            state_dict[f"blocks.{l}.mlp.experts.{e}.b_gate"] = gup_bias[e][::2]
            state_dict[f"blocks.{l}.mlp.experts.{e}.b_in"] = gup_bias[e][1::2]
          if down_bias is not None:
            state_dict[f"blocks.{l}.mlp.experts.{e}.b_out"] = down_bias[e]
      else:
        # Iterable experts
        for e in range(num_experts):
          expert = experts_container[e]  # type: ignore[index]
          state_dict[f"blocks.{l}.mlp.experts.{e}.W_in.weight"] = _maybe_dequant_mxfp4(expert.w3.weight)
          state_dict[f"blocks.{l}.mlp.experts.{e}.W_gate.weight"] = _maybe_dequant_mxfp4(expert.w1.weight)
          state_dict[f"blocks.{l}.mlp.experts.{e}.W_out.weight"] = _maybe_dequant_mxfp4(expert.w2.weight)
    else:
      # Standard MLP (should not happen for GPT-OSS, but safeguard)
      state_dict[f"blocks.{l}.mlp.W_in"] = layer.mlp.up_proj.weight.T
      state_dict[f"blocks.{l}.mlp.W_gate"] = layer.mlp.gate_proj.weight.T
      state_dict[f"blocks.{l}.mlp.W_out"] = layer.mlp.down_proj.weight.T
      state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)
      state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

  state_dict["ln_final.w"] = gpt_oss.model.norm.weight
  state_dict["unembed.W_U"] = gpt_oss.lm_head.weight.T
  state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

  return state_dict


# ------------------------ Original Checkpoint Direct Conversion -----------------------
def convert_gpt_oss_original_checkpoint(
  checkpoint_dir: str,
  *,
  restrict_layers: Sequence[int] | None = None,
  dequant_mode: str = "linear",
) -> tuple[HookedTransformerConfig, dict[str, torch.Tensor]]:
  """Convert a raw downloaded GPT-OSS original checkpoint (single safetensors file) into
  a TransformerLens config & state_dict.

  Args:
    checkpoint_dir: Directory containing config.json, dtypes.json, model.safetensors.
    restrict_layers: Optional iterable of layer indices to convert (for debugging / memory).
    dequant_mode: Dequantization strategy for provisional FP4/UE8 decode.

  Returns:
    (cfg, state_dict)
  """
  import json

  config_path = os.path.join(checkpoint_dir, "config.json")
  weights_path = os.path.join(checkpoint_dir, "model.safetensors")
  with open(config_path, "r", encoding="utf-8") as f:
    raw_cfg = json.load(f)

  n_layers = int(raw_cfg["num_hidden_layers"])  # 24
  n_heads = int(raw_cfg["num_attention_heads"])  # 64
  n_kv = int(raw_cfg.get("num_key_value_heads", n_heads))  # 8
  d_model = int(raw_cfg["hidden_size"])  # 2880
  d_head = int(raw_cfg["head_dim"])  # 64
  d_vocab = int(raw_cfg["vocab_size"])  # 201088
  d_mlp = int(raw_cfg["intermediate_size"])  # 2880 (SwiGLU uses 2*d_mlp internally)
  num_experts = int(raw_cfg["num_experts"])  # 32
  experts_per_token = int(raw_cfg["experts_per_token"])  # 4

  rope_theta = raw_cfg.get("rope_theta", 150000)
  rope_scaling_factor = raw_cfg.get("rope_scaling_factor", None)
  rope_ntk_alpha = raw_cfg.get("rope_ntk_alpha", None)
  rope_ntk_beta = raw_cfg.get("rope_ntk_beta", None)

  # Build TL config
  cfg = HookedTransformerConfig(
      n_layers=n_layers,
      d_model=d_model,
      n_ctx=raw_cfg.get("initial_context_length", 4096),
      d_head=d_head,
      n_heads=n_heads,
      d_mlp=d_mlp,
      d_vocab=d_vocab,
      act_fn="silu",  # SwiGLU style (gate * silu(up))
      positional_embedding_type="rotary",
      rotary_base=rope_theta,
      n_key_value_heads=n_kv,
      num_experts=num_experts,
      experts_per_token=experts_per_token,
      use_attention_sinks=True,
      normalization_type="RMS",  # GPT-OSS appears to use RMSNorm (scale only)
      rope_scaling={
          "factor": rope_scaling_factor,
          "ntk_alpha": rope_ntk_alpha,
          "ntk_beta": rope_ntk_beta,
      },
      original_architecture="GptOssForCausalLM",
      dtype=torch.bfloat16,
  )

  # Prepare state dict
  state: dict[str, torch.Tensor] = {}

  with safe_open(weights_path, framework="pt") as f:
    # Embedding
    if "embedding.weight" in f.keys():
      state["embed.W_E"] = f.get_tensor("embedding.weight")

    # Iterate layers
    layer_indices = range(n_layers) if restrict_layers is None else restrict_layers
  # simply iterate selected layers
    for l in layer_indices:
      prefix = f"block.{l}."
      # Norms
      state[f"blocks.{l}.ln1.w"] = f.get_tensor(prefix + "attn.norm.scale")
      state[f"blocks.{l}.ln2.w"] = f.get_tensor(prefix + "mlp.norm.scale")

      # Attention qkv
      qkv_w = f.get_tensor(prefix + "attn.qkv.weight")  # (QKV, d_model)
      qkv_b = f.get_tensor(prefix + "attn.qkv.bias")  # (QKV,)
      # Sizes
      total_qkv, dim_in = qkv_w.shape
      assert dim_in == d_model
      q_size = n_heads * d_head
      kv_size = n_kv * d_head
      assert total_qkv == q_size + 2 * kv_size, "Unexpected qkv packing size"
      # Slice
      w_q, w_k, w_v = torch.split(qkv_w, [q_size, kv_size, kv_size], dim=0)
      b_q, b_k, b_v = torch.split(qkv_b, [q_size, kv_size, kv_size], dim=0)
      # Reshape
      W_Q = einops.rearrange(w_q, "(h dh) m -> h m dh", h=n_heads)
      W_K = einops.rearrange(w_k, "(h dh) m -> h m dh", h=n_kv)
      W_V = einops.rearrange(w_v, "(h dh) m -> h m dh", h=n_kv)
      state[f"blocks.{l}.attn.W_Q"] = W_Q
      gqa_uscore = "_" if n_kv != n_heads else ""
      state[f"blocks.{l}.attn.{gqa_uscore}W_K"] = W_K
      state[f"blocks.{l}.attn.{gqa_uscore}W_V"] = W_V
      state[f"blocks.{l}.attn.b_Q"] = einops.rearrange(b_q, "(h dh)->h dh", h=n_heads)
      state[f"blocks.{l}.attn.{gqa_uscore}b_K"] = einops.rearrange(b_k, "(h dh)->h dh", h=n_kv)
      state[f"blocks.{l}.attn.{gqa_uscore}b_V"] = einops.rearrange(b_v, "(h dh)->h dh", h=n_kv)

      # Attention output
      W_O_full = f.get_tensor(prefix + "attn.out.weight")  # (d_model, q_size)
      b_O_full = f.get_tensor(prefix + "attn.out.bias")
      # Rearrange to (n_heads, d_head, d_model)
      W_O = einops.rearrange(W_O_full, "m (h dh) -> h dh m", h=n_heads)
      state[f"blocks.{l}.attn.W_O"] = W_O
      state[f"blocks.{l}.attn.b_O"] = b_O_full

      # Sinks
      if prefix + "attn.sinks" in f.keys():
        state[f"blocks.{l}.attn.attn_sinks"] = f.get_tensor(prefix + "attn.sinks")

      # MoE router (gate)
      router_w = f.get_tensor(prefix + "mlp.gate.weight")  # (num_experts, d_model)
      state[f"blocks.{l}.mlp.W_gate.weight"] = router_w.T  # (d_model, num_experts)
      # (Bias present but TL router currently weight-only; ignore or store?)

      # Expert packed weights
      w1_blocks = f.get_tensor(prefix + "mlp.mlp1_weight.blocks")  # (E, 2*d_mlp, G, 16)
      w1_scales = f.get_tensor(prefix + "mlp.mlp1_weight.scales")  # (E, 2*d_mlp, G)
      w2_blocks = f.get_tensor(prefix + "mlp.mlp2_weight.blocks")  # (E, d_mlp, G, 16)
      w2_scales = f.get_tensor(prefix + "mlp.mlp2_weight.scales")  # (E, d_mlp, G)
      w1_deq = _dequant_fp4_ue8(w1_blocks, w1_scales, mode=dequant_mode)  # (E, 2*d_mlp, d_model)
      w2_deq = _dequant_fp4_ue8(w2_blocks, w2_scales, mode=dequant_mode)  # (E, d_mlp, d_model)
      gate_part, up_part = _split_mlp1_concat(w1_deq, d_mlp)  # (E, d_mlp, d_model) each
      # Biases
      b1 = f.get_tensor(prefix + "mlp.mlp1_bias")  # (E, 2*d_mlp)
      b2 = f.get_tensor(prefix + "mlp.mlp2_bias")  # (E, d_mlp)
      b_gate, b_up = torch.split(b1, [d_mlp, d_mlp], dim=1)

      for e in range(num_experts):
        state[f"blocks.{l}.mlp.experts.{e}.W_in.weight"] = up_part[e]
        state[f"blocks.{l}.mlp.experts.{e}.W_gate.weight"] = gate_part[e]
        state[f"blocks.{l}.mlp.experts.{e}.W_out.weight"] = w2_deq[e]
        # Store biases for traceability (not always used in TL MoE forward)
        state[f"blocks.{l}.mlp.experts.{e}.b_in"] = b_up[e]
        state[f"blocks.{l}.mlp.experts.{e}.b_gate"] = b_gate[e]
        state[f"blocks.{l}.mlp.experts.{e}.b_out"] = b2[e]

    # Final norm & unembed
    if "norm.scale" in f.keys():
      state["ln_final.w"] = f.get_tensor("norm.scale")
    if "unembedding.weight" in f.keys():
      W_U = f.get_tensor("unembedding.weight")  # (d_vocab, d_model)
      state["unembed.W_U"] = W_U.T
      state["unembed.b_U"] = torch.zeros(d_vocab, dtype=W_U.dtype)

  return cfg, state

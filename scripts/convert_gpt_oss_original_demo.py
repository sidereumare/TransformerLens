import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from transformer_lens.pretrained.weight_conversions import (
    convert_gpt_oss_original_checkpoint,
)

CHK_DIR = os.path.join('gpt-oss-20b','original')

cfg, sd = convert_gpt_oss_original_checkpoint(CHK_DIR, restrict_layers=range(2))
print('Converted config (partial layers):')
print('n_layers', cfg.n_layers, 'n_heads', cfg.n_heads, 'n_key_value_heads', cfg.n_key_value_heads)
print('State dict keys sample:', len(sd), 'total entries')
for k in list(sd.keys())[:20]:
    v = sd[k]
    print(k, tuple(v.shape) if isinstance(v, torch.Tensor) else type(v))

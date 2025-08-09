import os
import sys

from safetensors import safe_open

path = os.path.join('gpt-oss-20b','original','model.safetensors')
with safe_open(path, framework='pt') as f:
    keys = list(f.keys())
    print('TOTAL_TENSORS', len(keys))
    for k in keys[:120]:
        t = f.get_tensor(k)
        print(k, tuple(t.shape), t.dtype)

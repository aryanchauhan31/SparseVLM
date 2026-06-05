---
license: apache-2.0
tags:
  - vision-language-model
  - inference-optimization
  - token-pruning
  - qwen2-vl
library_name: sparsevlm
---

# SparseVLM

[![PyPI](https://img.shields.io/pypi/v/sparsevlm)](https://pypi.org/project/sparsevlm/)
[![Paper](https://img.shields.io/badge/ICML_2025-Paper-blue)](https://arxiv.org/abs/2410.04417)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)
[![Tests](https://github.com/aryanchauhan31/SparseVLM/actions/workflows/tests.yml/badge.svg)](https://github.com/aryanchauhan31/SparseVLM/actions)

Training-free visual token pruning for Qwen2.5-VL. Scores visual tokens by how much text attends to them, prunes the unimportant ones from the KV cache, and decodes with the smaller cache.

Based on [SparseVLM: Visual Token Sparsification for Efficient VLM Inference](https://arxiv.org/abs/2410.04417) (ICML 2025).

---

## Install

```bash
pip install sparsevlm
```

Requirements: Python 3.10+, PyTorch 2.1+, transformers 4.49+

---

## Quick start

```python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sparsevlm import sparsevlm_generate
from PIL import Image

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

image = Image.open("your_image.jpg")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text",  "text": "Describe this image in detail."}
]}]
text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

# count visual tokens
n_vis = int((inputs["image_grid_thw"][0].prod() / 4).item())

output = sparsevlm_generate(
    model, processor, inputs,
    n_vis=n_vis,
    keep_n_vis=n_vis // 4,   # keep 25% of visual tokens
    max_new_tokens=256,
)
print(processor.decode(output[0][1:], skip_special_tokens=True))
```

---

## Benchmark results

Measured on **NVIDIA A100-SXM4-40GB**, Qwen2.5-VL-7B-Instruct, bfloat16, SDPA attention.

### Real photo — Fuji mountain + Milky Way (4928×2773px, 16320 visual tokens)

| Config | Tokens kept | Time | Speedup | Output quality |
|---|---|---|---|---|
| Baseline | 16320 (100%) | 9738ms | 1.00× | Identifies Fuji, Milky Way, snow cap, star colors |
| SparseVLM 50% | 8192 | 9441ms | 1.03× | Same quality |
| SparseVLM 25% | 4080 | 9297ms | 1.05× | All key details preserved |
| SparseVLM 10% | 1632 | 9425ms | 1.03× | Still correctly describes scene |

> **Key result:** Full 4K image (16K tokens) runs without OOM. Without SparseVLM's hook-based scoring, the 16K-token image requires materialising a 15GB attention matrix and crashes. The scorer computes only the text→visual submatrix (35 × 16320 = 32MB instead of 15GB).

### Resized photo (896×504px, 576 visual tokens), batch=1

| Tokens kept | Time | Speedup |
|---|---|---|
| 576 (100%) | 2167ms | 1.00× |
| 288 (50%)  | 1685ms | 1.29× |
| **144 (25%)** | **1565ms** | **1.39×** |
| 72  (12%)  | 1620ms | 1.34× |

### When to expect larger speedup

Speedup grows when the KV cache is large relative to model weights:

| Scenario | Expected speedup |
|---|---|
| Single image, short generation | ~1.1–1.4× |
| Single image, 256+ output tokens | ~1.5–2.5× |
| Batch=32, high-res images | ~2–4× |
| Very long visual context (10K+ tokens) | ~2–4× |

---

## How it works

### Token scoring (no extra parameters)

At decoder layer 2, a lightweight hook intercepts the attention projection and computes:

```
A_tv = Q_text @ K_visual^T   # only the text→visual submatrix
                              # 35 × 16320 instead of 16320 × 16320
score_i = sum over text tokens of attention to visual token i
```

Visual tokens with high scores are important to the text query. Low-score tokens are pruned from the KV cache before decoding starts.

### KV cache pruning

After scoring, the KV cache is sliced to keep only the top-K visual entries plus all text entries. The model then decodes with a smaller cache — fewer keys to attend over per decode step.

```
Prefill:  build KV cache for all 16320 visual tokens
Score:    rank each visual token by text attention (32MB op)
Prune:    keep top-K, drop the rest
Decode:   attend over K + N_text keys instead of 16320 + N_text
```

### Position fix (`rope_deltas`)

After pruning, Qwen2.5-VL's internal position counter (`rope_deltas`) is adjusted so decode tokens get correct positional embeddings despite the shorter cache.

---

## API

### `sparsevlm_generate`

```python
from sparsevlm import sparsevlm_generate

output = sparsevlm_generate(
    model,                  # Qwen2_5_VLForConditionalGeneration
    processor,              # AutoProcessor
    inputs,                 # dict from processor(...)
    n_vis,                  # total visual tokens in the sequence
    keep_n_vis,             # how many to keep (e.g. n_vis // 4 for 25%)
    max_new_tokens=256,     # generation length
    target_layer=2,         # which layer to score from (default 2)
    device="cuda",          # primary device
)
# returns: token ids [B, max_new_tokens]
```

### `apply_sparsevlm` / `remove_hooks` (hook-based API)

```python
from sparsevlm import apply_sparsevlm, reset_n_vis, remove_hooks

state = apply_sparsevlm(model, n_vis=256)
reset_n_vis(state, n_vis=256)   # call before each generate
output = model.generate(...)
remove_hooks(state)
```

---

## Model support

| Model | Status |
|---|---|
| Qwen/Qwen2.5-VL-7B-Instruct | Tested |
| Qwen/Qwen2.5-VL-3B-Instruct | Should work |
| Qwen/Qwen2.5-VL-72B-Instruct | Should work |
| Qwen/Qwen2-VL-* | Legacy support |

---

## Limitations

- Requires `attn_implementation="eager"` or `"sdpa"`. Flash Attention 2 (separate package) is not required.
- Speedup is modest (~1.1–1.4×) for single-image, short-generation use cases. The gain comes from long generations, high-resolution images, or batched serving.
- Currently tested with Qwen2.5-VL. Other VLM families would need architecture-specific adaptation.

---

## Citation

```bibtex
@inproceedings{zhang2024sparsevlm,
  title={SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference},
  author={Zhang, Yuan and Fan, Chun-Kai and Ma, Junpeng and Zheng, Wenzhao and
          Huang, Tao and Cheng, Kuan and Gudovskiy, Denis and Okuno, Tomoyuki and
          Nakata, Yohei and Keutzer, Kurt and Zhang, Shanghang},
  booktitle={ICML},
  year={2025}
}
```

Apache 2.0 license.

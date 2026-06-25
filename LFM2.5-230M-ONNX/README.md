---
license: other
license_name: lfm1.0
license_link: LICENSE
language:
- en
- ar
- zh
- fr
- de
- ja
- ko
- es
- pt
- it
pipeline_tag: text-generation
tags:
- liquid
- edge
- lfm2.5
- onnx
- onnxruntime
- webgpu
base_model:
- LiquidAI/LFM2.5-230M
---

<div align="center">
  <img 
    src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/2b08LKpev0DNEk6DlnWkY.png" 
    alt="Liquid AI" 
    style="width: 100%; max-width: 100%; height: auto; display: inline-block; margin-bottom: 0.5em; margin-top: 0.5em;"
  />
  <div style="display: flex; justify-content: center; gap: 0.5em; margin-bottom: 1em;">
    <a href="https://playground.liquid.ai/"><strong>Try LFM</strong></a> • 
    <a href="https://docs.liquid.ai/lfm/getting-started/welcome"><strong>Docs</strong></a> • 
    <a href="https://leap.liquid.ai/"><strong>LEAP</strong></a> • 
    <a href="https://discord.com/invite/liquid-ai"><strong>Discord</strong></a>
  </div>
</div>

# LFM2.5-230M-ONNX

ONNX export of [LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M) for cross-platform inference.

LFM2.5 is a hybrid architecture combining multiplicative gates and short convolutions, optimized for edge deployment with fast inference on CPU, GPU, and NPU hardware.

## Recommended Variants

| Precision | Size  | Platform        | Use Case |
|-----------|-------|-----------------|----------|
| Q4        | ~200 MB | WebGPU, Server | Recommended for most uses (quantized embedding) |
| Q4F32     | ~390 MB | Server (CPU/GPU) | Q4 weights with FP32 embedding — higher quality |
| FP16      | ~455 MB | WebGPU, Server | Higher quality |
| Q8        | ~470 MB | Server only    | Balance of quality and size |

- **WebGPU**: Use `Q4` or `FP16` (`Q4F32` and `Q8` are not supported on WebGPU).
- **Server (CPU/GPU)**: All variants supported. `Q4F32` keeps the embedding in FP32 for higher fidelity.

The tied embedding / LM head is kept in FP32 across all quantized builds.

## Model Files

```
onnx/
├── model.onnx              # FP32
├── model_fp16.onnx         # FP16
├── model_q4.onnx           # Q4, quantized embedding (WebGPU)
├── model_q4f32.onnx        # Q4 weights, FP32 embedding (server)
└── model_q8.onnx           # Q8
```

## Python (onnxruntime)

```bash
pip install onnxruntime transformers numpy huggingface_hub
# or, for GPU:
pip install onnxruntime-gpu transformers numpy huggingface_hub
```

```python
from huggingface_hub import hf_hub_download

model_id = "LiquidAI/LFM2.5-230M-ONNX"
# Q4F32 recommended for server CPU/GPU; use model_q4.onnx for WebGPU.
hf_hub_download(model_id, "onnx/model_q4f32.onnx")
hf_hub_download(model_id, "onnx/model_q4f32.onnx_data")
```

## WebGPU (Transformers.js)

```js
import { pipeline } from "@huggingface/transformers";

const generator = await pipeline("text-generation", "LiquidAI/LFM2.5-230M-ONNX", {
  device: "webgpu",
  dtype: "q4", // or "fp16"
});
```

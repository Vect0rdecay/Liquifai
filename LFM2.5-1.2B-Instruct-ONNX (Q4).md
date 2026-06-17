# GraphSurgeon Analysis Report: LFM2.5-1.2B-Instruct-ONNX (Q4)

**Model:** `model_q4.onnx` (4-bit quantized)  
**Source:** LFM2.5-1.2B-Instruct-ONNX  
**Analysis Date:** 2026-06-17  
**Tool:** GraphSurgeon (inspect, topology, patterns, motifs, flow)

---

## 1. What This Model Is

This is a 1.2 billion parameter language model from Liquid AI's LFM (Liquid Foundation Model) family, exported to ONNX in 4-bit quantized form. What makes it architecturally unusual — and worth studying at the graph level — is that it is **not a standard transformer**. Where GPT-style models repeat the same attention-plus-MLP block at every layer, LFM interleaves two fundamentally different sequence-mixing mechanisms: **1D depthwise convolutions** and **grouped query attention**. The graph structure makes this hybrid design legible in a way that weight inspection alone cannot.

The model has 385 nodes, 348 initializers (stored weights/constants), and a maximum graph depth of 282 hops from input to output. It accepts 24 input tensors and produces 23 outputs — the asymmetry between those numbers already hints at the dual-cache architecture described below.

---

## 2. Model Metadata

| Property | Value |
|----------|-------|
| Graph Name | `main_graph` |
| Total Nodes | 385 |
| Initializers | 348 |
| Max Graph Depth | 282–283 hops |
| Max Fan-In | 7 inputs (at GroupQueryAttention nodes) |
| Longest Linear Chain | 1 |
| Quantization | 4-bit (MatMulNBits) |
| Architecture | Hybrid Conv-Attention LLM (16 layers) |

### Inputs (24 tensors)

```
input_ids, attention_mask,
past_conv.0, past_conv.1,
past_key_values.2.key, past_key_values.2.value,
past_conv.3, past_conv.4,
past_key_values.5.key, past_key_values.5.value,
past_conv.6, past_conv.7,
past_key_values.8.key, past_key_values.8.value,
past_conv.9,
past_key_values.10.key, past_key_values.10.value,
past_conv.11,
past_key_values.12.key, past_key_values.12.value,
past_conv.13,
past_key_values.14.key, past_key_values.14.value,
past_conv.15
```

### Outputs (23 tensors)

```
logits,
present_conv.0, present_conv.1,
present.2.key, present.2.value,
present_conv.3, present_conv.4,
present.5.key, present.5.value,
present_conv.6, present_conv.7,
present.8.key, present.8.value,
present_conv.9,
present.10.key, present.10.value,
present_conv.11,
present.12.key, present.12.value,
present_conv.13,
present.14.key, present.14.value,
present_conv.15
```

### Operator Distribution

| Operator | Count | Role |
|----------|-------|------|
| MatMulNBits | 93 | 4-bit quantized matrix multiplications (projections, MLP layers) |
| Mul | 62 | Element-wise multiplications (gating, scaling) |
| SimplifiedLayerNormalization | 44 | Pre-norm layer normalization (RMSNorm) |
| Add | 32 | Residual connections |
| Reshape | 24 | Tensor shape manipulation for attention heads |
| Transpose | 20 | Dimension reordering (conv layout changes) |
| Slice | 20 | Cache management and output extraction |
| Sigmoid | 16 | SwiGLU activation component |
| Shape | 11 | Dynamic shape computation |
| Gather | 11 | Index-based lookups |
| Split | 10 | Conv cache splitting |
| Conv | 10 | 1D depthwise convolutions (kernel=3, groups=2048) |
| Concat | 10 | Conv input concatenation with cached states |
| Unsqueeze | 10 | Dimension expansion |
| GroupQueryAttention | 6 | Grouped query attention blocks |
| Cast | 2 | Type casting for attention mask |
| ReduceSum | 1 | Attention mask preprocessing |
| Sub | 1 | Attention mask preprocessing |
| SkipSimplifiedLayerNormalization | 1 | Final fused layer norm + skip |
| GatherBlockQuantized | 1 | Quantized token embedding lookup |

### Graph Topology — Depth Distribution

| Region | Depth Range | Node Count | Layers Covered |
|--------|-------------|------------|----------------|
| Early (~30%) | 0–~94 | 83 nodes | Embedding, mask preprocessing, layers 0–4 |
| Middle (~40%) | ~95–~207 | 227 nodes | Layers 5–11 |
| Late (~30%) | ~208–282 | 75 nodes | Layers 12–15, final norm, lm_head |

---

## 3. The Hybrid Layer Pattern

The model has 16 transformer-style layers, but they are not all the same. The ONNX graph reveals two distinct block types that alternate in a specific pattern:

**Convolutional layers** at indices 0, 1, 3, 4, 6, 7, 9, 11, 13, 15 (10 of 16 layers)  
**Attention layers** at indices 2, 5, 8, 10, 12, 14 (6 of 16 layers)

This is not a random arrangement. Looking at the pattern, the model front-loads convolutions — the first two layers are both convolutional — and then settles into a roughly 2-conv-then-1-attention rhythm, with some variation in the later layers where the ratio shifts slightly toward more attention. The design philosophy is clear: use cheap, local convolutions for most of the sequence mixing, and reserve the expensive quadratic-cost attention for periodic global context aggregation.

This is visible in the graph through the input tensor naming. Convolutional layers carry `past_conv.{N}` cache inputs, while attention layers carry `past_key_values.{N}.key` and `past_key_values.{N}.value` pairs. The model maintains **two separate cache systems** simultaneously during autoregressive generation — a convolutional state cache and a KV cache — and the ONNX graph explicitly threads both through the computation.

### Why This Matters at the Graph Level

The conv layers are significantly simpler subgraphs than the attention layers. Each conv block is a linear chain: project → transpose → concat with cache → convolve → gate → project back. Each attention block has a wider topology: three parallel projection paths (Q, K, V), followed by per-head normalization with reshape gymnastics, then the GroupQueryAttention fusion operator that takes 7 inputs simultaneously.

This means the graph has a rhythmic quality — stretches of relatively shallow, sequential computation (conv blocks) interrupted by wider, more complex subgraphs (attention blocks). The topology analysis confirms this: the "early" region (83 nodes) covers layers 0 through roughly 4, while the "middle" region balloons to 227 nodes covering layers 5–11, and the "late" region contracts to 75 nodes. The middle is denser because it contains more attention blocks with their wider fan-out.

---

## 4. Inside a Convolutional Block

Let's trace the data flow through a single conv layer (layer 0) as the graph reveals it, since this is the more unusual of the two block types:

**Step 1 — Pre-normalization.** The hidden state enters `SimplifiedLayerNormalization`. This is a "simplified" variant because it skips the mean subtraction that standard LayerNorm performs — it only divides by the root-mean-square of the values. This is computationally cheaper and is a common choice in modern LLMs (RMSNorm).

**Step 2 — Input projection.** A `MatMulNBits` operation projects the normalized hidden state. This is where the 4-bit quantization lives: rather than a standard MatMul, the weights are stored in packed N-bit format and dequantized on the fly during multiplication. The graph has 93 of these nodes total — they are the dominant operation and represent every learned linear transformation in the model.

**Step 3 — Reshape for convolution.** The projected tensor goes through `Transpose` to rearrange dimensions from (batch, seq, channels) to a layout suitable for 1D convolution. A `Shape → Gather` subgraph dynamically computes sequence length metadata.

**Step 4 — Cache concatenation.** This is the most architecturally interesting part. A `Concat` node merges the current sequence representation with `past_conv.0` (the cached conv state from the previous generation step) along the sequence dimension (axis=2). This is how the convolutional layer maintains temporal context during autoregressive generation — it literally concatenates past and present before convolving.

The concat is preceded by a `Mul` that computes a negative sequence length value, used with `Unsqueeze` and `Split` to set up the slicing boundaries for later cache extraction. This is bookkeeping for the sliding window, not learned computation.

**Step 5 — The actual convolution.** A `Conv` with kernel size 3 and **2048 groups**. The group count equals the channel count, making this a fully depthwise convolution — each of the 2048 channels is convolved independently with its own 3-element kernel. This is extremely parameter-efficient (only 2048 × 3 = 6,144 parameters per conv layer) and provides strictly local context mixing: each position can only see its two immediate neighbors.

**Step 6 — Gated output.** After convolution, the graph splits into two paths via `Slice` operations — one extracts the conv output for the current position, the other extracts the updated cache state (which becomes `present_conv.0` in the output). The conv output then passes through a `Mul` gating operation (element-wise multiplication with a split of the projected input), which functions as a learned gate controlling how much convolutional context to mix in.

**Step 7 — Output projection and residual.** The gated result goes through another `MatMulNBits` (out_proj) and then `Add` with the original hidden state — the classic residual connection.

**Step 8 — The MLP.** After the residual add, a second `SimplifiedLayerNormalization` (ffn_layernorm) feeds into the feed-forward network. This MLP uses the SwiGLU activation pattern, which is visible in the graph as three parallel `MatMulNBits` (gate_proj, up_proj) followed by `Sigmoid → Mul → Mul → MatMulNBits` (down_proj). The SwiGLU computes `sigmoid(gate(x)) * gate(x) * up(x)` — the sigmoid-times-gate portion is the "swish" activation, and the multiplication with the up-projection is the gating mechanism. A final `Add` creates the second residual connection.

The entire conv block therefore has **two residual additions** — one after the conv mixing, one after the MLP — contributing to the 32 total residual connections in the model.

---

## 5. Inside an Attention Block

Attention blocks follow a similar overall structure (pre-norm → mixing → residual → MLP → residual) but the mixing mechanism is completely different. Tracing layer 2 (the first attention layer):

**Q/K/V Projection.** Three parallel `MatMulNBits` operations produce query, key, and value projections. The graph makes the head geometry explicit through the Reshape nodes that follow:
- Query reshapes to `(batch, seq × 32, 64)` — 32 attention heads, each 64-dimensional
- Key reshapes to `(batch, seq × 8, 64)` — only 8 KV heads

This 32:8 ratio (4:1) is **grouped query attention (GQA)**, where every 4 query heads share a single key-value head. This cuts the KV cache size by 4x compared to standard multi-head attention, which is a significant memory savings during long-sequence generation.

**Per-head normalization.** After the initial reshape, both Q and K pass through their own `SimplifiedLayerNormalization` (q_norm, k_norm). This is **QK-norm**, a technique that normalizes the query and key vectors before computing attention scores. It stabilizes training at scale by preventing the dot-product attention scores from growing with model dimension. The graph shows this as Reshape → LayerNorm → Reshape (reshape to isolate heads, normalize, reshape back).

**GroupQueryAttention.** The core attention computation is a single fused `GroupQueryAttention` operator — this is an ONNX Runtime custom op, not a standard ONNX operator. It takes 7 inputs: normalized Q, normalized K, V, past_key, past_value, and sequence length metadata. The fusion means that the attention score computation, softmax, and value-weighted output are all inside this opaque node. The graph cannot tell us the internal attention pattern (causal masking, sliding window, etc.) — that's baked into the operator implementation.

**Output projection and residual.** Same as conv blocks: `MatMulNBits` (o_proj) → `Add` with the skip connection.

**MLP.** Identical SwiGLU pattern as conv blocks.

---

## 6. The Attention Mask Subgraph

Before any layer computation begins, the model preprocesses the `attention_mask` input through a small dedicated subgraph:

`ReduceSum → Sub → Cast → Gather`

This sequence sums the attention mask across the sequence dimension (producing a scalar count of non-masked tokens per batch element), subtracts to compute an offset, casts to the appropriate dtype, and gathers metadata. The result feeds into all 6 GroupQueryAttention nodes as sequence-length information.

This subgraph is noteworthy because it executes at depth 0–5 in the graph — before any learned computation — and its output influences every attention layer. From a structural perspective, this is an **in-graph trust boundary**: the model trusts the attention mask input to be well-formed, transforms it internally, and passes the result directly to attention operators without further validation. Malformed mask values would propagate through this path unchecked.

---

## 7. Token Embedding

The very first learned operation in the graph is `GatherBlockQuantized` at `/model/embed_tokens/Gather_Quant`. This is a quantized embedding lookup — rather than a standard Gather into a float32 embedding table, the embeddings are stored in block-quantized format and dequantized during lookup. This is the only `GatherBlockQuantized` in the graph; all subsequent quantized operations use `MatMulNBits`.

The embedding output has hidden dimension 2048 and feeds directly into the first convolutional layer's normalization. There is no positional encoding visible in the graph — positional information is either embedded within the GroupQueryAttention operator (likely rotary position embeddings applied internally) or handled by the convolutional layers' inherent position-sensitivity.

---

## 8. The Output Head

The final two nodes are:

1. `SkipSimplifiedLayerNormalization` — a fused operation that combines a skip connection with RMSNorm. This is the only instance of this operator in the graph, used exclusively at the output. It takes the final hidden state and the residual from the last layer and produces the normalized output in a single fused step.

2. `MatMulNBits` at `/lm_head/MatMul_Quant` — the language model head that projects from hidden dimension 2048 to vocabulary size, producing the `logits` output.

There is no softmax in the graph — the model outputs raw logits, and temperature scaling / sampling is expected to happen outside the ONNX graph. This is standard for LLM inference exports.

---

## 9. Structural Security Analysis

### Structural Findings Summary

| # | Finding | Category | Key Nodes |
|---|---------|----------|-----------|
| 1 | Dynamic Input Processing (4 shape ops) | adversarial_perturbation | `/model/layers.0/conv/Slice_Conv_Output` |
| 2 | Gradient Highway (32 skip connections) | adversarial_perturbation | `/model/layers.0/Add_1` (first of 32) |
| 3 | In-Graph Preprocessing Trust Boundary | adversarial_perturbation | `/model/attn_mask_reformat/attn_mask_subgraph/Sub` |
| 4 | Multiple Unprotected Feature Fusion (42 fusion, 0 norm) | adversarial_perturbation | All Add/Concat nodes |
| 5 | No Explicit Normalization Layers (10 conv, 0 BatchNorm) | adversarial_perturbation | Conv layers (likely fused during export) |
| 6 | ShadowLogic Injection Susceptibility (71/100) | shadowlogic_injection | 10 identified injection points |

### Structural Pattern Counts

| Pattern Type | Count | Example Nodes |
|-------------|-------|---------------|
| High Fan-In Nodes (7 inputs) | 6 | `layers.{2,5,8,10,12,14}/attn/GroupQueryAttention` |
| Multimodal Fusion Points (2 inputs) | 48 | All Add, Concat, and gating Mul nodes |
| Perturbation Fusion (Concat, axis=2) | 10 | `layers.{0,1,3,4,6,7,9,11,13,15}/conv/Conv_Input` |
| Residual Chain (no gradient regularization) | 32 blocks | `layers.*/Add_1`, `layers.*/Add_2` |
| Saturating Activations | 16 | `layers.*/mlp/act_fn/Sigmoid` |
| Shape Operation Chain | 95 ops | Reshape, Transpose, Slice, Gather, Split, etc. |
| Valid Conv (pads=0, boundary-sensitive) | 10 | `layers.{0,1,3,4,6,7,9,11,13,15}/conv/Conv` |
| ShadowLogic: Deep Architecture Risk (depth=283) | 1 | Full graph |
| In-Graph Preprocessing | 5 layers | Mask subgraph: Sub, Cast, Gather, Mul nodes |

### Gradient Flow Characteristics

The 32 residual connections create unobstructed gradient highways from the output back to the embedding layer. There are no Dropout, DropPath, or noise injection nodes anywhere in the graph. This is expected for an inference-mode export (dropout is identity at inference), but it means that if this graph were used for fine-tuning or if gradients are computed against it for any reason (adversarial example generation, interpretability analysis), gradients flow with maximum fidelity.

The 16 Sigmoid activations in SwiGLU blocks are the only non-linear bottlenecks in the gradient path. Sigmoid saturates for large-magnitude inputs, which causes vanishing gradients in those regions. The patterns analysis flags this as "gradient masking" — but it's a side effect of the activation choice, not an intentional defense. An adversary aware of the sigmoid saturation behavior would simply use attacks that avoid the saturated regime or use boundary/black-box methods that don't rely on gradients.

### Fusion Point Topology

GraphSurgeon identifies 42 fusion operations (Add and Concat nodes) where signals from multiple computation paths converge. The most complex fusion points are the 6 GroupQueryAttention nodes, each receiving 7 inputs. These are the widest points in the graph's topology and represent the locations where the most diverse information streams merge.

The conv cache concat points are a different kind of fusion — they merge temporal information (past and present states) rather than parallel computation paths. An important observation from the graph is that the conv cache has **no gating or normalization at the concat boundary**: past and present states are simply concatenated and convolved. The convolution kernel (size 3) then determines how past and present information interact, but there is no explicit mechanism to detect or reject anomalous cached states.

### ShadowLogic Assessment

GraphSurgeon's ShadowLogic analysis evaluates whether this model graph could be modified to contain a hidden backdoor — not whether one currently exists. The result: **no backdoor indicators detected** (no conditional operations like Where/If/Equal exist in the graph), but the structure scores 71/100 on injection susceptibility.

The high score comes primarily from two factors:

1. **Camouflage potential is HIGH.** The model already uses 93 MatMulNBits, 20 Slice, 11 Gather, and 10 Concat operations. An attacker adding a few more Slice/Gather/Concat nodes to implement trigger detection logic would be statistically invisible in a node-type histogram. The only operation types that would be genuinely anomalous are conditional operators (Where, If, Equal, Less, Greater) — the model currently uses none of these.

2. **No integrity verification.** The ONNX file format has no built-in signing or hash mechanism. A modified model file is byte-for-byte indistinguishable from the original absent external verification. This is a format-level issue, not specific to this model.

| Risk Factor | Level | Detail |
|-------------|-------|--------|
| **Overall Susceptibility** | **71/100 (HIGH)** | |
| Existing Backdoor Detected | **NO** | No conditional ops (Where/If/Equal) found |
| Format Editability | UNKNOWN | ONNX format allows direct graph editing |
| Audit Complexity | MEDIUM | 385 nodes — feasible but time-consuming |
| Parameter Hiding | LOW | Limited capacity in quantized model |
| Camouflage Potential | **HIGH** | 93× MatMulNBits; injected nodes blend in |
| Integrity Verification | **HIGH RISK** | No built-in signing or hash mechanism |

**Identified Injection Points (10):**

| Location Type | Node | Complexity | Detection |
|--------------|------|------------|-----------|
| before_output | `/model/layers.0/conv/Conv_Input` | trivial | moderate |
| before_output | `/model/layers.1/conv/Conv_Input` | trivial | moderate |
| before_output | `/model/layers.3/conv/Conv_Input` | trivial | moderate |
| branch_point | `/model/layers.0/conv/Conv_Input` | moderate | hard |
| branch_point | `/model/layers.0/Add_1` | moderate | hard |
| branch_point | `/model/layers.0/Add_2` | moderate | hard |
| branch_point | `/model/layers.1/conv/Conv_Input` | moderate | hard |
| branch_point | `/model/layers.1/Add_1` | moderate | hard |
| branch_point | `/model/layers.1/Add_2` | moderate | hard |
| branch_point | `/model/layers.2/Add_1` | moderate | hard |

The 10 identified injection points cluster at the conv Concat nodes (where inserting a Where node before the concat could conditionally swap in attacker-controlled values) and at the residual Add nodes (where a conditional branch could selectively suppress the skip connection). The before-output injection points at Concat nodes are rated "trivial complexity" because the concat already merges two inputs — replacing one with a conditional switch requires adding only two nodes (a Constant for the malicious payload and a Where for the conditional).

### Attack Surface by Component

The patterns analysis maps architectural components to the attack classes they structurally enable:

**The residual connections** are the most significant factor. 32 skip connections mean that gradient-based attacks (PGD, C&W, MI-FGSM) can optimize perturbations efficiently across the full depth of the model. Without the residuals, a 282-depth graph would suffer severe gradient degradation; with them, adversarial gradients remain informative from the logit output all the way back to the embedding.

**The GroupQueryAttention blocks** are susceptible to attention hijacking — crafted input sequences that cause the attention mechanism to attend to attacker-chosen positions rather than contextually relevant ones. The GQA design (4:1 head sharing) means that manipulating a single KV head affects 4 query heads simultaneously, potentially amplifying the effect of targeted perturbation.

**The 95 shape operations** (Reshape, Transpose, Slice, etc.) don't compute learned transformations but they do reorganize data layouts. The structural concern is that these operations assume well-formed inputs — a malformed tensor shape could trigger undefined behavior in downstream operators. In practice, ONNX Runtime validates shapes at operator boundaries, but the graph itself has no explicit shape assertions.

**The valid-mode convolutions** (pads=0, kernel=3) mean that the first and last positions in the concatenated sequence see an asymmetric receptive field. Position 0 sees only positions 0–2, while position 1 sees 0–3. At sequence boundaries, this asymmetry could cause systematically different representations that an adversary could exploit.

### Associated Attack Classes

| Attack Class | Techniques | Relevant Components |
|-------------|------------|-------------------|
| Gradient-Based Optimization | FGSM, PGD, C&W | 10 Conv layers, 93 MatMulNBits |
| Multi-Scale & Transfer Attacks | Multi-scale PGD, universal perturbations | 42 Concat/Add fusion points |
| Deep Network Gradient Attacks | PGD, MI-FGSM (Momentum), C&W | 32 residual connections |
| Attention Hijacking | Token manipulation, attention hijacking | 6 GroupQueryAttention blocks |
| Prompt Injection & Carrier | Prompt injection, data smuggling | 95 shape/view operations |
| Boundary & Black-Box | Boundary attacks, gradient masking exploitation | 16 Sigmoid activations |
| Boundary & Edge Attacks | Boundary manipulation, padding exploits | 10 valid-mode convolutions |

### Graph Workflow Targets

These are the nodes GraphSurgeon identifies as highest-priority targets for adversarial analysis or counterfactual surgery:

**Gradient Bottlenecks** (potential defense insertion points):
```
/model/layers.{0-15}/mlp/act_fn/Sigmoid  (16 nodes — all SwiGLU activations)
```

**Feature Fusion Points** (perturbation convergence):
```
/model/layers.{0,1,3,4,6,7,9,11,13,15}/conv/Conv_Input  (10 conv cache merges)
/model/layers.{0-15}/Add_1, Add_2                         (32 residual connections)
```

**Amplification Layers** (signal magnification):
```
/model/layers.{0,1,3,4,6,7,9,11,13,15}/conv/Conv  (10 depthwise convolutions)
```

---

## 10. What The Graph Tells Us That Weights Alone Cannot

Several architectural decisions are only visible at the graph level, not from inspecting weight tensors:

1. **The conv-attention interleaving pattern** — weight files show you have both Conv and attention weights, but the graph shows you the exact sequencing and how the two mechanisms interact through shared residual streams.

2. **The dual-cache architecture** — the graph makes explicit that this model maintains two independent state systems (conv cache and KV cache) during generation, and shows exactly how they thread through the computation.

3. **QK-normalization** — the per-head LayerNorm on queries and keys before attention is only visible in the graph. The norm weights are small initializers that could be overlooked in weight inspection.

4. **The SwiGLU gating in conv blocks** — the conv output goes through a learned gating mechanism (Split + Mul) that is structurally identical to the MLP's SwiGLU but operates on the conv output rather than the MLP intermediate. This is a design choice that is only apparent from the graph topology.

5. **Fused operators** — GroupQueryAttention and SkipSimplifiedLayerNormalization are ONNX Runtime-specific fused ops. Their internal computation is invisible to graph-level analysis, but the graph shows their connectivity and input/output contracts.

6. **The absence of positional encoding nodes** — no explicit position embedding, rotary embedding, or ALiBi bias nodes appear in the graph. Position information must be either internal to GroupQueryAttention or implicitly provided by the convolutional layers' position sensitivity. This is an important architectural detail for understanding the model's length generalization properties.

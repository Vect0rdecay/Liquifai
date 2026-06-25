# GraphSurgeon Analysis Report: LFM2.5-230M-ONNX (Q4)

**Model:** `model_q4.onnx` (4-bit quantized)  
**Source:** LFM2.5-230M-ONNX  
**Analysis Date:** 2026-06-25  
**Tool:** GraphSurgeon (inspect, topology, patterns, motifs, flow)

---

## 1. What This Model Is

This is the smallest member of Liquid AI's LFM2.5 family, a 230 million parameter language model exported to ONNX in 4-bit quantized form. Like its larger 1.2B sibling, it uses a hybrid architecture that interleaves 1D depthwise convolutions with grouped query attention rather than repeating identical transformer blocks. But the 230M variant makes different design choices that become visible at the graph level: a more regular layer interleaving pattern, explicit rotary position embedding nodes exposed in the graph, and a less aggressive GQA head-sharing ratio.

The model has 347 nodes, 230 initializers (stored weights/constants), and a maximum graph depth of 251 hops from input to output. It accepts 23 input tensors and produces 21 outputs. The input count includes `position_ids`, a tensor that does not appear in the 1.2B model's graph at all. Both models have the same 6 attention layers producing KV cache, but the 230M has 8 conv layers (each with one cache slot) versus the 1.2B's 10, which accounts for the difference in total I/O tensor count.

One structural metric that distinguishes this model from its larger sibling is the maximum fan-out: 12 outputs from a single node, double the 1.2B model's 6. This means individual nodes in the 230M graph broadcast their output to more downstream consumers. From a security perspective, the higher fan-out increases the blast radius of any single-node perturbation or injection, because a compromised node's output propagates to twice as many immediate dependents as in the 1.2B architecture.

---

## 2. Model Metadata

| Property | Value |
|----------|-------|
| Graph Name | `lfm2` |
| Total Nodes | 347 |
| Initializers | 230 |
| Max Graph Depth | 251 hops |
| Max Fan-In | 7 inputs (at GroupQueryAttention and RotaryEmbedding nodes) |
| Max Fan-Out | 12 outputs |
| Longest Linear Chain | 1 |
| Quantization | 4-bit (MatMulNBits) |
| Architecture | Hybrid Conv-Attention LLM (14 layers) |

### Inputs (23 tensors)

```
input_ids, attention_mask, position_ids,
past_conv.0, past_conv.1,
past_key_values.2.key, past_key_values.2.value,
past_conv.3,
past_key_values.4.key, past_key_values.4.value,
past_conv.5,
past_key_values.6.key, past_key_values.6.value,
past_conv.7,
past_key_values.8.key, past_key_values.8.value,
past_conv.9,
past_key_values.10.key, past_key_values.10.value,
past_conv.11,
past_key_values.12.key, past_key_values.12.value,
past_conv.13
```

### Outputs (21 tensors)

```
logits,
present_conv.0, present_conv.1,
present.2.key, present.2.value,
present_conv.3,
present.4.key, present.4.value,
present_conv.5,
present.6.key, present.6.value,
present_conv.7,
present.8.key, present.8.value,
present_conv.9,
present.10.key, present.10.value,
present_conv.11,
present.12.key, present.12.value,
present_conv.13
```

### Operator Distribution

| Operator | Count | Role |
|----------|-------|------|
| MatMulNBits | 83 | 4-bit quantized matrix multiplications (projections, MLP layers) |
| Mul | 52 | Element-wise multiplications (gating, scaling) |
| SimplifiedLayerNormalization | 40 | Pre-norm layer normalization (RMSNorm) |
| Add | 28 | Residual connections |
| Reshape | 24 | Tensor shape manipulation for attention heads |
| Transpose | 16 | Dimension reordering (conv layout changes) |
| Slice | 16 | Cache management and output extraction |
| Sigmoid | 14 | SwiGLU activation component |
| RotaryEmbedding | 12 | Explicit rotary position embeddings for Q and K |
| Shape | 9 | Dynamic shape computation |
| Gather | 9 | Index-based lookups |
| Split | 8 | Conv cache splitting |
| Concat | 8 | Conv input concatenation with cached states |
| Conv | 8 | 1D depthwise convolutions (kernel=3, groups=1024) |
| Unsqueeze | 8 | Dimension expansion |
| GroupQueryAttention | 6 | Grouped query attention blocks |
| Cast | 2 | Type casting for attention mask |
| ReduceSum | 1 | Attention mask preprocessing |
| Sub | 1 | Attention mask preprocessing |
| SkipSimplifiedLayerNormalization | 1 | Final fused layer norm + skip |
| GatherBlockQuantized | 1 | Quantized token embedding lookup |

### Graph Topology: Depth Distribution

| Region | Depth Range | Node Count | Layers Covered |
|--------|-------------|------------|----------------|
| Early (~30%) | 0-~83 | 76 nodes | Embedding, mask preprocessing, layers 0-3 |
| Middle (~40%) | ~84-~183 | 205 nodes | Layers 4-10 |
| Late (~30%) | ~184-250 | 66 nodes | Layers 11-13, final norm, lm_head |

---

## 3. The Hybrid Layer Pattern

The model has 14 transformer-style layers, two fewer than the 1.2B variant. The ONNX graph reveals the same two block types, but arranged in a distinctly more regular pattern:

**Convolutional layers** at indices 0, 1, 3, 5, 7, 9, 11, 13 (8 of 14 layers)  
**Attention layers** at indices 2, 4, 6, 8, 10, 12 (6 of 14 layers)

Written out as a sequence: C C A C A C A C A C A C A C

After the initial pair of convolutional layers, the model settles into strict 1:1 alternation between conv and attention for the remaining twelve layers. This is notably simpler than the 1.2B model, which uses a more irregular pattern with stretches of doubled convolutions (CC-A, CC-A, CC-A) in its early layers before shifting toward 1:1 alternation later. The 230M model skips that graduated transition entirely.

The conv-to-attention ratio also shifts. The 1.2B model uses 10 conv layers against 6 attention layers (5:3). The 230M model uses 8 conv against 6 attention (4:3), giving attention proportionally more representation in the smaller model. Whether this reflects a deliberate architectural decision or a constraint of the smaller parameter budget is not determinable from the graph alone, but the structural consequence is clear: at 230M parameters, the model allocates a larger fraction of its capacity to global context aggregation relative to local sequence mixing.

### Input Tensor Naming and the Dual-Cache Architecture

The input tensor naming reveals the same dual-cache system as the 1.2B model, but with one less conv cache slot (the 1.2B has conv caches at layers 0, 1, 3, 4, 6, 7, 9, 11, 13, 15, while the 230M has them at 0, 1, 3, 5, 7, 9, 11, 13). Both models carry separate convolutional state caches and KV caches threaded through the graph, requiring the inference runtime to maintain two independent state management systems during autoregressive generation.

The 230M model's graph name is `lfm2` rather than `main_graph`, confirming this is the LFM2 architecture family.

---

## 4. Inside a Convolutional Block

Tracing the data flow through layer 0 (the first conv block), the structure closely mirrors the 1.2B model's conv blocks with one dimensional difference:

**Step 1, Pre-normalization.** The hidden state enters `SimplifiedLayerNormalization` (RMSNorm), the same simplified variant that skips mean subtraction.

**Step 2, Input projection.** A `MatMulNBits` operation projects the normalized hidden state through 4-bit quantized weights, identical to the 1.2B.

**Step 3, Reshape for convolution.** `Transpose` reorders dimensions from (batch, seq, channels) to convolution-suitable layout. `Shape` and `Gather` compute sequence length metadata dynamically.

**Step 4, Cache concatenation.** `Concat` merges the current sequence representation with `past_conv.0` along the sequence dimension (axis=2). The `Mul` (Neg_Seq_Len) and `Unsqueeze` nodes compute slicing boundaries for cache extraction. This is structurally identical to the 1.2B.

**Step 5, The convolution.** A `Conv` with kernel size 3 and **1024 groups**, half the group count of the 1.2B model's 2048. Since group count equals channel count, this is still fully depthwise, with each of the 1024 channels convolved independently using its own 3-element kernel. Parameter cost per conv layer: 1024 x 3 = 3,072 weights. The receptive field is the same three-position window.

**Step 6, Gated output.** `Split` divides the projected input. `Slice_Conv_Output` extracts the current-position output, `Slice_Cache` extracts the updated cache state (becoming `present_conv.0`). The `Mul_2` node gates the conv output against the split projection.

**Step 7, Output projection and residual.** `MatMulNBits` (out_proj) followed by `Add` with the original hidden state. `Transpose_2` reorders back to (batch, seq, channels) before the output projection.

**Step 8, The MLP.** After the residual add, `SimplifiedLayerNormalization` (ffn_layernorm) feeds into the SwiGLU feed-forward network. Three parallel `MatMulNBits` (gate_proj, up_proj) followed by `Sigmoid`, `Mul` (swish), `Mul` (gate), and `MatMulNBits` (down_proj). A final `Add` creates the second residual connection.

Each conv block produces two residual additions, contributing to the 28 total residual connections in the model.

---

## 5. Inside an Attention Block

Tracing layer 2 (the first attention layer), the overall structure follows the same pre-norm, mixing, residual, MLP, residual pattern as the conv blocks:

**Q/K/V Projection.** Three parallel `MatMulNBits` produce query, key, and value tensors. The subsequent Reshape nodes reveal the head geometry:

The query reshape target is `('batch_size', 'q_seq_heads', 64)` with a second reshape back to `('batch_size', 'sequence_length', 1024)`. Given hidden dimension 1024 and head dimension 64, this gives **16 query heads**.

The key reshape target is `('batch_size', 'kv_seq_heads', 64)` with output dimension 512, giving **8 KV heads**.

This is a **2:1 GQA ratio**, every 2 query heads sharing a single key-value head. The 1.2B model uses a 4:1 ratio (32 query heads, 8 KV heads). The 230M model's less aggressive head sharing means each query head gets a more specialized view of the key-value space, at the cost of a proportionally larger KV cache per parameter.

**Per-head normalization (QK-norm).** Both Q and K pass through `SimplifiedLayerNormalization` (q_norm, k_norm) sandwiched between Reshape operations that isolate individual heads. This stabilizes attention scores by normalizing query and key vectors before computing dot products. Same technique as the 1.2B.

**Rotary Position Embedding.** This is where the 230M graph diverges most visibly from the 1.2B. Two explicit `RotaryEmbedding` nodes (`q_rotary/RotaryEmbedding` and `k_rotary/RotaryEmbedding`) appear after QK-norm, applying rotary position encodings to the normalized query and key tensors. These are ONNX Runtime custom ops, each taking 7 inputs including the `position_ids` tensor that feeds in from the graph's third input.

The 1.2B model's graph contains zero RotaryEmbedding nodes. Its positional encoding was presumably fused inside the GroupQueryAttention operator, making the position encoding mechanism invisible to graph-level analysis. The 230M model exposes the full RoPE pipeline as separate nodes, which means the position encoding strategy is structurally auditable in this model but not in its larger sibling.

This has a direct analytical consequence. In the 230M graph, you can trace exactly how position information enters the attention computation: `position_ids` flows into `RotaryEmbedding`, which transforms Q and K before they reach `GroupQueryAttention`. In the 1.2B graph, you have to infer the existence of RoPE from the absence of any other positional mechanism. The same architecture, two different levels of graph-level visibility.

**GroupQueryAttention.** The fused attention operator takes 7 inputs: rotary-embedded Q, rotary-embedded K, V, past_key, past_value, and sequence metadata. Internal computation (attention scores, softmax, value weighting) remains opaque inside this custom op.

**Output projection and residual.** `MatMulNBits` (o_proj) followed by `Add`.

**MLP.** Identical SwiGLU pattern as conv blocks.

---

## 6. The Attention Mask Subgraph

The attention mask preprocessing follows the same pattern as the 1.2B model:

`ReduceSum` on `attention_mask` (sum across sequence dimension) produces a per-batch token count. `Sub` computes an offset. `Cast` converts to the appropriate dtype. `Shape` and `Gather` extract metadata. The result feeds into all 6 GroupQueryAttention nodes as sequence-length information.

This subgraph executes at depth 0-5 in the graph, before any learned computation. It constitutes the same in-graph trust boundary identified in the 1.2B analysis: the model transforms the attention mask internally and passes the result to attention operators without further validation. Malformed mask values propagate through unchecked.

---

## 7. Token Embedding

`GatherBlockQuantized` at `/model/embed_tokens/GatherBlockQuantized` performs quantized embedding lookup into a block-quantized embedding table. Output hidden dimension is 1024 (half the 1.2B model's 2048). This is the only `GatherBlockQuantized` in the graph.

The embedding output feeds into the first conv layer's normalization. Unlike the 1.2B model, positional information is not a mystery here: the explicit `RotaryEmbedding` nodes in each attention layer receive `position_ids` directly from the graph input. The convolutional layers still rely on their inherent position-sensitivity from the kernel's local receptive field.

---

## 8. The Output Head

Two final nodes:

1. `SkipSimplifiedLayerNormalization` at `/model/layers.14/final_norm_layernorm/SkipLayerNorm` fuses the skip connection with RMSNorm. This is the only instance of this operator in the graph, used exclusively at the output.

2. `MatMulNBits` at the `logits` output projects from hidden dimension 1024 to vocabulary size, producing raw logits. No softmax in the graph.

---

## 9. Structural Security Analysis

### Structural Findings Summary

| # | Finding | Category | Key Nodes |
|---|---------|----------|-----------|
| 1 | Dynamic Input Processing (4 shape ops) | adversarial_perturbation | `/model/layers.0/conv/Slice_Conv_Output` |
| 2 | Gradient Highway (28 skip connections) | adversarial_perturbation | `/model/layers.0/Add_1` (first of 28) |
| 3 | In-Graph Preprocessing Trust Boundary | adversarial_perturbation | `/model/attn_mask_reformat/attn_mask_subgraph/Sub` |
| 4 | Multiple Unprotected Feature Fusion (36 fusion, 0 norm) | adversarial_perturbation | All Add/Concat nodes |
| 5 | No Explicit Normalization Layers (8 conv, 0 BatchNorm) | adversarial_perturbation | Conv layers (likely fused during export) |
| 6 | ShadowLogic Injection Susceptibility (71/100) | shadowlogic_injection | 10 identified injection points |

### Structural Pattern Counts

GraphSurgeon identifies 92 structural patterns across the graph. The per-node multimodal fusion count (72) reflects individual nodes where two or more computation paths converge; the aggregated feature fusion point count (36) groups these by functional role.

| Pattern Type | Count | Example Nodes |
|-------------|-------|---------------|
| Multimodal Fusion Points (2+ inputs, per-node) | 72 | Add, Concat, gating Mul, Conv_Input nodes across all layers |
| High Fan-In Nodes (7 inputs) | 6 | `layers.{2,4,6,8,10,12}/attn/GroupQueryAttention` |
| Perturbation Fusion (Concat, axis=2) | 8 | `layers.{0,1,3,5,7,9,11,13}/conv/Conv_Input` |
| Residual Chain (no gradient regularization) | 28 blocks | `layers.*/Add_1`, `layers.*/Add_2` |
| Saturating Activations | 14 | `layers.*/mlp/act_fn/Sigmoid` |
| Shape Operation Chain | 81 ops | Reshape, Transpose, Slice, Gather, Split, etc. |
| Valid Conv (pads=0, boundary-sensitive) | 8 | `layers.{0,1,3,5,7,9,11,13}/conv/Conv` |
| ShadowLogic: Deep Architecture Risk (depth=251) | 1 | Full graph |
| In-Graph Preprocessing | 5 layers | Mask subgraph: Sub, Cast, Gather, Mul nodes |
| **Total Structural Patterns** | **92** | |

### Gradient Flow Characteristics

28 residual connections create unobstructed gradient highways from the output back to the embedding. No Dropout, DropPath, or noise injection nodes exist anywhere in the graph. This is expected for an inference-mode export, but it means gradient-based adversarial methods (PGD, C&W) can optimize perturbations with full gradient fidelity across the entire depth.

The 14 Sigmoid activations in SwiGLU blocks are the only non-linear bottlenecks in the gradient path. Sigmoid saturates at extreme input magnitudes, but this is a side effect of the activation choice, not a defense. The shorter gradient path (250 hops vs. 282 in the 1.2B) and fewer residual connections (28 vs. 32) mean slightly less gradient signal attenuation in absolute terms, though the proportional structure is the same.

### Fusion Point Topology

GraphSurgeon identifies 36 fusion operations where signals from multiple computation paths converge (vs. 42 in the 1.2B model). The 6 GroupQueryAttention nodes remain the widest fusion points at 7 inputs each. The RotaryEmbedding nodes also register as 7-input high fan-in nodes, though their inputs are structurally constrained (position_ids, Q or K tensor, and configuration constants).

The conv cache concat points carry the same structural characteristic as in the 1.2B: past and present states concatenate with no gating or normalization at the boundary. The kernel-3 convolution then determines interaction between past and present, with no mechanism to detect or reject anomalous cached states.

### Gadget Analysis

GraphSurgeon's gadget framework identifies **44 exploitable graph primitives** classified by type and position within the execution graph. A gadget is a structural motif that an adversary can leverage as a building block for attacks, whether for gradient-based optimization, perturbation amplification, or injection camouflage.

| Gadget Type | Count | Role |
|------------|-------|------|
| Skip Connection | 28 | Gradient highways enabling efficient PGD/C&W convergence |
| Shape Operation | 4 | Tensor layout manipulation without semantic validation |
| Spatial Attention | 3 | Channel gating mechanisms (partial defensive value) |
| Multimodal Fusion Point | 3 | Multi-branch signal convergence for coordinated attacks |
| Cross-Modal Fusion (Late) | 2 | Output-proximal fusion enabling late-stage perturbation injection |
| Multimodal Fusion (Aggregate) | 1 | Global fusion pattern across the architecture |
| In-Graph Preprocessing | 3 | Trust boundary nodes that transform inputs without validation |

**Position distribution across the graph:**

| Region | Gadget Count | Percentage |
|--------|-------------|------------|
| Early (layers 0-3) | 13 | 30% |
| Middle (layers 4-10) | 24 | 54% |
| Late (layers 11-13) | 7 | 16% |

The gadget distribution in the 230M model differs meaningfully from the 1.2B. The middle layers still hold the majority of gadgets (54%), but the late region contains **7 gadgets** compared to just 1 in the 1.2B model. This concentration of output-proximal exploitable primitives is significant: late-stage gadgets sit closer to the logit output, meaning perturbations introduced through them have fewer normalization or gating operations to pass through before affecting model predictions.

The 2 cross-modal late-fusion gadgets (versus 1 in the 1.2B) contribute to this late-stage exposure. The 230M architecture's shallower depth and more regular layer alternation leave fewer intermediate layers between the final fusion points and the output head.

The 5 highest-priority gradient highway gadgets have skip-connection distances ranging from 8 to 17 hops. These per-block distances are much shorter than the 1.2B model's 251-273 hop highways, reflecting the shallower graph. However, the shorter distances do not reduce attack effectiveness. They indicate tighter, more efficient gradient loops within each block, and the 28 skip connections still chain together to create an unobstructed gradient path across the full 251-hop depth.

GraphSurgeon also identifies 3 defensive features: channel gating mechanisms at the first three SwiGLU activations. These attention-like gating patterns structurally reduce the effectiveness of sparse patch attacks by forcing perturbations through a learned gate, though they do not constitute a robust defense against targeted adversarial optimization.

### ShadowLogic Assessment

No backdoor indicators detected. No conditional operations (Where, If, Equal) exist in the graph. Injection susceptibility scores **71/100 (HIGH)**, identical to the 1.2B model.

The score distribution:

| Risk Factor | Level | Detail |
|-------------|-------|--------|
| **Overall Susceptibility** | **71/100 (HIGH)** | |
| Existing Backdoor Detected | **NO** | No conditional ops found |
| Format Editability | UNKNOWN | ONNX format allows direct graph editing |
| Audit Complexity | MEDIUM | 347 nodes, feasible but time-consuming |
| Parameter Hiding | LOW | Limited capacity in quantized model |
| Camouflage Potential | **HIGH** | 83x MatMulNBits; injected nodes blend in |
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

The injection point distribution is functionally identical to the 1.2B model. The conv Concat nodes and early residual Add nodes cluster as the highest-priority targets. The before-output injection points at Concat nodes remain "trivial complexity" because the concat already merges two inputs, and replacing one with a conditional switch requires only two additional nodes (a Constant and a Where).

A representative injection scenario proceeds in three stages. First, trigger detection: additional nodes after the input layer compare incoming token patterns against an attacker-defined signature. Second, output override: at a targeted Concat or Add node (such as `/model/layers.0/conv/Conv_Input`), a Where node conditionally substitutes the normal computation with an attacker-controlled Constant tensor when the trigger is detected. Third, persistence: the backdoor survives format conversions (ONNX to TensorRT), survives fine-tuning on clean data (because the trigger pathway is rarely activated), and passes all standard test suites on non-triggered inputs. The model currently uses ReduceSum, Slice, Gather, and Concat operations that would provide statistical cover for injected trigger-detection nodes. The only genuinely anomalous additions would be conditional operators (Where, If, Equal), which the model currently uses none of.

### Attack Surface by Component

| Attack Class | Techniques | Relevant Components |
|-------------|------------|-------------------|
| Gradient-Based Optimization | FGSM, PGD, C&W | 8 Conv layers, 83 MatMulNBits |
| Multi-Scale & Transfer Attacks | Multi-scale PGD, universal perturbations | 36 Concat/Add fusion points |
| Deep Network Gradient Attacks | PGD, MI-FGSM (Momentum), C&W | 28 residual connections |
| Attention Hijacking | Token manipulation, attention hijacking | 6 GroupQueryAttention blocks |
| Prompt Injection & Carrier | Prompt injection, data smuggling | 81 shape/view operations |
| Boundary & Black-Box | Boundary attacks, gradient masking exploitation | 14 Sigmoid activations |
| Boundary & Edge Attacks | Boundary manipulation, padding exploits | 8 valid-mode convolutions |

### Graph Workflow Targets

These are the nodes GraphSurgeon identifies as highest-priority targets for adversarial analysis or counterfactual surgery. The gadget analysis reinforces these classifications: 7 of 44 gadgets cluster in the late layers (11-13), giving the 230M model a notably higher concentration of output-proximal attack surface than the 1.2B.

**Gradient Bottlenecks** (potential defense insertion points):
```
/model/layers.{0-13}/mlp/act_fn/Sigmoid  (14 nodes, all SwiGLU activations)
```
The 5 flagged gradient highway gadgets span distances of 8-17 hops per block. These short, tight gradient loops chain across 28 residual connections to create an unobstructed path spanning the full 251-hop graph depth.

**Feature Fusion Points** (perturbation convergence):
```
/model/layers.{0,1,3,5,7,9,11,13}/conv/Conv_Input  (8 conv cache merges)
/model/layers.{0-13}/Add_1, Add_2                    (28 residual connections)
```
The gadget framework identifies 3 multimodal fusion gadgets and 2 cross-modal late-fusion gadgets within this set. The 2 late-fusion gadgets (versus 1 in the 1.2B) represent output-proximal nodes where perturbations can bypass most of the graph's normalization layers.

**Amplification Layers** (signal magnification):
```
/model/layers.{0,1,3,5,7,9,11,13}/conv/Conv  (8 depthwise convolutions)
```

---

## 10. What The Graph Tells Us That Weights Alone Cannot

1. **The layer interleaving pattern simplifies at smaller scale.** The 1.2B model uses an irregular CC-A, CC-A, CC-A pattern that shifts toward C-A alternation in later layers. The 230M model starts with a single CC pair and then strictly alternates C-A for the remaining twelve layers. This regularity is only visible in the graph topology, not from counting weight tensors.

2. **Rotary position embedding is architecturally exposed.** The 230M graph contains 12 explicit `RotaryEmbedding` nodes (2 per attention layer), taking `position_ids` as an input tensor. The 1.2B graph contains zero such nodes and has no `position_ids` input, meaning its RoPE implementation is fused inside GroupQueryAttention. The same underlying mechanism produces two different levels of graph-level auditability depending on model size or export configuration.

3. **The GQA head-sharing ratio changes with scale.** The 1.2B model uses 4:1 head sharing (32 Q heads, 8 KV heads). The 230M model uses 2:1 (16 Q heads, 8 KV heads). The KV head count stays constant while query heads halve, meaning the smaller model gives each query head a more independent view of the key-value space. This is a non-obvious scaling decision: the parameter-constrained model is the one with less KV compression.

4. **The dual-cache architecture persists at the smaller scale** with the same structural pattern of separate conv and KV caches threaded through the graph. The cache tensor naming follows the layer index exactly, making the conv-vs-attention assignment for each layer immediately readable from the input/output tensor list alone.

5. **Conv group count scales with hidden dimension.** The 1.2B model uses 2048 groups (matching its 2048 hidden dim), and the 230M uses 1024 groups (matching its 1024 hidden dim). Both are fully depthwise. The convolution remains extremely parameter-efficient at both scales: 3,072 parameters per conv layer at 230M vs. 6,144 at 1.2B.

6. **The fused output head operator appears at both scales.** `SkipSimplifiedLayerNormalization` is used exactly once in both models, exclusively at the final output position. This suggests it is an export-time optimization specific to the terminal position rather than a general-purpose operator the architecture makes available throughout the graph.

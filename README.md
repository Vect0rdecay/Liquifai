# Liquifai

Graph-level analysis of [Liquid AI's](https://www.liquid.ai/) foundation models in ONNX format, using [GraphSurgeon](https://github.com/Vect0rdecay/GraphSurgeon).

## What This Is

Liquid AI builds LFMs (Liquid Foundation Models) on an architecture that breaks from the standard transformer pattern. Instead of repeating identical attention-plus-MLP blocks, LFMs interleave 1D depthwise convolutions with grouped query attention layers, drawing on dynamical systems theory and signal processing rather than pure self-attention. The result is a hybrid architecture with a dual-cache system (convolutional state caches alongside traditional KV caches) that behaves fundamentally differently from GPT-style models at the graph level.

This repo contains GraphSurgeon analysis reports for LFM models exported to ONNX. Each report traces the full computational graph of a model, documents its topology and operator distribution, and explains the architectural patterns that make these models structurally distinct.

## What GraphSurgeon Does

GraphSurgeon reads an ONNX model file and performs graph-level inspection across several dimensions: topology analysis (depth, fan-in, linear chains), pattern detection (residual highways, fusion points, cache threading), motif identification (block-level architectural patterns), flow tracing (data paths through representative blocks), and ShadowLogic injection susceptibility scoring.

The output is a structural map of the model that reveals architectural decisions invisible to weight inspection or behavioral testing alone.

## Reports

| Model | Format | Report |
|-------|--------|--------|
| LFM2.5-1.2B-Instruct | ONNX Q4 | [Analysis](LFM2.5-1.2B-Instruct-ONNX%20(Q4).md) |

## What Each Report Covers

**Architecture.** The layer interleaving pattern, what block types exist, how they differ structurally, and why the arrangement matters for inference behavior.

**Block-level tracing.** Step-by-step data flow through one representative instance of each distinct block type, naming every operator and showing how tensors reshape and route through the computation.

**Graph topology.** Depth distribution, operator counts, fan-in characteristics, fusion point topology where multiple computation paths converge, and gradient flow across the residual chain.

**Security-relevant motifs.** Where the analysis surfaces structurally notable patterns (ShadowLogic injection susceptibility, unprotected cache boundaries, gradient highways), it flags them alongside the architectural context that produces them.

## Disclaimer

An LLM assists with analyzing the raw data GraphSurgeon produces and with drafting the initial version of each report. All findings are reviewed and verified against the actual ONNX graph before publication.

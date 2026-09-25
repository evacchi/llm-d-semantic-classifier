# ADR-0006 — Vendor `modernbert.rs` to support YaRN rotary scaling

**Status:** accepted · **Date:** 2026-09-25 · **Supersedes:** nothing

## Context

`candle-transformers` 0.11 builds ModernBERT's rotary frequencies as plain
`1/theta^(i/dim)` and ignores `rope_scaling` — its `Config` does not model the
field at all.

Every model in the vSR **Vela** family (all twelve: base, Domain, Shield, Guard,
Safety, Hazard, FactCheck, Halu, Modality, Feedback, Reranker, Embedding) extends
ModernBERT's context 8192 → 32768 using **YaRN**, which reshapes that frequency
spectrum: 22 of 32 components change, by up to 75%, and cos/sin are additionally
scaled by `0.1·ln(factor)+1` = 1.1386.

Serving a Vela checkpoint through upstream therefore produces embeddings of the
right **magnitude** and the wrong **direction** — measured hidden-state cosine
0.92–0.96 against the reference, and a top-1 logit of 11.4986 where the checkpoint
says 9.9652. Nothing errors, and argmax often survives, so it is invisible without
a parity check against the original implementation. It was invisible here until
one was run.

This is **not a candle defect**. Candle never claimed YaRN support for ModernBERT;
it is a missing capability. Confirmed by running the architecture authors' own
`answerdotai/ModernBERT-base` (no `rope_scaling`) through the same harness, where
candle reproduces PyTorch to **five decimal places**.

## Decision

**Vendor `candle-transformers/src/models/modernbert.rs` into
`src/modernbert.rs` and add YaRN there.** Upstream is not modified and nothing is
pushed to `huggingface/candle`.

The BERT path is **not** vendored. It continues to use upstream
`candle_transformers::models::bert`, which reproduces PyTorch to six decimal
places unchanged.

### Why not the alternatives

| option | rejected because |
|---|---|
| PR to `huggingface/candle` | Vela stays unservable until it merges *and* ships in a release. Maintainer chose to keep the change internal. |
| Fork candle, `[patch.crates-io]` by rev | A whole repo to rebase forever, for a diff confined to one function in one file. |
| Vendor all of `candle-transformers` | 3.1 MB, 209 files, to change one function. |
| Reimplement ModernBERT ourselves | Forbidden by `AGENTS.md`, and rightly — this is a copy with one change, which is a different thing. |

### Why this is a copy, not a reimplementation

`modernbert.rs` is self-contained: 504 lines importing only **public** `candle`
and `candle_nn` APIs, with both helpers (`prepare_4d_attention_mask`,
`get_local_attention_mask`) defined inside the file. The diff against upstream is
confined to `RotaryEmbedding::new` and the `RopeScaling` struct it reads.
Everything else is upstream's, byte-for-byte, so a re-sync is a three-way merge
rather than a rewrite.

## Consequences

**Accepted cost.** We now own a 504-line copy of an upstream file. It will drift.
The file header records which version it came from and what changed, and this ADR
records why; without both, it becomes untraceable within a release or two.

**Re-sync procedure.** Diff `src/modernbert.rs` against the same path in the
candle version pinned in `Cargo.toml`, re-apply the YaRN block, and re-run the
parity tests. Both must hold:

- `answerdotai/ModernBERT-base` (no scaling) — exact to 5 dp, proving the plain
  path is untouched;
- `llm-semantic-router/Vela-1.0-Encoder-307M-Domain` (YaRN) — matches PyTorch.

**Scaling types we do not implement are still refused at load** (`longrope`,
`dynamic`, …). Refusing is the only point where an unsupported scaling is
visible; once the service is answering, nothing downstream can tell.

**Validated before adoption.** The change was proven on-cluster against a patched
candle before any of it was brought in-tree: the top-1 logit delta went from
**1.5334 to 0.0004**, and hidden states matched to five decimal places.

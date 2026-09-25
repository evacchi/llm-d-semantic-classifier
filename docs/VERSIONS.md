# Pre-1.0 Phase and Version Roadmap

The project remains intentionally pre-1.0. Phase names describe maturity; versions remain `0.x`.

| Phase | Version | Focus | Promotion evidence |
|---|---:|---|---|
| Phase 1 — MVP | **0.1** | Rust service + runtime trait + Candle + real ModelCar + dummy gateway | functional E2E + initial latency evidence |
| Phase 2 — Runtime hardening | **0.20** | scheduler/deadlines/load shedding/readiness/shutdown/metrics | deterministic failure/concurrency tests |
| Phase 2.1 — Performance characterization | **0.21** | CPU/GPU, cache, input length, concurrency, topology | repeatable homelab benchmark report |
| Phase 2.2 — Cache/session optimization | **0.22** | exact-result + optional feature cache + recovery/abstention | crash/cache-loss correctness |
| Phase 2.3 — Multi-signal runtime | **0.23** | multiple classifiers, partial failure, per-classifier lanes | domain/complexity/sensitivity contract suite |
| Phase 2.4 — Runtime pluggability | **0.24** | backend conformance, atomic model swaps, library-ready core | backend/lifecycle conformance |
| Phase 3 — Kubernetes production-like | **0.30** | registry/disconnected/scaling/rollout/security | full system suite |
| Phase 3.1 — the AI Gateway integration | **0.31** | replace dummy boundary with real the AI Gateway integration | real gateway E2E |
| Phase 3.2 — Targeted inference optimization | **0.32** | measured bottlenecks only | equivalent accuracy + before/after p99 |
| Phase 4 — Feedback ecosystem | **0.40** | telemetry/artifact hooks for SDG/Training/Eval | external-loop contract, not training in service |

## 0.1 MVP

### Unreleased fixes

- ModelCar content digests now include the BERT architecture `config.json`,
  alongside weights, tokenizer, and pooling configuration. All existing
  ModelCar content digests change, even when artifact bytes are unchanged;
  regenerate any recorded digest expectations. Cache identities derived from
  these digests also change. Mounts missing `config.json` fail readiness checks.
  Hugging Face revision pins and OCI image digests are unaffected. Fixes
  [#4](https://github.com/llm-d-incubation/llm-d-semantic-classifier/issues/4).

### Scope

Prove the shape of the service:
- Rust server and protobuf contract;
- `ClassifierRuntime` abstraction;
- Candle first backend;
- model/tokenizer loaded once and resident;
- readiness after successful warmup;
- one real sensitivity embedding classifier fixture;
- external model delivered as OCI ModelCar;
- exact-result cache with versioned key;
- basic bounded inference queue;
- dummy gateway integration;
- Kubernetes same-pod sidecar and ClusterIP RTT measurements;
- queue/tokenize/forward/total timing separation;
- restart with complete context recomputes correctly.

Not required: distributed cache, custom kernels, vLLM backend, multiple signals, RL/training, production control plane, hard universal 20 ms SLA.

## 0.2 — release notes

### Classification heads are now the default decision rule

llm-d-sc previously ranked by cosine similarity against taxonomy anchors and
discarded the trained classifier every ModelCar ships. Measured on 552 real rows,
anchor ranking scores **below the majority-class baseline** on 3 of 5 signals
(complexity −2.54, cx2 −3.99, sensitivity −2.73): cosine-to-a-centroid is a
rank-1 decision rule and cannot express those boundaries.

The trained head now runs whenever a checkpoint ships one, reproducing it to
**six decimal places**. There is no flag: a deployment that wants anchor ranking
gets it by serving an embedding-only artifact, which is an honest statement of
what it has. Scores are softmax probabilities, not similarities.

**Cost, measured through the gateway on 3 replicas: 3.5% of classification
throughput** and +0.14 to +0.63 ms of p50. Below saturation the two are
indistinguishable.

`RuntimeMetadata.ranking_mode` reports which rule is live, and it is logged at
load. The difference between the two never surfaces as an error — only as worse
routing — so it has to be observable.

### Replicas now add capacity

The Praxis filter held one lazily-connected HTTP/2 channel. `kube-proxy`
load-balances at CONNECT time, so a multiplexed channel rode that one connection
forever and additional replicas sat idle: 1→2 replicas previously moved capacity
462 → 463/sec (**1.00×**), with one pod serving 10,404 classifications and the
other serving 0.

With `balance_endpoints: true` against a **headless** Service, measured
**1.96× at 2 replicas** and per-pod shares of 33.1/33.2/33.7 at 3. Coverage at
1200 offered rps went from ~41% to **101%**. Off by default: it changes the
connection topology of a deployed proxy, which should be a decision.

### ModernBERT support, including YaRN-scaled checkpoints

llm-d-sc can now serve ModernBERT alongside BERT. Architecture is detected from
`model_type`; a missing value means BERT (every previously published ModelCar
omits it) and an **unknown** value is an error rather than a fallback.

`src/modernbert.rs` is **vendored from `candle-transformers` 0.11 with YaRN
rotary scaling added** — see [ADR-0006](adr/0006-vendored-modernbert.md).
Upstream ignores `rope_scaling`, and all twelve models in the vSR Vela family use
YaRN to reach a 32k context, so serving one through upstream produced embeddings
of the right magnitude and the wrong direction while argmax usually survived. The
BERT path is NOT vendored and is unchanged.

**Scaling types we do not implement are refused at load** (`longrope`, `dynamic`,
…). Refusing is the only point at which an unsupported scaling is visible.

### Cache: `exact` remains the default, now for a measured reason

The L2 semantic tier sits between `embed` and `rank`, so its lookup key **is the
model forward's output**. A hit skips only `rank` — microseconds — while paying a
Redis KNN round trip on top of a forward that already cost ~31 ms (99.4% of
request latency). No hit rate or load level makes that profitable; under load the
round trip gets worse.

An L0 text prefilter (MinHash, consulted *before* the forward, where a hit would
skip the whole 31 ms) was built and measured against the full model on 6,000 real
prompts. Its error rate is **flat at 7.6–10.2%** from J≥0.50 to J≥0.80 while the
hit rate collapses 50% → 4.6%: text similarity and label agreement are close to
independent, so no threshold separates them. Errors are asymmetric — about 20% of
TRIVIAL prompts are served a WORK answer at every threshold. **It is off by
default and should stay off.**

### Classification coverage is now a first-class signal

Classifier saturation is invisible to every signal an operator normally watches:
above capacity the gateway fails open, throughput tracks offered load, errors stay
at zero, and p50 *improves* as requests bypass classification. Measured on 3
replicas, coverage falls ~101% → 43% between 1600 and 3200 offered rps with zero
errors and p50 dropping 562 ms → 2 ms.

`llm_d_sc_classify_total` now carries a `classified` dimension so coverage is an
unambiguous ratio. The judgement is made in code because it is subtle:
`LOW_CONFIDENCE`, `UNMAPPED_LABEL` and `ABSTAIN` are *successful* classifications
the router declined to act on.

```promql
sum(rate(llm_d_sc_classify_total{classified="true"}[5m]))
  / sum(rate(llm_d_sc_classify_total[5m]))
```

**Alert on this, not on latency or errors.**

### Fixes

- ModelCar required files key on `modules.json`, not architecture. A plain
  HuggingFace `ForSequenceClassification` checkpoint has no `1_Pooling/` and must
  still serve; the readiness gate was stricter than the loader it gates.
- The L2 isolation tag carries the artifact digest, so a rebuilt artifact under an
  unchanged revision can no longer miss in L1 and hit in L2.
- `src/bin/playground.rs` sets `context_completeness`; the branch did not compile.

### Known gaps

- `identity_tag` aliases when a field contains a pipe. L1 is immune (length-prefixed).
- Vela **quality** numbers are being re-measured now that YaRN is honoured; its
  **capacity** figure stands: 53.9 cls/sec against 449.6 for a 6-layer BERT
  ModelCar, i.e. 8.3× slower.

## 0.20 Runtime hardening

Make the service trustworthy before making it clever:
- bounded queue;
- per-job deadline;
- queued cancellation;
- load shedding;
- liveness/readiness distinction;
- graceful shutdown/drain;
- structured errors;
- metric-cardinality bounds;
- prompt redaction;
- deterministic concurrency configuration.

## 0.21 Performance characterization

Establish named hardware profiles and benchmark:
- 0/50/90/100% cache hit where useful;
- 32/64/128/256-token inputs;
- concurrency 1/2/4/8/16/32;
- CPU worker/math/tokenizer thread configurations;
- GPU if available;
- localhost sidecar;
- same-node ClusterIP;
- cross-node when possible.

Only after this phase should an absolute p99 threshold become a hard gate for a named hardware profile.

## 0.22 Cache/session optimization

Keep three concepts distinct:
1. resident model/tokenizer runtime state;
2. exact-result cache;
3. optional session/prefix/feature cache.

Routing/session authority remains the AI Gateway. Complete cache loss must not silently turn `continue` into a confident downgrade; insufficient context yields abstention.

## 0.23 Multi-signal runtime

- registry of multiple classifiers;
- independent signal status;
- partial success;
- classifier-specific queue/concurrency limits;
- serial vs parallel measurement;
- failed sensitivity never becomes a benign low-sensitivity result.

## 0.24 Runtime pluggability

- backend conformance suite;
- Candle passes it;
- test/mock backend passes it;
- candidate load/warm -> atomic active-handle switch;
- old in-flight work drains on old immutable handle;
- `runtime-core` does not depend on network server so future library embedding is possible.

## 0.30 Kubernetes production-like

- private OCI registry;
- digest pinning;
- no runtime Hugging Face download;
- egress-denied/disconnected start;
- random UID/read-only model data;
- NetworkPolicy;
- 1->N scaling;
- rolling service/model revision;
- pod/node disruption;
- resource pressure;
- metrics/provenance evidence.

## 0.32 Optimization

Only optimize measured bottlenecks: tokenizer/threading, copies/allocations, mask reuse, sequence buckets, dtype, CPU affinity/NUMA, true local/flash attention, custom kernels/ops, alternative runtimes. Every change needs accuracy parity and comparable latency evidence.

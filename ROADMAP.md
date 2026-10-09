# Roadmap

Where `llm-d-sc` goes next, and why in that order.

This is a proposal for discussion, not a commitment. Work is organized as
parallel workstreams, not as strictly sequential phases: reliability fixes and
measurement must continue throughout, while the product tracks can progress at
different rates. `docs/known-gaps.md` remains the authoritative record of what
is missing today; this document explains the intended direction and priority.

## Product direction

The project supports two distinct classification workloads. They have different
contracts, ownership boundaries, and optimization goals, so the roadmap keeps
them separate.

### Zero/one-shot classification: always classify

The primary near-term workload is request-intrinsic classification over an
unordered domain of labels. For every eligible request, `llm-d-sc` should
produce a classification (or an explicit low-confidence/abstention result).
Examples include domain, sensitivity, safety, intent, language, modality, and
coarse complexity.

This is the natural home for anchor-based engineering. A user supplies or
updates labelled anchors; the service embeds them and ranks a request against
the configured label set. It should work without retraining, use small
CPU-friendly models, and build first on model types and implementations that
are already robust in the project.

Candidate-conditioned scoring is a related, but distinct, signature:
`score = f(request, candidate)`. The Gateway supplies eligible candidates and
`llm-d-sc` returns suitability per candidate. The Gateway retains the final
selection responsibility, combining those scores with cost, policy, data
sovereignty, endpoint health, capacity, latency, and stickiness.

### Multi-turn classification: classify when needed

The second workload is classification over an ordered set of labels or states
across a conversation. It should not run blindly on every turn. The Gateway
owns session identity, turn accounting, and routing state; it asks for a
reclassification when a material change is detected or policy requires it—for
example, a change in saturation, flow-control state, or another routing-relevant
conversation signal.

This avoids treating session state as an implementation detail of a stateless
classifier. The classifier supplies scores and a stable contract; the Gateway
decides when to invoke it and how to prevent route flapping.

## Delivery priorities

1. **Production foundations first.** Fix correctness issues and add the
   observability, health, lifecycle, and deployment infrastructure needed to
   operate the service safely.
2. **Benchmark and evaluate continuously.** Carry benchmarking, model
   evaluation, and refinement with the AI Innovation team/MLflow alongside
   every workstream; do not defer evidence until the end of a phase.
3. **Prioritize existing support and configurable anchors.** Start from the
   supported embedding/anchor path and small models that work well on CPU.
   Select models using repeatable evidence before adding new runtime families.
4. **Defer complex architectures.** Explore Vela, decision models,
   Jev-like models, and vLLM-backed paths only after the core workload and its
   evaluation baseline are established.

The v0.2 campaign makes the production priority concrete: saturation can fail
open and appear healthy, approximate input caching did not predict label
agreement, and the model forward sets the observed throughput floor. Those are
reasons to improve observability and evaluate each change, not reasons to make
low-level throughput optimization the first product milestone.

---

## Workstream 1 — Production foundations

This is the first delivery priority and remains active for the lifetime of the
project. It covers bug fixes, observability, and the operational infrastructure
that makes benchmark and production results trustworthy.

| Work | Outcome | Existing issue |
| --- | --- | --- |
| Metrics endpoint | Export classification coverage, per-stage latency histograms, queue depth, admission rejections, cache hit ratio by tier, and classifier/model revision and digest. | [#11](https://github.com/llm-d-incubation/llm-d-semantic-classifier/issues/11) |
| Coverage alert and semantics | Document coverage as `classified / total`; distinguish an invalid zero-delta read from genuine 0%, and count `LOW_CONFIDENCE`, `UNMAPPED_LABEL`, and `ABSTAIN` as completed classifications. | new |
| Health-check endpoint | Expose readiness so an orchestrator can probe actual service state. | known gaps |
| Graceful drain on shutdown | Define drain semantics so scaling or rollout does not drop in-flight work. | [#13](https://github.com/llm-d-incubation/llm-d-semantic-classifier/issues/13) |
| Cache-hit admission fix | Ensure a cache hit cannot be rejected with `RESOURCE_EXHAUSTED`. | [#2](https://github.com/llm-d-incubation/llm-d-semantic-classifier/issues/2) |
| Exact-cache eviction | Replace FIFO with LRU where measurements show exact caching is valuable. | [#9](https://github.com/llm-d-incubation/llm-d-semantic-classifier/issues/9) |
| CPU-limited deployment behavior | Benchmark and document the service under pod CPU limits rather than relying on unconstrained-host figures. | [#14](https://github.com/llm-d-incubation/llm-d-semantic-classifier/issues/14) |
| Scaling investigation | Establish replica linearity under demand-sufficient load before deciding whether coverage-keyed autoscaling is warranted. | new |

Coverage-keyed autoscaling is deliberately an investigation, not a committed
near-term feature. It depends on the metrics, alert semantics, and scaling
evidence above. Per-request cancellation is also deferred until it is shown to
matter for an active workload.

**Success criteria:** operators can probe the service, observe classification
coverage and saturation, safely drain it, and reproduce CPU-constrained
throughput figures.

---

## Workstream 2 — Continuous benchmarking, evaluation, and model selection

Benchmarking is not a later phase. Every material runtime, model, anchor, and
contract change needs a comparable evaluation row. Coordinate the refinement
loop with the AI Innovation team and MLflow so datasets, runs, artifacts, and
results are traceable.

| Work | Outcome |
| --- | --- |
| Reproduce benchmark results | Obtain an independent run on a second environment and record hardware, limits, load shape, and model revision. |
| Evaluate supported models first | Compare the existing supported embedding models and trained-head/anchor variants before introducing another architecture. Favor small CPU-friendly models. |
| Improve evaluation integrity | Continue blind label adjudication, publish the estimated gold-label ceiling with accuracy, and use the contested split to refine taxonomy or measure abstention. |
| Evaluate configurable anchors | Measure configurable anchor sets, their stability, and their accuracy by label domain; custom anchors are an extension of current support, not a new runtime family. |
| Resolve unexplained backend results | Investigate the vLLM Semantic Router timeout-like benchmark arm before treating it as capacity evidence. |
| MLflow and AI Innovation coordination | Track datasets, artifacts, model revisions, evaluation runs, and proposed refinement or fine-tuning work with the partner teams. |

Performance experiments such as batching concurrent misses, quantization,
shape discipline, executor-width tuning, and fusing the trained head belong in
this workstream. They are worthwhile when profiling and benchmark evidence
identifies them as the next constraint; they are not ahead of correctness,
operability, or core model selection.

**Success criteria:** comparable runs are reproducible, model selection is
evidence-led, and every promoted artifact has a traceable evaluation record.

---

## Workstream 3 — Always-classify zero/one-shot domains

Build the primary product path around small supported models and configurable
anchors. The label set is unordered and classification is attempted for every
eligible request. The service must make uncertainty explicit rather than
silently converting it to a routing decision.

| Work | Outcome | Existing issue |
| --- | --- | --- |
| Make anchor configuration a first-class contract | Document and harden the existing ability to update labelled anchors; define configuration, startup embedding, versioning, and reproducibility expectations. | known gaps |
| Domain routing with user-supplied anchors | Allow a domain taxonomy to be supplied without a rebuild or mandatory training; rank requests against labelled anchors. | known gaps |
| Explicit `ABSTAIN` / low-confidence behavior | Return an honest result when context or taxonomy separation is insufficient, with observable downstream fallback. | [#8](https://github.com/llm-d-incubation/llm-d-semantic-classifier/issues/8) |
| Candidate-conditioned scoring contract | Add `f(request, candidate)` only where the use case needs model-affinity or capability-fit scores; preserve request-intrinsic classification as its own contract. | [#18](https://github.com/llm-d-incubation/llm-d-semantic-classifier/issues/18) |
| Fine-tuning as an upgrade path | Explore AI Innovation embedding SFT or equivalent only for taxonomies that measured anchor separation cannot support. | future refinement |

Evaluation of multiple classifier signals and their weighting may be composed at
the Praxis/Gateway layer where appropriate. `llm-d-sc` should expose clear,
independently evaluable scores rather than take ownership of policy and final
routing selection.

**Success criteria:** a user can configure an unordered domain label set with
anchors, receive a classification for each eligible request, and see an
explicit, observable uncertainty result when it cannot be resolved.

---

## Workstream 4 — Multi-turn, conditional classification

This workstream is jointly defined with the Gateway. It models ordered labels
or states over time and invokes classification only when needed—not as a
per-turn replacement for zero/one-shot classification.

| Work | Outcome |
| --- | --- |
| Session and turn contract | Define Gateway ownership of session identity, turn count, classification state, and score provenance. |
| Reclassification triggers | Specify material-change triggers, including saturation or flow-control changes and conversation signals relevant to routing. |
| Stable routing behavior | Define hysteresis, update signaling, and fallback behavior so a session does not flap between routes. |
| Gateway integration | Keep infrastructure-aware routing, capacity, KV-cache pressure, policy, and final routing decisions in the Gateway. |
| Decision-cache strategy | Explore cross-replica Gateway/classifier cache sharing only after the ownership and correctness contract is clear. |

Session-scoped classification is therefore not a standalone classifier
throughput optimization. Fewer forwards may be a beneficial consequence, but
the principal deliverable is a correct Gateway–classifier contract for
conditional reclassification.

**Success criteria:** the Gateway can demonstrate that it reclassifies only on
defined triggers, preserves stable routing across a conversation, and records
why each reclassification occurred.

---

## Future work — Additional model architectures and backends

Complex model families are intentionally deferred. They should enter through a
stable adapter/backend boundary and only with a workload-specific evaluation
plan—not merely because they are available.

| Exploration | Entry condition |
| --- | --- |
| Sequence-classification runtime adapter | A selected model and benchmark show a need that the supported embedding/anchor path cannot meet. [#7](https://github.com/llm-d-incubation/llm-d-semantic-classifier/issues/7) |
| vLLM backend | A/B evidence shows a material advantage for an intended workload over the embedded CPU-friendly path. |
| Vela, decision models, and Jev-like architectures | A concrete classification use case, model artifact, integration contract, and evaluation dataset exist. |
| Other decision-model families | A small initial set has demonstrated value; each additional family brings an evaluation row and maintenance owner. |

---

## Ongoing project hygiene

| Work | Existing issue |
| --- | --- |
| Migrate default classifier artifacts from a personal namespace to the `llm-d` organization | [#6](https://github.com/llm-d-incubation/llm-d-semantic-classifier/issues/6) |
| Make `fetch-model` verify present files against the pinned revision | [#5](https://github.com/llm-d-incubation/llm-d-semantic-classifier/issues/5) |
| Reconcile the cost-classifier definition with `fetch-model` and pin the mutable reference | [#1](https://github.com/llm-d-incubation/llm-d-semantic-classifier/issues/1) |
| Refresh `known-gaps.md` as evidence and decisions change | [#15](https://github.com/llm-d-incubation/llm-d-semantic-classifier/issues/15) |

//! The L0 text prefilter: an approximate cache consulted BEFORE the forward.
//!
//! The L2 semantic tier sits between `embed` and `rank`, so its key is the model
//! forward's OUTPUT. A hit there skips only `rank` -- microseconds -- while paying
//! a Redis KNN round trip on top of a forward that already cost 30.72 ms p50.
//! Nothing about hit rate or load makes that profitable.
//!
//! L0 is keyed on TEXT, which costs microseconds to compute, so a hit skips the
//! whole 30.72 ms. That inverts the economics: here even an in-process lookup is
//! worth roughly 30,000x its own cost.
//!
//! IT IS APPROXIMATE, AND THAT IS A DIFFERENT KIND OF CACHE.
//! L1 (exact) and L2 (embedding-KNN) both key on something derived from the
//! request itself. L0 serves a result computed for a DIFFERENT, similar prompt.
//! It therefore has an ERROR RATE, not merely a hit rate, and three things follow:
//!
//!   1. it is OFF by default;
//!   2. a band collision is never sufficient -- the candidate's full signature is
//!      verified against a Jaccard threshold before anything is served;
//!   3. entries are isolated by cache identity exactly as L1/L2 are, so a model
//!      or taxonomy change cannot serve a stale label.
//!
//! Recall and precision are properties to MEASURE against the full model, not to
//! assume from the hit rate. A high hit rate with a loose threshold is the failure
//! mode, not the success mode.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::classify::ClassificationResult;
use crate::prefilter::{bands, Signature};

/// Number of LSH bands.
///
/// BAND FOR RECALL, VERIFY FOR PRECISION. Retrieval only has to propose the right
/// neighbour; the Jaccard check decides whether it may be served. Tuning the bands
/// for precision as well just loses true positives that verification would have
/// kept anyway.
///
/// The band scheme's 50% retrieval point is at `(1/b)^(1/r)`. The first version
/// used 16 bands x 8 rows, putting that point at `(1/16)^(1/8)` ~ 0.707 -- and
/// the test case "...liability risk" -> "...liability risks" is J ~ 0.714, landing
/// exactly on the coin flip, which is how it failed intermittently.
///
/// 32 bands x 4 rows moves the point to `(1/32)^(1/4)` ~ 0.42:
///   J = 0.71  ->  1 - (1 - 0.71^4)^32  ~ 1.000 recall
///   J = 0.10  ->  1 - (1 - 0.10^4)^32  ~ 0.003 spurious candidates
/// and a spurious candidate costs one Jaccard comparison, not a wrong answer.
const BANDS: usize = 32;

/// The L0 seam.
pub trait TextCache: Send + Sync {
    /// A stored result whose signature is within threshold of `sig` and shares
    /// `identity`, else `None`.
    fn lookup(&self, sig: &Signature, identity: &str) -> Option<ClassificationResult>;

    /// Record `result` under `sig` and `identity`. Best-effort.
    fn insert(&self, sig: &Signature, result: &ClassificationResult, identity: &str);
}

/// The default: always misses, never stores. Zero cost when L0 is off.
pub struct NoopTextCache;

impl TextCache for NoopTextCache {
    fn lookup(&self, _sig: &Signature, _identity: &str) -> Option<ClassificationResult> {
        None
    }
    fn insert(&self, _sig: &Signature, _result: &ClassificationResult, _identity: &str) {}
}

struct Entry {
    sig: Signature,
    identity: String,
    result: ClassificationResult,
}

/// A bounded, in-process, band-indexed approximate cache.
///
/// IN-PROCESS ON PURPOSE. A network round trip is affordable here -- it would
/// still beat 30.72 ms -- but it is not NECESSARY, and the L2 experience is that
/// a round trip on the request path is the thing that turned a plausible idea
/// into measured overhead. Start where the cost is unambiguous.
pub struct MemoryTextCache {
    /// band key -> entry ids
    index: Mutex<HashMap<u64, Vec<usize>>>,
    /// id -> entry; ids are positions in a bounded ring
    entries: Mutex<Vec<Entry>>,
    capacity: usize,
    threshold: f64,
    /// Next slot to overwrite once full. Without an explicit cursor,
    /// `entries.len() % capacity` is 0 forever after the ring fills, so slot 0
    /// would be the only one ever evicted and the cache would hold capacity-1
    /// permanently stale entries.
    cursor: Mutex<usize>,
}

impl MemoryTextCache {
    /// `threshold` is the minimum verified Jaccard required to SERVE a result.
    #[must_use]
    pub fn new(capacity: usize, threshold: f64) -> Self {
        Self {
            index: Mutex::new(HashMap::new()),
            entries: Mutex::new(Vec::new()),
            capacity: capacity.max(1),
            threshold: threshold.clamp(0.0, 1.0),
            cursor: Mutex::new(0),
        }
    }
}

impl TextCache for MemoryTextCache {
    fn lookup(&self, sig: &Signature, identity: &str) -> Option<ClassificationResult> {
        let keys = bands(sig, BANDS);
        let index = self.index.lock().ok()?;
        let entries = self.entries.lock().ok()?;
        let mut best: Option<(f64, &Entry)> = None;
        // Candidates are whatever shares a band. Verification, not retrieval, is
        // what decides: banding is tuned for recall and will propose neighbours
        // that are not close enough to serve.
        for k in keys {
            for id in index.get(&k).map(Vec::as_slice).unwrap_or(&[]) {
                let Some(e) = entries.get(*id) else { continue };
                if e.identity != identity {
                    continue; // a different model/taxonomy: never serve across it
                }
                let j = sig.jaccard(&e.sig);
                if best.as_ref().is_none_or(|(bj, _)| j > *bj) {
                    best = Some((j, e));
                }
            }
        }
        match best {
            Some((j, e)) if j >= self.threshold => Some(e.result.clone()),
            _ => None,
        }
    }

    fn insert(&self, sig: &Signature, result: &ClassificationResult, identity: &str) {
        let Ok(mut entries) = self.entries.lock() else {
            return;
        };
        let Ok(mut index) = self.index.lock() else {
            return;
        };
        let id = if entries.len() < self.capacity {
            entries.push(Entry {
                sig: sig.clone(),
                identity: identity.to_string(),
                result: result.clone(),
            });
            entries.len() - 1
        } else {
            // Ring eviction. The evicted entry's band keys are removed first;
            // leaving them would point at a slot now holding someone else's
            // result, and the Jaccard verify would then run against the WRONG
            // signature -- a stale-serve bug that a hit-rate metric would report
            // as success.
            let Ok(mut cursor) = self.cursor.lock() else {
                return;
            };
            let id = *cursor % self.capacity;
            *cursor = (*cursor + 1) % self.capacity;
            let old_keys = bands(&entries[id].sig, BANDS);
            for k in old_keys {
                if let Some(v) = index.get_mut(&k) {
                    v.retain(|x| *x != id);
                }
            }
            entries[id] = Entry {
                sig: sig.clone(),
                identity: identity.to_string(),
                result: result.clone(),
            };
            id
        };
        for k in bands(sig, BANDS) {
            index.entry(k).or_default().push(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::{ClassifyStatus, RankedSignal};
    use crate::prefilter::signature;

    fn result(label: &str) -> ClassificationResult {
        ClassificationResult {
            classifier_id: "c".into(),
            model_revision: "m".into(),
            tokenizer_revision: "t".into(),
            taxonomy_revision: "x".into(),
            status: ClassifyStatus::Ok,
            ranked: vec![RankedSignal {
                id: label.into(),
                score: 1.0,
            }],
        }
    }

    /// The point of L0: the same prompt, lightly edited, must hit -- that is the
    /// 30.72 ms the tier exists to avoid.
    ///
    /// The threshold here is 0.6 for a measured reason. "...liability risk" ->
    /// "...liability risks" is only **J = 0.68**: an 8-token prompt has 6
    /// shingles, and changing the last word alters the 2 windows containing it,
    /// so 4/6 survive. Short prompts lose Jaccard FAST under small edits -- see
    /// u107.
    #[test]
    fn u100_near_duplicate_hits() {
        let c = MemoryTextCache::new(64, 0.6);
        let a = signature("summarise this contract and flag any liability risk");
        c.insert(&a, &result("COMPLEX"), "id-1");
        let b = signature("summarise this contract and flag any liability risks");
        let hit = c.lookup(&b, "id-1").expect("near-duplicate must hit");
        assert_eq!(hit.ranked[0].id, "COMPLEX");
    }

    /// THE HAZARD THIS TIER CARRIES, pinned so it can never be forgotten.
    ///
    /// Text similarity is not meaning similarity, and the margin between them is
    /// thin. MEASURED on this cache's own signatures:
    ///
    /// | edit | J |
    /// |---|---|
    /// | benign: "...liability risk" -> "...liability risks" | **0.680** |
    /// | INVERTING: "is it safe to..." -> "is it unsafe to..." | **0.578** |
    ///
    /// The inverting edit scores LOWER, but only by ~0.10, and that gap is an
    /// artifact of these two sentences' lengths rather than a property of meaning.
    /// No global threshold separates "paraphrase" from "opposite request"
    /// reliably: set it above 0.68 and ordinary paraphrase stops hitting; set it
    /// below 0.578 and an inverted request is served the original's answer.
    ///
    /// L1 (exact) and L2 (embedding) do not have this problem -- L1 keys on the
    /// exact text, and an embedding moves when meaning moves. L0 is trading that
    /// guarantee for the 30.72 ms forward, and the trade must be made knowingly.
    ///
    /// This is why L0 is OFF by default and why its error rate must be measured
    /// against the full model before anyone enables it. The assertion is on the
    /// SIMILARITY, not on a refusal: the cache has no way to tell these apart,
    /// and a test pretending it could would be the actual bug.
    #[test]
    fn u106_meaning_inverting_edits_stay_textually_similar() {
        let inverting = signature("is it safe to run this migration on the production database")
            .jaccard(&signature(
                "is it unsafe to run this migration on the production database",
            ));
        let benign = signature("summarise this contract and flag any liability risk").jaccard(
            &signature("summarise this contract and flag any liability risks"),
        );
        assert!(
            inverting > 0.5,
            "an inverted request measured J={inverting:.3} -- still similar enough \
             to be retrieved and served"
        );
        assert!(
            (benign - inverting).abs() < 0.2,
            "benign {benign:.3} and inverting {inverting:.3} are only \
             {:.3} apart; no global threshold separates them",
            (benign - inverting).abs()
        );
    }

    /// Short prompts lose Jaccard fast; long ones barely move. The threshold is
    /// therefore length-sensitive, and a single global value is a compromise.
    #[test]
    fn u107_short_prompts_lose_similarity_faster_than_long_ones() {
        let short_j = signature("summarise this contract and flag any liability risk").jaccard(
            &signature("summarise this contract and flag any liability risks"),
        );
        let long = "summarise this contract, flag any liability risk, list the \
                    termination clauses, and call out anything unusual about the \
                    indemnity provisions or the governing law section";
        let long_j = signature(long).jaccard(&signature(&format!("{long} please")));
        assert!(
            long_j > short_j,
            "long {long_j:.3} must degrade less than short {short_j:.3}"
        );
    }

    /// Unrelated text must MISS. An approximate cache that serves anything is
    /// worse than no cache -- it trades a measured error rate for an unmeasured one.
    #[test]
    fn u101_unrelated_text_misses() {
        let c = MemoryTextCache::new(64, 0.7);
        c.insert(
            &signature("summarise this contract and flag liability"),
            &result("COMPLEX"),
            "id-1",
        );
        assert!(c
            .lookup(&signature("write a haiku about the sea"), "id-1")
            .is_none());
    }

    /// Identity isolation: a model or taxonomy change must never serve the
    /// previous model's label, exactly as for L1 and L2.
    #[test]
    fn u102_identity_isolates_entries() {
        let c = MemoryTextCache::new(64, 0.7);
        let s = signature("explain the CAP theorem with an example");
        c.insert(&s, &result("COMPLEX"), "model-A");
        assert!(c.lookup(&s, "model-A").is_some());
        assert!(
            c.lookup(&s, "model-B").is_none(),
            "a different identity must never hit"
        );
    }

    /// A band collision alone must not serve. Retrieval is tuned for recall and
    /// WILL propose neighbours that are not close enough.
    #[test]
    fn u103_threshold_gates_serving_not_retrieval() {
        let loose = MemoryTextCache::new(64, 0.5);
        let strict = MemoryTextCache::new(64, 0.999);
        let a = signature("rewrite this paragraph to be more concise and direct");
        let b = signature("rewrite this paragraph to be more concise and direct please");
        for c in [&loose, &strict] {
            c.insert(&a, &result("MEDIUM"), "id-1");
        }
        assert!(loose.lookup(&b, "id-1").is_some(), "loose threshold serves");
        assert!(
            strict.lookup(&b, "id-1").is_none(),
            "a near-duplicate below threshold must NOT be served"
        );
    }

    /// Eviction must not leave band keys pointing at a reused slot. A stale key
    /// would make the Jaccard verify run against the WRONG signature -- a
    /// stale-serve bug that a hit-rate metric reports as success.
    #[test]
    fn u104_eviction_does_not_leave_stale_band_keys() {
        let c = MemoryTextCache::new(2, 0.9);
        let first = signature("the first prompt about contracts and liability terms");
        c.insert(&first, &result("FIRST"), "id-1");
        // Overwrite the ring several times with unrelated text.
        for i in 0..6 {
            c.insert(
                &signature(&format!(
                    "completely different prompt number {i} about weather"
                )),
                &result("OTHER"),
                "id-1",
            );
        }
        // The first entry is long gone; it must not resurface.
        match c.lookup(&first, "id-1") {
            None => {}
            Some(hit) => assert_ne!(
                hit.ranked[0].id, "FIRST",
                "an evicted entry was served from a stale band key"
            ),
        }
    }

    /// The default tier must cost nothing and never serve.
    #[test]
    fn u105_noop_never_serves() {
        let c = NoopTextCache;
        let s = signature("anything at all");
        c.insert(&s, &result("X"), "id-1");
        assert!(c.lookup(&s, "id-1").is_none());
    }
}

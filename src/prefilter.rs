//! Cheap text keys, so an approximate cache can be consulted BEFORE the forward.
//!
//! The L2 semantic tier cannot pay for itself as placed. Its lookup key is the
//! model forward's OUTPUT (`classify.rs`: embed, then `semantic.lookup`), so a hit
//! skips only `rank` -- microseconds -- while adding a Redis KNN round trip
//! (~1 ms) on top of a forward that already cost 30.72 ms p50. There is no hit
//! rate or load level at which that is profitable; under load the round trip gets
//! worse, not better.
//!
//! vSR's semantic cache wins because it caches the LLM RESPONSE, where a hit skips
//! an entire generation and the embedding is cheap by comparison. The technique
//! does not transfer to a classifier, because here the expensive step is the one
//! that produces the key.
//!
//! This module supplies keys computable from TEXT ALONE, in microseconds, so a
//! cache can be asked before the 30.72 ms is spent. SimHash over token shingles:
//! near-duplicate prompts land at small Hamming distance, and the distance is a
//! bounded, testable proxy for similarity rather than an assumed one.
//!
//! IT IS A PREFILTER, NOT A DECIDER. It proposes candidates; whether a candidate
//! may be served without running the model is a separate policy decision with its
//! own recall measurement. Serving on a SimHash hit alone would trade a measured
//! error rate for an unmeasured one -- exactly the substitution this project has
//! been burned by before.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Shingle width, in tokens.
///
/// 3 is the usual near-duplicate compromise: 1 ignores word order entirely (an
/// anagram of a prompt would collide), and 5+ makes a single edited word drop too
/// many shingles from the set.
const SHINGLE: usize = 3;

/// Number of MinHash permutations in a signature.
///
/// The Jaccard estimate's standard error is 1/sqrt(K), so 128 gives ~0.088 --
/// tight enough to separate "edited one word" (J ~ 0.88) from "different prompt"
/// (J ~ 0.0) with room to spare.
pub const K: usize = 128;

/// WHY MINHASH AND NOT SIMHASH.
///
/// The first version of this module used SimHash over the same shingles and was
/// replaced on measurement, not taste. SimHash sums a +/-1 vote per shingle into
/// each of 64 bits, so its stability depends on having MANY features. A 9-token
/// prompt yields ~7 shingles, each carrying ~1/7 of every bit's vote, and
/// appending a single word ("...direct" -> "...direct please") moved the hash by
/// **15 of 64 bits**. Banded retrieval needs two hashes within `bands - 1` flips
/// to share a band, so covering distance 15 would need 16 bands of 4 bits, which
/// collide with everything.
///
/// MinHash estimates Jaccard over the shingle SET directly. The same edit is
/// J ~ 0.88, and banded MinHash LSH has a tunable recall curve rather than a
/// pigeonhole bound that does not hold at these distances.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature(pub Vec<u64>);

impl Signature {
    /// Estimated Jaccard similarity: the fraction of permutations that agree.
    #[must_use]
    pub fn jaccard(&self, other: &Signature) -> f64 {
        if self.0.is_empty() || self.0.len() != other.0.len() {
            return 0.0;
        }
        let same = self
            .0
            .iter()
            .zip(other.0.iter())
            .filter(|(a, b)| a == b)
            .count();
        same as f64 / self.0.len() as f64
    }
}

/// Normalize for hashing: lowercase, collapse whitespace, drop nothing else.
///
/// Deliberately conservative. Stripping punctuation would collapse "delete the
/// user?" and "delete the user!" onto one key, and for a classifier reading intent
/// those are not obviously the same request.
fn normalize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| w.to_lowercase())
        .filter(|w| !w.is_empty())
        .collect()
}

fn hash_one<T: Hash>(t: &T) -> u64 {
    let mut h = DefaultHasher::new();
    t.hash(&mut h);
    h.finish()
}

/// The shingle set for `text`.
///
/// Text shorter than one shingle becomes a single whole-text shingle rather than
/// an empty set: short prompts are ordinary traffic, and an empty set would make
/// the prefilter silently skip exactly the inputs it should handle best.
fn shingles(text: &str) -> Vec<u64> {
    let tokens = normalize(text);
    if tokens.is_empty() {
        return Vec::new();
    }
    if tokens.len() < SHINGLE {
        return vec![hash_one(&tokens.join(" "))];
    }
    let mut v: Vec<u64> = tokens
        .windows(SHINGLE)
        .map(|w| hash_one(&w.join(" ")))
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// MinHash signature over token shingles.
///
/// Permutations are simulated with a cheap multiply-xor mix per index, which is
/// standard practice and avoids carrying 128 random seeds as state that would
/// then have to be pinned in the ModelCar for signatures to stay comparable
/// across restarts.
#[must_use]
pub fn signature(text: &str) -> Signature {
    let sh = shingles(text);
    if sh.is_empty() {
        return Signature(vec![0; K]);
    }
    let mut sig = vec![u64::MAX; K];
    for h in sh {
        for (i, slot) in sig.iter_mut().enumerate() {
            // A distinct, deterministic permutation per index.
            let mixed = (h ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                .wrapping_mul(0xBF58_476D_1CE4_E5B9);
            let mixed = mixed ^ (mixed >> 31);
            if mixed < *slot {
                *slot = mixed;
            }
        }
    }
    Signature(sig)
}

/// Banded LSH keys for candidate retrieval.
///
/// Scanning every stored signature defeats the purpose -- that is a linear scan in
/// front of the 30.72 ms forward we are trying to avoid. Hashing each band of
/// `K / bands` rows into one key means two texts are retrieved if they agree on
/// ALL rows of ANY band, giving the standard 1 - (1 - J^r)^b recall curve: tunable
/// and analyzable, unlike a bound that does not hold.
#[must_use]
pub fn bands(sig: &Signature, bands: usize) -> Vec<u64> {
    let bands = bands.clamp(1, sig.0.len().max(1));
    let rows = sig.0.len() / bands;
    if rows == 0 {
        return Vec::new();
    }
    (0..bands)
        .map(|b| {
            let slice = &sig.0[b * rows..(b + 1) * rows];
            // The band index is mixed in so band 0's value cannot collide with
            // band 1's identical value.
            hash_one(&(b as u64, slice))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The premise: near-duplicates must score far higher than unrelated text.
    #[test]
    fn u090_near_duplicates_score_higher_than_unrelated() {
        let a = signature("summarise this contract and flag any liability risk");
        let b = signature("Summarise this contract and flag any liability risks");
        let c = signature("write a haiku about the sea in autumn");
        assert!(
            a.jaccard(&b) > a.jaccard(&c),
            "near-dup {:.3} must beat unrelated {:.3}",
            a.jaccard(&b),
            a.jaccard(&c)
        );
        assert!(a.jaccard(&c) < 0.2, "unrelated text must not look similar");
    }

    /// Identical text must be J=1, or the case the prefilter is most certain
    /// about would miss. Normalization makes casing and spacing irrelevant.
    #[test]
    fn u091_identical_text_is_jaccard_one() {
        let t = "explain the CAP theorem with an example";
        assert!((signature(t).jaccard(&signature(t)) - 1.0).abs() < 1e-9);
        assert!(
            (signature(t).jaccard(&signature("  Explain   the CAP Theorem with an EXAMPLE "))
                - 1.0)
                .abs()
                < 1e-9
        );
    }

    /// The edit that broke SimHash. Appending one word moved SimHash 15 of 64
    /// bits; MinHash must still read it as highly similar.
    #[test]
    fn u092_appending_one_word_stays_similar() {
        let a = signature("rewrite this paragraph to be more concise and direct");
        let b = signature("rewrite this paragraph to be more concise and direct please");
        let j = a.jaccard(&b);
        assert!(j > 0.7, "one appended word dropped similarity to {j:.3}");
    }

    /// Short prompts are ordinary traffic and must produce a usable signature.
    #[test]
    fn u093_short_text_still_produces_a_signature() {
        assert!((signature("hi").jaccard(&signature("hi")) - 1.0).abs() < 1e-9);
        assert!(signature("hi").jaccard(&signature("bye")) < 0.5);
        assert_eq!(
            signature("").0,
            vec![0; K],
            "empty text is the only empty signature"
        );
    }

    /// Word ORDER must matter: a classifier reading intent cannot treat
    /// "ignore the safety policy" and its reversal as one prompt.
    #[test]
    fn u094_word_order_changes_the_signature() {
        let a = signature("delete the production database now please");
        let b = signature("please now database production the delete");
        assert!(a.jaccard(&b) < 0.5, "shingles must encode order");
    }

    /// Banding must actually retrieve the near-duplicate. This is the test
    /// SimHash failed at distance 15.
    #[test]
    fn u095_bands_retrieve_a_near_duplicate() {
        let a = signature("rewrite this paragraph to be more concise and direct");
        let b = signature("rewrite this paragraph to be more concise and direct please");
        let (ba, bb) = (bands(&a, 16), bands(&b, 16));
        assert!(
            ba.iter().any(|x| bb.contains(x)),
            "near-duplicate at J={:.3} shared no band",
            a.jaccard(&b)
        );
    }

    /// Unrelated text must NOT collide, or every lookup pays a verification
    /// forward and the prefilter costs more than it saves.
    #[test]
    fn u096_unrelated_text_does_not_share_a_band() {
        let a = bands(&signature("summarise this contract and flag liability"), 16);
        let b = bands(&signature("write a haiku about the sea in autumn"), 16);
        assert!(!a.iter().any(|x| b.contains(x)), "unrelated text collided");
    }

    /// Band keys must be distinct across positions.
    #[test]
    fn u097_band_keys_are_position_qualified() {
        let b = bands(&signature("some ordinary prompt text here"), 8);
        assert_eq!(b.len(), 8);
        assert_eq!(b.iter().collect::<std::collections::HashSet<_>>().len(), 8);
    }

    /// The prefilter must be orders of magnitude cheaper than the 30.72 ms
    /// forward it exists to avoid, so a rewrite cannot quietly make it expensive.
    #[test]
    fn u098_signing_is_far_cheaper_than_a_forward() {
        let text = "summarise this contract and flag any liability risk ".repeat(20);
        let start = std::time::Instant::now();
        for _ in 0..200 {
            let _ = signature(&text);
        }
        let per = start.elapsed() / 200;
        assert!(
            per < std::time::Duration::from_millis(1),
            "signature took {per:?}; the forward it avoids is ~30 ms"
        );
    }
}

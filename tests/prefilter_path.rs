//! L0 must skip the FORWARD, not merely return faster.
//!
//! The whole claim for this tier is that its hit avoids the model forward --
//! 30.72 ms p50, 99.4% of request latency. A timing assertion would prove
//! nothing (a warm L2 is also fast), so these count the runtime's own forward
//! calls. If a future change moves the L0 lookup back below `embed`, the tier
//! silently becomes L2 with extra steps and only this test notices.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use llm_d_sc::cache::text::MemoryTextCache;
use llm_d_sc::classify::{
    ClassificationInput, ClassificationResult, ClassifierRuntime, ClassifyError, ClassifyStatus,
    Embedding, RankedSignal, RuntimeMetadata, ServiceCore,
};
use llm_d_sc::metrics::Metrics;

struct CountingRuntime {
    forwards: Arc<AtomicUsize>,
}

impl ClassifierRuntime for CountingRuntime {
    fn metadata(&self) -> RuntimeMetadata {
        RuntimeMetadata {
            classifier_id: "test".into(),
            signal: "complexity".into(),
            model_revision: "rev-1".into(),
            tokenizer_revision: "tok-1".into(),
            taxonomy_revision: "tax-1".into(),
            artifact_digest: None,
        }
    }

    fn embed(&self, _input: &ClassificationInput) -> Result<Embedding, ClassifyError> {
        // This stands in for the 30.72 ms forward.
        self.forwards.fetch_add(1, Ordering::SeqCst);
        Ok(Embedding::new(vec![0.1, 0.2, 0.3]))
    }

    fn rank(
        &self,
        _embedding: &Embedding,
        _input: &ClassificationInput,
    ) -> Result<ClassificationResult, ClassifyError> {
        Ok(ClassificationResult {
            classifier_id: "test".into(),
            model_revision: "rev-1".into(),
            tokenizer_revision: "tok-1".into(),
            taxonomy_revision: "tax-1".into(),
            status: ClassifyStatus::Ok,
            ranked: vec![RankedSignal {
                id: "COMPLEX".into(),
                score: 0.9,
            }],
        })
    }
}

fn input(text: &str) -> ClassificationInput {
    ClassificationInput {
        text: text.to_string(),
        requested_signals: vec![],
        session_metadata: Default::default(),
        context_completeness: Default::default(),
    }
}

#[test]
fn i100_l0_hit_skips_the_model_forward() {
    let forwards = Arc::new(AtomicUsize::new(0));
    let core = ServiceCore::with_text_prefilter(
        CountingRuntime {
            forwards: forwards.clone(),
        },
        Metrics::new(),
        Arc::new(llm_d_sc::cache::NoopSemanticCache),
        Arc::new(MemoryTextCache::new(64, 0.6)),
    );

    let long = "summarise this contract, flag any liability risk, and list the \
                termination clauses with their notice periods";
    core.classify(input(long)).expect("first call");
    assert_eq!(
        forwards.load(Ordering::SeqCst),
        1,
        "first call must forward"
    );

    // A DIFFERENT but near-duplicate prompt: L1 exact cannot help here, so any
    // saving is L0's.
    let edited = format!("{long} please");
    let r = core.classify(input(&edited)).expect("second call");
    assert_eq!(
        forwards.load(Ordering::SeqCst),
        1,
        "an L0 hit must NOT run the forward again"
    );
    assert_eq!(r.ranked[0].id, "COMPLEX");
}

#[test]
fn i101_l0_miss_still_forwards() {
    let forwards = Arc::new(AtomicUsize::new(0));
    let core = ServiceCore::with_text_prefilter(
        CountingRuntime {
            forwards: forwards.clone(),
        },
        Metrics::new(),
        Arc::new(llm_d_sc::cache::NoopSemanticCache),
        Arc::new(MemoryTextCache::new(64, 0.6)),
    );
    core.classify(input("summarise this contract and flag liability risk"))
        .expect("first");
    core.classify(input("write a haiku about the sea in autumn"))
        .expect("second");
    assert_eq!(
        forwards.load(Ordering::SeqCst),
        2,
        "unrelated text must not be served from L0"
    );
}

#[test]
fn i102_default_core_has_no_prefilter() {
    let forwards = Arc::new(AtomicUsize::new(0));
    let core = ServiceCore::new(CountingRuntime {
        forwards: forwards.clone(),
    });
    let long = "summarise this contract, flag any liability risk, and list the \
                termination clauses with their notice periods";
    core.classify(input(long)).expect("first");
    core.classify(input(&format!("{long} please")))
        .expect("second");
    assert_eq!(
        forwards.load(Ordering::SeqCst),
        2,
        "L0 must be OFF by default: an approximate tier is opt-in"
    );
}

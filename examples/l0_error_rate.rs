//! L0's ERROR RATE against the full model, over real traffic.
//!
//! L0 serves a result computed for a DIFFERENT prompt, so a hit rate says nothing
//! about whether it is safe. The number that decides is: of the requests L0 would
//! have served, on how many does its answer differ from what the model would have
//! said? Measured against the full model on every row, never inferred.
//!
//! Swept across thresholds, because the threshold IS the safety knob and a single
//! value would hide the shape of the trade.
use std::collections::HashMap;

use llm_d_sc::cache::text::{MemoryTextCache, TextCache};
use llm_d_sc::classify::{ClassificationInput, ClassifierRuntime};
use llm_d_sc::prefilter::signature;

fn ask(
    clf: &llm_d_sc::classify::CandleClassifier,
    p: &str,
) -> llm_d_sc::classify::ClassificationResult {
    clf.classify(ClassificationInput {
        text: p.to_string(),
        requested_signals: vec![],
        session_metadata: Default::default(),
        context_completeness: Default::default(),
    })
    .expect("classify")
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or("/work/models-head".into());
    let corpus = std::env::args()
        .nth(2)
        .unwrap_or("/work/corpus.jsonl".into());
    let n: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4000);

    let clf = llm_d_sc::classify::load_and_warm_modelcar(&dir).expect("ModelCar must load");
    let text = std::fs::read_to_string(&corpus).expect("corpus");
    let prompts: Vec<String> = text
        .lines()
        .filter_map(|l| {
            serde_json::from_str::<serde_json::Value>(l)
                .ok()
                .and_then(|v| {
                    v.get("text")
                        .or_else(|| v.get("prompt"))
                        .and_then(|t| t.as_str().map(str::to_string))
                })
        })
        .take(n)
        .collect();
    eprintln!("prompts: {}", prompts.len());

    let truth: Vec<String> = prompts
        .iter()
        .map(|p| ask(&clf, p).ranked[0].id.clone())
        .collect();

    println!("threshold\thits\thit_rate_pct\tagree\tERROR_RATE_pct");
    for t in [0.50_f64, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99] {
        let cache = MemoryTextCache::new(50_000, t);
        let (mut hits, mut agree) = (0usize, 0usize);
        let mut by_label: HashMap<String, (usize, usize)> = HashMap::new();
        for (p, want) in prompts.iter().zip(truth.iter()) {
            let sig = signature(p);
            if let Some(served) = cache.lookup(&sig, "id") {
                hits += 1;
                let got = &served.ranked[0].id;
                if got == want {
                    agree += 1;
                }
                let e = by_label.entry(want.clone()).or_insert((0, 0));
                e.0 += 1;
                if got != want {
                    e.1 += 1;
                }
            } else {
                // A miss is what populates the cache -- exactly as in the serving
                // path, where L0 is written from the computed result.
                cache.insert(&sig, &ask(&clf, p), "id");
            }
        }
        let hr = hits as f64 / prompts.len().max(1) as f64 * 100.0;
        let err = if hits > 0 {
            (1.0 - agree as f64 / hits as f64) * 100.0
        } else {
            0.0
        };
        println!("{t:.2}\t{hits}\t{hr:.2}\t{agree}\t{err:.3}");
        let mut worst: Vec<_> = by_label.into_iter().filter(|(_, v)| v.1 > 0).collect();
        worst.sort_by_key(|(_, v)| std::cmp::Reverse(v.1));
        for (label, (seen, wrong)) in worst.into_iter().take(3) {
            eprintln!("   t={t:.2} label {label}: {wrong}/{seen} served wrong");
        }
    }
}

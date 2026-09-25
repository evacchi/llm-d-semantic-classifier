//! Locate the Rust/PyTorch divergence on a ModernBERT head by instrumenting the
//! stages, rather than guessing at causes one fix at a time.
//!
//! Prints the token IDs and the FULL probability vector. Token IDs isolate the
//! tokenizer from the model; the full vector distinguishes a temperature-like
//! difference (all mass shifted) from a structural one (different ordering).
fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or("/work/models-vela-head".into());
    let tok =
        llm_d_sc::tokenizer::Tokenizer::load(std::path::Path::new(&dir).join("tokenizer.json"))
            .expect("tokenizer");
    let emb = llm_d_sc::embedding::Embedder::load(
        std::path::Path::new(&dir).join("config.json"),
        std::path::Path::new(&dir).join("model.safetensors"),
        std::path::Path::new(&dir).join("tokenizer.json"),
        std::path::Path::new(&dir).join("1_Pooling/config.json"),
    )
    .expect("embedder");
    let labels: Vec<String> = emb
        .labels()
        .map(|l| l.to_vec())
        .unwrap_or_else(|| vec!["<no-head>".into()]);
    eprintln!("labels({}): {:?}", labels.len(), labels);

    for text in ["hi", "what is 2+2"] {
        let ids = tok.tokenize(text).expect("tokenize");
        println!("IDS\t{text}\t{ids:?}");
        // BISECT: the encoder's CLS row, before any head touches it. If this
        // already differs from PyTorch, the gap is candle's ModernBert forward
        // and belongs upstream; if it matches, the gap is ours.
        let (n, h) = emb.hidden_states(ids.clone()).expect("hidden");
        let hid = h.len() / n;
        let cls = &h[0..hid];
        let norm: f32 = cls.iter().map(|x| x * x).sum::<f32>().sqrt();
        let head8: Vec<String> = cls.iter().take(8).map(|x| format!("{x:.5}")).collect();
        println!("CLS\t{text}\tnorm={norm:.5}\t[{}]", head8.join(", "));
        let (_v, logits) = emb.embed_and_classify(ids).expect("forward");
        match logits {
            None => println!("LOGITS\t{text}\tNONE (head did not load)"),
            Some(l) => {
                let p = llm_d_sc::head::softmax(&l);
                let mut pairs: Vec<(String, f32, f32)> = labels
                    .iter()
                    .cloned()
                    .zip(l.iter().copied())
                    .zip(p.iter().copied())
                    .map(|((a, b), c)| (a, b, c))
                    .collect();
                pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
                let top: Vec<String> = pairs
                    .iter()
                    .take(4)
                    .map(|(n, lg, pr)| format!("{n}={pr:.6}(logit {lg:.4})"))
                    .collect();
                println!("RUSTP\t{text}\t{}", top.join("  "));
            }
        }
    }
}

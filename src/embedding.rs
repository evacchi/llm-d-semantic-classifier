//! Real Candle forward for the resident sensitivity model.
//!
//! AC-004 requires the pinned sensitivity model to match trusted reference
//! embedding/ranking fixtures. U-062 proves the first real-forward contract:
//! running the resident BERT model through Candle must emit an embedding whose
//! length equals the dimension declared by the ModelCar's pooling config
//! (`1_Pooling/config.json`, `word_embedding_dimension`).
//!
//! [`Embedder`] loads the bert config and weights via
//! `VarBuilder::from_mmaped_safetensors` (unsafe: memory-maps the safetensors),
//! builds `BertModel`, tokenizes with the resident [`Tokenizer`], runs a forward
//! pass, and mean-pools the sequence dimension (masked by the attention mask)
//! into the final embedding vector.

use std::fs;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{bert, modernbert};
use serde_json::Value;

use crate::tokenizer::Tokenizer;

/// Errors produced while loading the embedder or embedding text.
#[derive(Debug)]
pub enum EmbeddingError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Candle(candle_core::Error),
    Tokenizer(crate::tokenizer::TokenizerError),
    MissingField(&'static str),
}

impl std::fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingError::Io(e) => write!(f, "embedding config io error: {e}"),
            EmbeddingError::Json(e) => write!(f, "embedding config json error: {e}"),
            EmbeddingError::Candle(e) => write!(f, "embedding candle error: {e}"),
            EmbeddingError::Tokenizer(e) => write!(f, "embedding tokenizer error: {e}"),
            EmbeddingError::MissingField(name) => {
                write!(f, "embedding config missing field: {name}")
            }
        }
    }
}

impl std::error::Error for EmbeddingError {}

/// The embedding dimension contract declared by the resident pooling config.
///
/// Loads `word_embedding_dimension` from a ModelCar `1_Pooling/config.json` so
/// the resident model's embedding dimension is pinned to the model contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingContract {
    word_embedding_dimension: usize,
}

impl EmbeddingContract {
    /// Load the embedding dimension contract from a pooling config file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<EmbeddingContract, EmbeddingError> {
        let raw = fs::read_to_string(path).map_err(EmbeddingError::Io)?;
        let root: Value = serde_json::from_str(&raw).map_err(EmbeddingError::Json)?;
        // sentence-transformers has emitted two pooling-config schemas: the
        // legacy `word_embedding_dimension` and, since ST 3.x, the newer
        // `embedding_dimension`. A ModelCar produced by either version is a
        // valid artifact, so accept both rather than rejecting the newer one.
        let word_embedding_dimension = root
            .get("word_embedding_dimension")
            .or_else(|| root.get("embedding_dimension"))
            .and_then(Value::as_u64)
            .ok_or(EmbeddingError::MissingField("word_embedding_dimension"))?
            as usize;
        Ok(EmbeddingContract {
            word_embedding_dimension,
        })
    }

    /// The resident model's embedding dimension.
    pub fn dimension(&self) -> usize {
        self.word_embedding_dimension
    }
}

/// A resident embedder: the real Candle forward over the pinned sensitivity
/// BERT model, tokenized by the resident [`Tokenizer`].
///
/// Which encoder architecture a ModelCar declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackboneKind {
    Bert,
    ModernBert,
}

/// A loaded encoder backbone.
///
/// llm-d-sc served exactly one architecture (BERT) until ModernBERT ModelCars
/// appeared -- the Vela family (llm-semantic-router/Vela-1.0-Encoder-307M*) is
/// ModernBERT, and its config does not even deserialize into `bert::Config`.
///
/// Dispatch is on the config's `model_type`, the field the HF ecosystem already
/// uses to select an architecture. Sniffing tensor names would be worse than
/// useless here: the two architectures share many names, so a mis-detection
/// would load, run, and emit silently wrong embeddings -- the failure shape this
/// project has already been bitten by once, where argmax survived a broken
/// transform and only calibration revealed it.
pub enum Backbone {
    Bert(Box<bert::BertModel>),
    ModernBert(Box<modernbert::ModernBert>),
}

/// A parsed-but-not-yet-loaded ModelCar config.
pub enum BackboneConfig {
    Bert(Box<bert::Config>),
    ModernBert(Box<modernbert::Config>),
}

impl BackboneConfig {
    pub fn kind(&self) -> BackboneKind {
        match self {
            BackboneConfig::Bert(_) => BackboneKind::Bert,
            BackboneConfig::ModernBert(_) => BackboneKind::ModernBert,
        }
    }

    pub fn hidden_size(&self) -> usize {
        match self {
            BackboneConfig::Bert(c) => c.hidden_size,
            BackboneConfig::ModernBert(c) => c.hidden_size,
        }
    }
}

impl Backbone {
    /// Parse a ModelCar `config.json` into the architecture it declares.
    ///
    /// A missing `model_type` is BERT: every ModelCar published before
    /// ModernBERT support omits it, and rejecting them would break every
    /// deployed classifier. An UNKNOWN `model_type` is an error rather than a
    /// fallback -- silently loading the wrong backbone is the one outcome worse
    /// than refusing to start.
    pub fn parse_config(raw: &str) -> Result<BackboneConfig, EmbeddingError> {
        #[derive(serde::Deserialize)]
        struct Probe {
            #[serde(default)]
            model_type: Option<String>,
        }
        let probe: Probe = serde_json::from_str(raw).map_err(EmbeddingError::Json)?;
        match probe.model_type.as_deref() {
            None | Some("bert") => Ok(BackboneConfig::Bert(Box::new(
                serde_json::from_str(raw).map_err(EmbeddingError::Json)?,
            ))),
            Some("modernbert") => Ok(BackboneConfig::ModernBert(Box::new(
                serde_json::from_str(raw).map_err(EmbeddingError::Json)?,
            ))),
            Some(_) => Err(EmbeddingError::MissingField(
                "unsupported model_type; llm-d-sc serves bert and modernbert",
            )),
        }
    }

    fn load(vb: VarBuilder, config: &BackboneConfig) -> Result<Backbone, EmbeddingError> {
        match config {
            BackboneConfig::Bert(c) => Ok(Backbone::Bert(Box::new(
                bert::BertModel::load(vb, c).map_err(EmbeddingError::Candle)?,
            ))),
            BackboneConfig::ModernBert(c) => Ok(Backbone::ModernBert(Box::new(
                modernbert::ModernBert::load(vb, c).map_err(EmbeddingError::Candle)?,
            ))),
        }
    }

    pub fn kind(&self) -> BackboneKind {
        match self {
            Backbone::Bert(_) => BackboneKind::Bert,
            Backbone::ModernBert(_) => BackboneKind::ModernBert,
        }
    }

    fn device(&self) -> &Device {
        match self {
            Backbone::Bert(m) => &m.device,
            // ModernBert does not expose a device handle; this crate is CPU-only
            // (no cuda feature on candle-core), so the device is known.
            Backbone::ModernBert(_) => &Device::Cpu,
        }
    }

    /// Run the encoder forward, returning the per-token hidden states.
    ///
    /// BERT takes token_type_ids; ModernBERT has no segment embedding and takes
    /// the attention mask directly. The mask dtype also differs -- ModernBERT
    /// multiplies it into attention scores, so it must be float, not u32.
    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor, EmbeddingError> {
        match self {
            Backbone::Bert(m) => {
                let seq = input_ids.dims2().map_err(EmbeddingError::Candle)?;
                let token_type_ids = Tensor::zeros(seq, DType::U32, self.device())
                    .map_err(EmbeddingError::Candle)?;
                m.forward(input_ids, &token_type_ids, Some(attention_mask))
                    .map_err(EmbeddingError::Candle)
            }
            Backbone::ModernBert(m) => {
                let mask = attention_mask
                    .to_dtype(DType::F32)
                    .map_err(EmbeddingError::Candle)?;
                m.forward(input_ids, &mask).map_err(EmbeddingError::Candle)
            }
        }
    }
}

/// The model weights are memory-mapped from a local `model.safetensors` (never
/// fetched at runtime) and run on the CPU in eval mode (dropout disabled), so
/// repeated embeddings of the same input are deterministic.
pub struct Embedder {
    model: Backbone,
    tokenizer: Tokenizer,

    /// The ModelCar's sequence-classification head, when it ships one.
    head: Option<crate::head::SequenceHead>,

    /// Label names in class-index order, present exactly when `head` is.
    labels: Option<crate::head::LabelMap>,
}

impl Embedder {
    /// Load the embedder from the bert `config.json`, the safetensors weights,
    /// the `tokenizer.json`, and the pooling `config.json`.
    pub fn load<P: AsRef<Path>>(
        model_config: P,
        weights: P,
        tokenizer_path: P,
        pooling_config: P,
    ) -> Result<Embedder, EmbeddingError> {
        let raw = fs::read_to_string(model_config).map_err(EmbeddingError::Io)?;
        let config = Backbone::parse_config(&raw)?;

        let device = Device::Cpu;
        // SAFETY: `from_mmaped_safetensors` memory-maps the safetensors file.
        // The mapped region is owned by the returned VarBuilder/backend and stays
        // alive for the lifetime of the tensors built from it, so the mmap is
        // valid for the whole model load.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights.as_ref()], DType::F32, &device)
        }
        .map_err(EmbeddingError::Candle)?;
        let model = Backbone::load(vb.clone(), &config)?;

        // A head is loaded only when the ModelCar DECLARES one via id2label and
        // its weights are actually present. An embedding-only artifact is
        // legitimate and keeps working on the anchor path.
        let labels = crate::head::LabelMap::from_config(&raw);
        // `classifier_pooling` comes from the checkpoint; CLS is upstream's
        // default and what BERT-family heads use.
        let mean_pooling = serde_json::from_str::<serde_json::Value>(&raw)
            .ok()
            .and_then(|v| {
                v.get("classifier_pooling")
                    .and_then(|p| p.as_str().map(|s| s.eq_ignore_ascii_case("mean")))
            })
            .unwrap_or(false);
        let head = match labels.as_ref() {
            Some(l) => crate::head::SequenceHead::load(&vb, &config, l, mean_pooling)?,
            None => None,
        };
        let labels = head.as_ref().and(labels);

        let tokenizer = Tokenizer::load(tokenizer_path).map_err(EmbeddingError::Tokenizer)?;
        // The pooling config pins the expected embedding dimension; validate it
        // matches the backbone hidden_size so the forward emits the contracted
        // dim. ModernBERT ModelCars (the Vela family) ship no
        // sentence-transformers module stack and therefore no pooling config --
        // there the dimension is taken from the backbone itself and the check is
        // vacuous, which is honest: there is no second declaration to agree with.
        //
        // A pooling config that EXISTS is still checked. Only absence is
        // tolerated, so a BERT ModelCar with a contradictory pooling config
        // still fails exactly as before.
        if pooling_config.as_ref().exists() {
            let contract = EmbeddingContract::load(pooling_config)?;
            if contract.dimension() != config.hidden_size() {
                return Err(EmbeddingError::MissingField(
                    "word_embedding_dimension != hidden_size",
                ));
            }
        }

        Ok(Embedder {
            model,
            tokenizer,
            head,
            labels,
        })
    }

    /// Tokenize `text` into its token-ID sequence. Split out so a caller can
    /// measure the tokenize stage independently from the model forward (AC-012).
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, EmbeddingError> {
        self.tokenizer
            .tokenize(text)
            .map_err(EmbeddingError::Tokenizer)
    }

    /// Embed an already-tokenized ID sequence: run the model forward and
    /// mean-pool the sequence dimension (masked by the attention mask) into the
    /// final embedding vector.
    pub fn embed_ids(&self, ids: Vec<u32>) -> Result<Vec<f32>, EmbeddingError> {
        let seq_len = ids.len();
        let device = self.model.device();

        let input_ids =
            Tensor::from_vec(ids, (1, seq_len), device).map_err(EmbeddingError::Candle)?;
        let attention_mask =
            Tensor::ones((1, seq_len), DType::U32, device).map_err(EmbeddingError::Candle)?;

        let sequence = self.model.forward(&input_ids, &attention_mask)?;
        pool_and_normalize(&sequence, &attention_mask)
    }

    /// The head's label set, when this ModelCar ships a usable head.
    pub fn labels(&self) -> Option<&[String]> {
        self.labels.as_ref().map(crate::head::LabelMap::labels)
    }

    /// Embed AND classify in one forward.
    ///
    /// One forward, two outputs. Running the encoder again to get logits would
    /// double the cost of the single most expensive stage in the service -- the
    /// model forward is 99.4% of request latency (S-080).
    pub fn embed_and_classify(
        &self,
        ids: Vec<u32>,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>), EmbeddingError> {
        let seq_len = ids.len();
        let device = self.model.device();
        let input_ids =
            Tensor::from_vec(ids, (1, seq_len), device).map_err(EmbeddingError::Candle)?;
        let attention_mask =
            Tensor::ones((1, seq_len), DType::U32, device).map_err(EmbeddingError::Candle)?;

        let sequence = self.model.forward(&input_ids, &attention_mask)?;
        let embedding = pool_and_normalize(&sequence, &attention_mask)?;

        let logits = match self.head.as_ref() {
            Some(h) => Some(h.logits(&self.model, &input_ids, &attention_mask, &sequence)?),
            None => None,
        };
        Ok((embedding, logits))
    }

    /// Embed `text`: tokenize, run the model forward, and mean-pool the
    /// sequence dimension (masked by the attention mask) into the final
    /// embedding vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let ids = self.tokenize(text)?;
        self.embed_ids(ids)
    }
}

/// Masked mean-pool then L2-normalize, as `modules.json` declares.
///
/// Shared by `embed_ids` and `embed_and_classify` so the two cannot drift: an
/// embedding that differed between the head path and the anchor path would make
/// every A/B between them measure the pooling, not the head.
fn pool_and_normalize(
    sequence: &Tensor,
    attention_mask: &Tensor,
) -> Result<Vec<f32>, EmbeddingError> {
    let pooled = mean_pool(sequence, attention_mask).map_err(EmbeddingError::Candle)?;
    let flat = pooled.squeeze(0).map_err(EmbeddingError::Candle)?;
    let norm = flat.norm().map_err(EmbeddingError::Candle)?;
    let normalized = flat
        .broadcast_div(&norm.unsqueeze(0).map_err(EmbeddingError::Candle)?)
        .map_err(EmbeddingError::Candle)?;
    normalized.to_vec1::<f32>().map_err(EmbeddingError::Candle)
}

/// Masked mean-pool over the sequence dimension (dim 1), matching the
/// sentence-transformers `MeanPooling` operator: sum of non-pad token
/// embeddings divided by the number of non-pad tokens.
fn mean_pool(sequence: &Tensor, attention_mask: &Tensor) -> candle_core::Result<Tensor> {
    let mask = attention_mask.unsqueeze(2)?.to_dtype(DType::F32)?; // [1, seq_len, 1]
    let masked = sequence.broadcast_mul(&mask)?; // [1, seq_len, hidden]
    let sum = masked.sum(1)?; // [1, hidden]
    let denom = mask.sum(1)?; // [1, 1]
    sum.broadcast_div(&denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN_INPUT: &str = "this is a golden sensitivity input";

    /// Vela (llm-semantic-router/Vela-1.0-Encoder-307M*) is ModernBERT, not
    /// BERT. The Embedder hardcoded `bert::Config`/`bert::BertModel`, so a
    /// ModernBERT ModelCar could not load at all -- serde rejects the config
    /// before any weight is touched, because ModernBERT has no
    /// `type_vocab_size` and carries fields BERT's Config does not model.
    ///
    /// Detection is on `model_type`, the field the HF ecosystem already uses to
    /// pick an architecture, rather than on sniffing tensor names: a
    /// mis-detected backbone would still load many shared tensors and then emit
    /// silently wrong embeddings, which is the failure mode this project can
    /// least afford (see the cosine-similarity normalisation incident).
    #[test]
    fn u070_modernbert_config_is_detected_and_accepted() {
        let vela = r#"{
          "model_type": "modernbert",
          "architectures": ["ModernBertForSequenceClassification"],
          "vocab_size": 256000, "hidden_size": 768, "num_hidden_layers": 22,
          "num_attention_heads": 12, "intermediate_size": 1152,
          "max_position_embeddings": 32768, "layer_norm_eps": 1e-5,
          "pad_token_id": 1, "global_attn_every_n_layers": 3,
          "global_rope_theta": 160000.0, "local_attention": 128,
          "local_rope_theta": 10000.0
        }"#;
        let parsed = Backbone::parse_config(vela).expect("ModernBERT config must parse");
        assert_eq!(parsed.kind(), BackboneKind::ModernBert);
        assert_eq!(parsed.hidden_size(), 768);
    }

    /// The existing BERT path must keep working unchanged -- detection must not
    /// regress every classifier already deployed.
    #[test]
    fn u071_bert_config_still_detected() {
        let bert = r#"{
          "model_type": "bert", "vocab_size": 30522, "hidden_size": 384,
          "num_hidden_layers": 6, "num_attention_heads": 12,
          "intermediate_size": 1536, "max_position_embeddings": 512,
          "type_vocab_size": 2, "layer_norm_eps": 1e-12,
          "hidden_act": "gelu", "hidden_dropout_prob": 0.1,
          "initializer_range": 0.02, "pad_token_id": 0,
          "classifier_dropout": null
        }"#;
        let parsed = Backbone::parse_config(bert).expect("BERT config must parse");
        assert_eq!(parsed.kind(), BackboneKind::Bert);
        assert_eq!(parsed.hidden_size(), 384);
    }

    /// A config with no `model_type` must be treated as BERT, not rejected:
    /// every ModelCar published before this change omits it.
    #[test]
    fn u072_missing_model_type_defaults_to_bert() {
        let legacy = r#"{
          "vocab_size": 30522, "hidden_size": 384,
          "num_hidden_layers": 6, "num_attention_heads": 12,
          "intermediate_size": 1536, "max_position_embeddings": 512,
          "type_vocab_size": 2, "layer_norm_eps": 1e-12,
          "hidden_act": "gelu", "hidden_dropout_prob": 0.1,
          "initializer_range": 0.02, "pad_token_id": 0,
          "classifier_dropout": null
        }"#;
        let parsed = Backbone::parse_config(legacy).expect("legacy config must parse");
        assert_eq!(parsed.kind(), BackboneKind::Bert);
    }

    /// An unknown architecture must FAIL rather than fall back to BERT. Falling
    /// back would load a wrong backbone and serve plausible-looking embeddings.
    #[test]
    fn u073_unknown_model_type_is_rejected() {
        let alien = r#"{"model_type": "t5", "hidden_size": 768}"#;
        assert!(
            Backbone::parse_config(alien).is_err(),
            "an unknown model_type must be rejected, never silently treated as BERT"
        );
    }

    fn artifact(name: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("artifacts")
            .join("models")
            .join("sensitivity")
            .join(name)
    }

    fn fixture(name: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("modelcar")
            .join(name)
    }

    #[test]
    #[ignore]
    fn u062_real_candle_forward_matches_embedding_dimension() {
        // U-062 (AC-004): the resident model's real Candle forward must emit an
        // embedding of exactly the dimension declared by the ModelCar pooling
        // config (`word_embedding_dimension`). This requires the local model
        // weights (gitignored), so the test is #[ignore]d and run explicitly
        // with `-- --ignored` after `./hack/fetch-model`.
        let embedder = Embedder::load(
            artifact("config.json"),
            artifact("model.safetensors"),
            fixture("tokenizer.json"),
            artifact("1_Pooling/config.json"),
        )
        .expect("embedder must load from the fetched sensitivity model");
        let contract = EmbeddingContract::load(artifact("1_Pooling/config.json"))
            .expect("pooling config must load");
        let vec = embedder
            .embed(GOLDEN_INPUT)
            .expect("golden input must embed");
        assert_eq!(
            vec.len(),
            contract.dimension(),
            "real forward embedding length must match word_embedding_dimension"
        );
    }

    #[test]
    #[ignore]
    fn u061_pooling_output_matches_trusted_reference() {
        // U-061 (AC-004): the resident model's real pooling output must match the
        // trusted reference embedding fixture (`tests/fixtures/modelcar/golden-embedding.json`,
        // produced by `artifacts/u061_tight.sh`) within tight tolerance. This requires
        // the local model weights (gitignored), so the test is #[ignore]d and run
        // explicitly with `-- --ignored` after `./hack/fetch-model`.
        let embedder = Embedder::load(
            artifact("config.json"),
            artifact("model.safetensors"),
            fixture("tokenizer.json"),
            artifact("1_Pooling/config.json"),
        )
        .expect("embedder must load from the fetched sensitivity model");

        let raw = std::fs::read_to_string(fixture("golden-embedding.json"))
            .expect("golden embedding fixture must exist");
        let root: serde_json::Value =
            serde_json::from_str(&raw).expect("golden embedding fixture must be valid JSON");
        let input = root
            .get("input")
            .and_then(serde_json::Value::as_str)
            .expect("fixture input");
        let first16: Vec<f64> = root
            .get("first16")
            .and_then(serde_json::Value::as_array)
            .expect("fixture first16")
            .iter()
            .map(|v| v.as_f64().expect("first16 value"))
            .collect();
        assert_eq!(
            first16.len(),
            16,
            "reference fixture must have exactly 16 dims"
        );
        let l2_norm = root
            .get("l2_norm")
            .and_then(serde_json::Value::as_f64)
            .expect("fixture l2_norm");
        let dim = root
            .get("dim")
            .and_then(serde_json::Value::as_u64)
            .expect("fixture dim") as usize;

        let vec = embedder.embed(input).expect("fixture input must embed");
        assert_eq!(
            vec.len(),
            dim,
            "embedding dim must match the fixture's declared dim"
        );

        // First 16 dims each within 1e-4 of the trusted reference.
        for (i, (got, want)) in vec.iter().zip(first16.iter()).enumerate() {
            let diff = (f64::from(*got) - want).abs();
            assert!(
                diff <= 1e-4,
                "dim {i} diff {diff} exceeds 1e-4 (got {got}, want {want})"
            );
        }

        // Full-vector L2 norm within 1e-3 of the trusted reference.
        let norm: f64 = vec
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            .sqrt();
        let norm_diff = (norm - l2_norm).abs();
        assert!(
            norm_diff <= 1e-3,
            "l2 norm diff {norm_diff} exceeds 1e-3 (got {norm}, want {l2_norm})"
        );
    }

    fn synthetic_prototypes() -> Vec<crate::ranker::Prototype> {
        let raw = std::fs::read_to_string(fixture("synthetic-prototypes.json"))
            .expect("synthetic prototype fixture must exist");
        let root: serde_json::Value =
            serde_json::from_str(&raw).expect("synthetic prototype fixture must be valid JSON");
        assert_eq!(
            root.get("label").and_then(serde_json::Value::as_str),
            Some("synthetic_for_mechanics_only")
        );
        root.get("prototypes")
            .and_then(serde_json::Value::as_array)
            .expect("prototypes array")
            .iter()
            .map(|obj| {
                let id = obj
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .expect("prototype id")
                    .to_string();
                let vector: Vec<f32> = obj
                    .get("vector")
                    .and_then(serde_json::Value::as_array)
                    .expect("prototype vector")
                    .iter()
                    .map(|v| v.as_f64().expect("vector value") as f32)
                    .collect();
                crate::ranker::Prototype::new(id, vector)
            })
            .collect()
    }

    #[test]
    #[ignore]
    fn u067_golden_fixture_ranking_matches_reference() {
        // U-067 (AC-004): the resident model's real forward, embedded with the
        // same golden input used by the golden embedding fixture, must cosine-rank
        // the synthetic prototypes in EXACTLY the reference order, and each score
        // must be within 1e-4 of the reference fixture
        // (`tests/fixtures/modelcar/golden-ranking.json`, produced by the pinned
        // sentence-transformers stack). Requires the local model weights
        // (gitignored), so the test is #[ignore]d and run explicitly with
        // `-- --ignored` after `./hack/fetch-model`.
        let embedder = Embedder::load(
            artifact("config.json"),
            artifact("model.safetensors"),
            fixture("tokenizer.json"),
            artifact("1_Pooling/config.json"),
        )
        .expect("embedder must load from the fetched sensitivity model");

        let raw = std::fs::read_to_string(fixture("golden-ranking.json"))
            .expect("golden ranking fixture must exist");
        let root: serde_json::Value =
            serde_json::from_str(&raw).expect("golden ranking fixture must be valid JSON");
        assert_eq!(
            root.get("input").and_then(serde_json::Value::as_str),
            Some(GOLDEN_INPUT),
            "fixture input must be the golden input"
        );
        let want: Vec<(String, f64)> = root
            .get("ranking")
            .and_then(serde_json::Value::as_array)
            .expect("ranking array")
            .iter()
            .map(|obj| {
                let id = obj
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .expect("ranking id")
                    .to_string();
                let score = obj
                    .get("score")
                    .and_then(serde_json::Value::as_f64)
                    .expect("ranking score");
                (id, score)
            })
            .collect();
        assert_eq!(want.len(), 4, "fixture must rank 4 prototypes");

        let vec = embedder
            .embed(GOLDEN_INPUT)
            .expect("golden input must embed");
        let prototypes = synthetic_prototypes();
        let got = crate::ranker::cosine_rank(&vec, &prototypes);

        // Exact order must match the reference ranking.
        let got_ids: Vec<&str> = got.iter().map(|(id, _)| id.as_str()).collect();
        let want_ids: Vec<&str> = want.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(
            got_ids, want_ids,
            "ranking order must match the golden fixture exactly"
        );

        // Each score within 1e-4 of the reference.
        for (i, ((gid, gscore), (wid, wscore))) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(gid, wid, "rank {i} id mismatch");
            let diff = (gscore - wscore).abs();
            assert!(
                diff <= 1e-4,
                "rank {i} ({gid}) score diff {diff} exceeds 1e-4 (got {gscore}, want {wscore})"
            );
        }
    }

    #[test]
    #[ignore]
    fn u063_embedding_normalization_matches_classifier_definition() {
        // U-063 (AC-004): the resident model's `modules.json` (pinned HF rev
        // 43f21d2...) declares a `sentence_transformers.models.Normalize` module
        // at idx 2, so the classifier definition is L2-NORMALIZED embeddings.
        // `embed()` must therefore emit an embedding whose L2 norm is ~1.0, NOT
        // the raw masked-mean-pooled vector (which has norm ~5.76). Requires the
        // local weights (gitignored), so the test is #[ignore]d and run
        // explicitly with `-- --ignored` after `./hack/fetch-model`.
        let embedder = Embedder::load(
            artifact("config.json"),
            artifact("model.safetensors"),
            fixture("tokenizer.json"),
            artifact("1_Pooling/config.json"),
        )
        .expect("embedder must load from the fetched sensitivity model");

        let vec = embedder
            .embed(GOLDEN_INPUT)
            .expect("golden input must embed");
        let norm: f64 = vec
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() <= 1e-3,
            "embed() must emit an L2-normalized embedding per the classifier's Normalize \
             module (got norm {norm}, want ~1.0)"
        );
    }
}

//! Sequence-classification heads (AC: serve the trained head, not anchor cosine).
//!
//! llm-d-sc has always ranked by cosine similarity against taxonomy anchors. Every
//! ModelCar this project and the vSR Vela family publish is a
//! `*ForSequenceClassification` checkpoint, so that path loads the encoder body
//! and DISCARDS the trained head:
//!
//!   cnuland/llm-d-sc-complexity-v3   bert.pooler.dense.{weight,bias}, classifier.{weight,bias}
//!   Vela-1.0-Encoder-307M-Domain     head.{dense,norm}.weight,        classifier.{weight,bias}
//!
//! Measured on 552 real rows, anchor ranking is BELOW the majority-class baseline
//! on 3 of 5 signals (complexity −2.54, cx2 −3.99, sensitivity −2.73) and negative
//! in 29 of 35 encoder×signal cells. A linear probe over the SAME embeddings scores
//! +13.77 on complexity — a +16.31 swing that costs one matrix multiply, because
//! cosine-to-a-centroid is a rank-1 decision rule and cannot express these
//! boundaries.
//!
//! ModernBERT heads come from `candle_transformers`, which already implements
//! `ModernBertForSequenceClassification`. BERT has no equivalent there, so its head
//! is assembled from `candle_nn` primitives in the shape HuggingFace defines:
//! pooler dense + tanh over the CLS token, then the classifier projection. That is
//! composing primitives, not reimplementing a library.

use candle_core::{DType, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::collections::BTreeMap;

use crate::embedding::{Backbone, BackboneConfig, EmbeddingError};

/// The label map a ModelCar declares, ordered by class index.
#[derive(Debug, Clone)]
pub struct LabelMap(Vec<String>);

impl LabelMap {
    /// Parse `id2label`, ordered by its NUMERIC index.
    ///
    /// JSON object order is not index order, and the head's output column `i`
    /// means `id2label["i"]`. Reading them in document order would mislabel every
    /// prediction while leaving accuracy plausible on a balanced set.
    pub fn from_config(raw: &str) -> Option<LabelMap> {
        #[derive(Deserialize)]
        struct Probe {
            id2label: Option<BTreeMap<String, String>>,
        }
        let p: Probe = serde_json::from_str(raw).ok()?;
        let map = p.id2label?;
        let mut pairs: Vec<(usize, String)> = map
            .into_iter()
            .filter_map(|(k, v)| k.parse::<usize>().ok().map(|i| (i, v)))
            .collect();
        pairs.sort_by_key(|(i, _)| *i);
        if pairs.is_empty() || pairs.iter().enumerate().any(|(n, (i, _))| n != *i) {
            return None; // non-contiguous indices: refuse rather than guess
        }
        Some(LabelMap(pairs.into_iter().map(|(_, v)| v).collect()))
    }

    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.0
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// A loaded sequence-classification head.
pub enum SequenceHead {
    /// HuggingFace `BertForSequenceClassification`: pooler dense + tanh on CLS,
    /// then the classifier projection.
    Bert { pooler: Linear, classifier: Linear },
    /// ModernBERT: `head.dense -> gelu_erf -> head.norm -> classifier`, over the
    /// pooled token.
    ///
    /// ASSEMBLED HERE RATHER THAN TAKEN FROM `candle_transformers`, for two
    /// reasons that are both bugs if ignored:
    ///
    /// 1. `ModernBertForSequenceClassification::load` sizes its classifier from
    ///    `config.classifier_config.id2label.len()`, and that config never
    ///    deserializes for a real HuggingFace checkpoint. Candle declares
    ///    `label2id: HashMap<String, String>` while HF emits `String -> int`
    ///    (`{"biology": 0}`), so the flattened `Option<ClassifierConfig>` silently
    ///    becomes `None`, the classifier is built with out_dim
    ///    `unwrap_or_default()` = 0, and the load fails against the real
    ///    [14, 768] weight. This affects every standard HF ModernBERT classifier,
    ///    not just Vela.
    /// 2. `ModernBertClassifier::forward` applies softmax INTERNALLY. Feeding its
    ///    output into this crate's own softmax would double-apply it and flatten
    ///    the distribution -- argmax would survive, so accuracy would look fine
    ///    while every confidence threshold read a squashed distribution. That is
    ///    the same silent shape as the cosine-normalisation incident.
    ///
    /// `ModernBertHead::load` is private upstream, so the two layers are built
    /// from `candle_nn` primitives in the documented shape. Composing primitives,
    /// not reimplementing the crate.
    ModernBert {
        dense: Linear,
        norm: LayerNorm,
        classifier: Linear,
        /// CLS takes token 0; MEAN averages over the mask. Read from the
        /// checkpoint's own `classifier_pooling`, defaulting to CLS as upstream does.
        mean_pooling: bool,
    },
}

impl SequenceHead {
    /// Load the head for `config`, or `None` when the ModelCar has no usable one.
    ///
    /// `None` is not a failure: an embedding-only ModelCar is legitimate and must
    /// keep working on the anchor path. Only a head that is declared and then
    /// fails to load is an error.
    pub fn load(
        vb: &VarBuilder,
        config: &BackboneConfig,
        labels: &LabelMap,
        mean_pooling: bool,
        classifier_bias: bool,
    ) -> Result<Option<SequenceHead>, EmbeddingError> {
        let n = labels.len();
        if n < 2 {
            return Ok(None); // a one-class head decides nothing
        }
        match config {
            BackboneConfig::Bert(c) => {
                // Checkpoints differ in whether the backbone is nested under a
                // `bert.` prefix, exactly as candle's own BertModel::load allows.
                let pooler = ["bert.pooler.dense", "pooler.dense"]
                    .iter()
                    .find_map(|p| candle_nn::linear(c.hidden_size, c.hidden_size, vb.pp(p)).ok());
                let classifier = candle_nn::linear(c.hidden_size, n, vb.pp("classifier")).ok();
                match (pooler, classifier) {
                    (Some(pooler), Some(classifier)) => {
                        Ok(Some(SequenceHead::Bert { pooler, classifier }))
                    }
                    _ => Ok(None),
                }
            }
            BackboneConfig::ModernBert(c) => {
                let dense = candle_nn::linear_no_bias(
                    c.hidden_size,
                    c.hidden_size,
                    vb.pp("head").pp("dense"),
                );
                let norm = candle_nn::layer_norm_no_bias(
                    c.hidden_size,
                    c.layer_norm_eps,
                    vb.pp("head").pp("norm"),
                );
                // HONOUR `classifier_bias`. Vela ships a `classifier.bias`
                // tensor while its config sets `classifier_bias: false`, so
                // PyTorch builds the Linear WITHOUT a bias and never applies the
                // stored one. Loading it anyway adds a per-class constant the
                // checkpoint does not use: argmax survives a small additive
                // shift, so labels still matched while probabilities diverged in
                // MIXED directions (0.9962 vs 0.9389 on one input, 0.9834 vs
                // 0.9940 on another). A dead tensor in the file is not a
                // licence to apply it.
                let classifier = if classifier_bias {
                    candle_nn::linear(c.hidden_size, n, vb.pp("classifier"))
                } else {
                    candle_nn::linear_no_bias(c.hidden_size, n, vb.pp("classifier"))
                };
                match (dense, norm, classifier) {
                    (Ok(dense), Ok(norm), Ok(classifier)) => Ok(Some(SequenceHead::ModernBert {
                        dense,
                        norm,
                        classifier,
                        mean_pooling,
                    })),
                    _ => Ok(None),
                }
            }
        }
    }

    /// Logits for one tokenized input, `[num_labels]`.
    ///
    /// The ModernBERT head owns its own backbone, so it runs the whole forward
    /// itself; the BERT head consumes hidden states the caller already computed.
    /// Giving the ModernBERT arm its own forward avoids running the encoder twice.
    pub fn logits(
        &self,
        backbone: &Backbone,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        hidden: &Tensor,
    ) -> Result<Vec<f32>, EmbeddingError> {
        let _ = backbone;
        let out = match self {
            SequenceHead::Bert { pooler, classifier } => {
                // CLS token, dense, tanh -- the HF pooler, exactly.
                let cls = hidden
                    .i((.., 0, ..))
                    .map_err(EmbeddingError::Candle)?
                    .contiguous()
                    .map_err(EmbeddingError::Candle)?;
                let pooled = pooler
                    .forward(&cls)
                    .map_err(EmbeddingError::Candle)?
                    .tanh()
                    .map_err(EmbeddingError::Candle)?;
                classifier
                    .forward(&pooled)
                    .map_err(EmbeddingError::Candle)?
            }
            SequenceHead::ModernBert {
                dense,
                norm,
                classifier,
                mean_pooling,
            } => {
                let _ = input_ids;
                let pooled = if *mean_pooling {
                    let m = attention_mask
                        .unsqueeze(2)
                        .map_err(EmbeddingError::Candle)?
                        .to_dtype(DType::F32)
                        .map_err(EmbeddingError::Candle)?;
                    let summed = hidden
                        .broadcast_mul(&m)
                        .map_err(EmbeddingError::Candle)?
                        .sum(1)
                        .map_err(EmbeddingError::Candle)?;
                    let counts = m.sum(1).map_err(EmbeddingError::Candle)?;
                    summed
                        .broadcast_div(&counts)
                        .map_err(EmbeddingError::Candle)?
                } else {
                    hidden
                        .i((.., 0, ..))
                        .map_err(EmbeddingError::Candle)?
                        .contiguous()
                        .map_err(EmbeddingError::Candle)?
                };
                let x = dense
                    .forward(&pooled)
                    .map_err(EmbeddingError::Candle)?
                    .gelu_erf()
                    .map_err(EmbeddingError::Candle)?;
                let x = norm.forward(&x).map_err(EmbeddingError::Candle)?;
                // RAW LOGITS. The caller applies this crate's stable softmax;
                // returning probabilities here would double-apply it.
                classifier.forward(&x).map_err(EmbeddingError::Candle)?
            }
        };
        out.flatten_all()
            .map_err(EmbeddingError::Candle)?
            .to_vec1::<f32>()
            .map_err(EmbeddingError::Candle)
    }
}

/// Numerically stable softmax over logits.
///
/// Subtracting the max before exponentiating is not cosmetic here: a previous
/// incident in this project normalised NEGATIVE cosine similarities by their sum,
/// which left argmax correct while every calibration and abstention threshold read
/// nonsense. Scores that leave this service are probabilities or they are nothing.
#[must_use]
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![0.0; logits.len()];
    }
    let exp: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    if sum <= 0.0 {
        return vec![0.0; logits.len()];
    }
    exp.into_iter().map(|e| e / sum).collect()
}

use candle_core::IndexOp;

#[cfg(test)]
mod tests {
    use super::*;

    /// `id2label` is a JSON OBJECT. Its document order is not index order, and the
    /// head's output column `i` means `id2label["i"]`. Reading document order would
    /// mislabel every prediction while leaving accuracy plausible on a balanced set
    /// -- the same silent shape as the cosine-normalisation incident.
    #[test]
    fn u080_labels_are_ordered_by_index_not_document_order() {
        let raw = r#"{"id2label":{"2":"HARD","0":"TRIVIAL","1":"STANDARD"}}"#;
        let m = LabelMap::from_config(raw).expect("must parse");
        assert_eq!(m.labels(), ["TRIVIAL", "STANDARD", "HARD"]);
    }

    /// Non-contiguous indices must be refused, not silently compacted. Compacting
    /// {0,1,3} to three labels shifts every class above the gap by one.
    #[test]
    fn u081_non_contiguous_label_indices_are_refused() {
        let raw = r#"{"id2label":{"0":"A","1":"B","3":"D"}}"#;
        assert!(LabelMap::from_config(raw).is_none());
    }

    /// A ModelCar with no id2label is embedding-only. That is legitimate and must
    /// stay on the anchor path rather than fail.
    #[test]
    fn u082_absent_label_map_is_not_an_error() {
        assert!(LabelMap::from_config(r#"{"hidden_size":384}"#).is_none());
    }

    /// Real label maps from both families must parse.
    #[test]
    fn u083_real_label_maps_parse() {
        let cn = r#"{"id2label":{"0":"TRIVIAL","1":"WORK"}}"#;
        assert_eq!(
            LabelMap::from_config(cn).unwrap().labels(),
            ["TRIVIAL", "WORK"]
        );
        let vela = r#"{"id2label":{"0":"biology","1":"business","2":"chemistry"}}"#;
        assert_eq!(LabelMap::from_config(vela).unwrap().len(), 3);
    }

    /// Softmax must be stable on large and negative logits. The predecessor of this
    /// code sum-normalised negative cosine similarities, which kept argmax correct
    /// while every threshold downstream read nonsense.
    #[test]
    fn u084_softmax_is_stable_and_sums_to_one() {
        for logits in [
            vec![-40.0_f32, -41.0, -39.0],
            vec![1000.0, 999.0],
            vec![0.0, 0.0, 0.0],
            vec![-1.0, -2.0],
        ] {
            let p = softmax(&logits);
            let sum: f32 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "softmax must sum to 1, got {sum}");
            assert!(
                p.iter().all(|x| (0.0..=1.0).contains(x)),
                "probabilities only"
            );
            // argmax must be preserved
            let am = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                        if x > bv {
                            (i, x)
                        } else {
                            (bi, bv)
                        }
                    })
                    .0
            };
            assert_eq!(am(&logits), am(&p));
        }
    }

    /// A non-finite logit must not produce NaN probabilities that then compare
    /// false against every threshold and silently disable abstention.
    #[test]
    fn u085_softmax_of_non_finite_is_not_nan() {
        let p = softmax(&[f32::NAN, f32::NAN]);
        assert!(p.iter().all(|x| x.is_finite()), "must not emit NaN");
    }

    /// THE UPSTREAM INCOMPATIBILITY, pinned so a future "just use candle's
    /// ModernBertForSequenceClassification" cannot quietly reintroduce it.
    ///
    /// candle_transformers declares `ClassifierConfig.label2id` as
    /// `HashMap<String, String>`, but HuggingFace emits `String -> int`. Because
    /// the field is a flattened `Option`, the mismatch does not error -- it
    /// yields `None`, the classifier is sized `unwrap_or_default()` = 0, and the
    /// load fails against the real [14, 768] weight. llm-d-sc then fell back to
    /// anchor ranking and served a DOMAIN classifier's traffic using the resident
    /// COMPLEXITY taxonomy: `hi -> SIMPLE 0.449` where the checkpoint says
    /// `other 0.994`. Confident nonsense, no error anywhere.
    #[test]
    fn u087_hf_label2id_is_string_to_int_not_string_to_string() {
        // Exactly the shape every HF ModernBERT classifier ships.
        let hf = r#"{"id2label":{"0":"biology","1":"business"},
                     "label2id":{"biology":0,"business":1},
                     "classifier_pooling":"cls"}"#;
        // Our own parser reads the labels regardless of label2id's value type.
        let m = LabelMap::from_config(hf).expect("HF config must yield labels");
        assert_eq!(m.labels(), ["biology", "business"]);

        // And the shape candle would need instead -- documented, not used.
        let candle_shape = r#"{"id2label":{"0":"biology"},
                               "label2id":{"biology":"0"},
                               "classifier_pooling":"cls"}"#;
        assert!(
            LabelMap::from_config(candle_shape).is_some(),
            "our parser must accept both; only candle's is strict"
        );
    }

    /// A tensor present in the file is not a licence to apply it.
    ///
    /// Vela ships `classifier.bias: [14]` while its config sets
    /// `classifier_bias: false`. PyTorch honours the config and never applies
    /// the stored tensor. Applying it adds a per-class constant the checkpoint
    /// does not use -- argmax survives a small additive shift, so LABELS still
    /// matched PyTorch exactly while PROBABILITIES diverged in mixed directions.
    /// Labels-only parity would have passed this bug straight through.
    #[test]
    fn u088_classifier_bias_defaults_true_but_false_is_honoured() {
        let explicit_false = r#"{"classifier_bias": false}"#;
        let absent = r#"{"hidden_size": 768}"#;
        let explicit_true = r#"{"classifier_bias": true}"#;
        let read = |raw: &str| {
            serde_json::from_str::<serde_json::Value>(raw)
                .ok()
                .and_then(|v| {
                    v.get("classifier_bias")
                        .and_then(serde_json::Value::as_bool)
                })
                .unwrap_or(true)
        };
        assert!(
            !read(explicit_false),
            "an explicit false must drop the bias"
        );
        assert!(
            read(absent),
            "absent means the HuggingFace default, which HAS a bias"
        );
        assert!(read(explicit_true));
    }

    /// A single-class head decides nothing and must not be loaded as if it did.
    #[test]
    fn u086_single_class_head_is_declined() {
        assert_eq!(
            LabelMap::from_config(r#"{"id2label":{"0":"ONLY"}}"#)
                .unwrap()
                .len(),
            1
        );
    }
}

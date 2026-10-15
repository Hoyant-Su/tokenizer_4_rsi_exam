# Learning Domain Structure through Tokenization

Design a deterministic, reversible tokenizer that learns vocabulary and segmentation rules from native domain-specific text. The scientific question is whether recurring structures and constrained combinations can be represented in units that help a fixed-capacity language model learn and apply domain knowledge.

Start with the supplied BPE implementation. Fit the tokenizer on TRAIN only, freeze it, then train the supplied autoregressive model from scratch under the same configuration. Evaluate on TEST without adapting the tokenizer. Keep the original input bytes, vocabulary budget, training records, model size, training schedule and evaluation protocol unchanged.

A solution must learn a shared rule across domains; it must not rely on dataset-name routing, hand-written domain parsers, externally pretrained models, extra training corpora, or a separate auxiliary predictor. Smaller token counts alone do not establish improved learning.

Compare held-out BPB per dataset, native-label prediction accuracy where available, token counts, tokenizer fitting and encoding time, and model-training time. The proposed competition target is at least a 2% equal-dataset mean relative BPB gain and a two-percentage-point equal-task mean accuracy gain over BPE, with no per-dataset BPB or per-task accuracy regression. These are proposed targets, not results achieved by this baseline.

For candidate BPB b_d, baseline BPB b_d^B, candidate accuracy a_k and baseline accuracy a_k^B, define G = mean_d(1 - b_d / b_d^B), A = mean_k(a_k - a_k^B), and R = min(G / 0.02, A / 0.02). Higher R is better. BPE has R = 0; meeting the target requires R >= 1 plus the non-regression and resource constraints. Token counts and runtime remain separately reported efficiency outcomes.

This starter implements tokenizer fitting, full-model training and BPB evaluation. A sealed competition split and the additional native-label accuracy evaluator must be supplied by the competition organizer. The included test files are visible benchmark data, not a sealed exam set.

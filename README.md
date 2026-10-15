# Structured-domain tokenization baseline

A BPE baseline for learning token vocabularies from domain-specific text and evaluating them with a fixed autoregressive language model. The repository includes the training and test data, a trained tokenizer, and per-dataset and per-record results.

## Repository layout

```text
algo/                   Tokenizer and optimizer algorithms
model/                  Language-model construction and initialization
train/                  Model training, scheduling and checkpoint handling
eval/                   Language-model evaluation
scripts/                Command-line entrypoint
config/                 Fixed experiment settings
data/language_model/    Data loading and batching code
data/streams/           Input readers
data/corpus/train/      Training data
data/corpus/test/       Test data
results/bpe/            Trained BPE tokenizer and reference measurements
task/                   Research task specification
licenses/               Third-party code notices
```

`results/bpe/` contains data artifacts only, with no executable code. Running the experiment writes new outputs to `artifacts/` and model weights to `checkpoints/`; both are excluded from Git and separate from the reference results. To reproduce the BPB measurement, run the full training and test commands below. Checking the included tokenizer verifies tokenization and reconstruction, not language-model BPB.

## Install

Use Python 3.12 and an NVIDIA GPU with CUDA support for model training. Tokenizer fitting and encoding run on CPU.

```bash
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python -m scripts.run --config config/bpe.yaml --stage fit
python -m scripts.run --config config/bpe.yaml --stage test
python -m scripts.run --config config/train.yaml --stage prepare_tokens
python -m scripts.run --config config/train.yaml --stage train
python -m scripts.run --config config/train.yaml --stage test
```

To use and check the included tokenizer without fitting it again:

```bash
python -m scripts.run --config config/trained_bpe.yaml --stage test
```

Training updates all model parameters from scratch for one epoch. The tokenizer remains fixed. There is no separate instruction-tuning stage. Generated outputs go to `artifacts/`; model and optimizer checkpoints go to `checkpoints/`, retaining at most two checkpoints. Large model weights are not included in Git.

## Configuration

| Setting | Value |
|---|---|
| Seed | 42 |
| Vocabulary | 8,192 IDs, including special tokens |
| Decoder | 8 layers, hidden size 512, 8 attention heads, intermediate size 1,536 |
| Model parameters | 31,465,984 |
| Context capacity | 4,096 tokens, including BOS/EOS |
| Batch size | 8 records |
| Training | 1 epoch, all parameters trainable |
| Optimizer | Muon for eligible matrices; AdamW for the remaining parameters |
| Tokenizer threads | 8 |

All records are preserved during training and evaluation. Inputs exceeding the configured context cause an explicit error rather than truncation.

## Data

`data/corpus/train/` contains 138,706 records and `data/corpus/test/` contains 35,879 records. These files are the fixed seed-42 5% experimental split; the percentage is a sampling designation, not a dataset name. Their combined Parquet size is 74,293,430 bytes. The tokenizer reads the `text` column. Other columns retain source identifiers and associated task fields; they are not prepended to the input.

The collection contains 14 dataset sources in 15 files per split because code refinement has separate buggy and fixed files. The test set is included for reproducible evaluation and is **not a hidden competition test**. Dataset licenses remain separate; see [DATA_LICENSES.md](DATA_LICENSES.md).

## BPE results

The included tokenizer is `results/bpe/tokenizer.json`. Numerical results are in `results/bpe/metrics.json`; all 35,879 per-record results are in `results/bpe/per_record.jsonl`.

Test content BPB: **1.175644717**. Test content tokens: **7,053,358**. BPB is summed content-token negative log-likelihood divided by the original UTF-8 byte count and log(2); BOS, EOS and padding are excluded from this metric. EOS is included in the training loss.

| Dataset | Test records | BPB |
|---|---:|---:|
| moleculenet_esol | 8 | 3.025620 |
| ds1000 | 12 | 3.523774 |
| moleculenet_lipophilicity | 49 | 2.179446 |
| moleculenet_freesolv | 4 | 3.742736 |
| spider | 20 | 1.983197 |
| webnlg_2017 | 86 | 1.623225 |
| matbench_expt_gap | 57 | 2.814295 |
| ucr_ecg5000 | 64 | 2.858745 |
| pfam32_seed | 13,333 | 1.831971 |
| genomic_benchmarks_human_ensembl_regulatory | 2,913 | 1.876339 |
| ddxplus | 12,912 | 0.530287 |
| im2latex_100k | 988 | 2.358502 |
| leandojo_benchmark_4 | 2,873 | 1.674582 |
| codexglue_code_refinement | 2,560 | 0.428811 |

The pooled BPB above weights bytes. Use the individual dataset scores when comparing domain effects; the small test sets should not be used to claim statistically robust improvements.

## Task

[Task specification](task/proposal.md): learn a general vocabulary and segmentation strategy that helps a fixed model predict domain-specific text and apply domain knowledge. The baseline implementation provides tokenizer fitting, full-model training, and BPB evaluation. The combined competition score and sealed native-label task evaluator are not implemented in this starter.

## Code attribution

The BPE implementation uses Hugging Face Tokenizers. The optimizer includes Moonlight Muon code, and byte-to-Unicode mapping derives from OpenAI GPT-2. Their notices are in `licenses/`. Data are not relicensed under a software license.

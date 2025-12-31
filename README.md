# EHRMamba Reproduction (MIMIC-IV)
Reproduction of "EHRMamba: Towards Generalizable and Scalable Foundation Models for Electronic Health Records"
focused on MIMIC-IV. This repo contains standalone pretraining and finetuning scripts, plus a report summarizing
the scope, methodology, and results.

## Contents
- `odyssey_repro/`: standalone training scripts (no odyssey imports)
- `repro_report.pdf`: full reproduction report
- `LICENSE`: license for this repo

## Reproduction scope (from the report)
We reproduce key claims from EHRMamba, evaluating performance and practical reproducibility on MIMIC-IV.
The main focus is mortality prediction and training efficiency relative to Transformer baselines.

### Hypotheses
1. Clinical prediction: EHRMamba outperforms baseline models (BigBird, CEHR-BERT) on mortality prediction.
2. Efficiency: EHRMamba uses less training time and memory than Transformer baselines due to linear complexity.
3. MPF (optional): multitask prompted finetuning improves performance vs single-task finetuning.
4. Long sequences (optional): EHRMamba handles long patient histories without OOM where Transformers fail.

## Dataset and preprocessing
- Dataset: MIMIC-IV v2.2 (hosp + icu), filtered to adult patients (age >= 18).
- Size after filtering: 364,627 patients, 308,782 admissions.
- Split: 70% / 10% / 20% train/val/test, stratified by patient.
- Mortality label distribution: 90% negative, 10% positive.
- Preprocessing:
  - Convert raw data to HL7 FHIR compliant sequences.
  - Remove records with missing demographics or labels.
  - Tokenize patient timelines per EHRMamba pipeline.
  - Truncate to max token length and pad shorter sequences.

## Model summary
EHRMamba is built on the Mamba selective state-space architecture for linear-time sequence modeling.
Key configuration used in the report:
- Layers: 24 (base) or 48 (large)
- Hidden size: 1024
- State dimension: 16
- Mamba blocks with linear-time processing, RMSNorm, gated activations
- HL7 FHIR token embeddings + temporal position encodings

Training details (mortality fine-tuning):
- Loss: binary cross entropy (mortality)
- Optimizer: AdamW, lr 5e-5, weight decay 0.01
- Warmup: linear warmup for 6% of steps, then linear decay
- AMP enabled, gradient clipping at 1.0

## Results (from the report)
Mortality prediction results from the paper and reproduction:

| Model       | AUROC | AUPRC | F1   | Notes |
|------------|-------|-------|------|-------|
| XGBoost    | 0.956 | 0.795 | 0.633 | paper baseline |
| LSTM       | 0.942 | 0.771 | 0.603 | paper baseline |
| CEHR-BERT  | 0.967 | 0.857 | 0.751 | paper baseline |
| BigBird    | 0.965 | 0.852 | 0.754 | paper baseline |
| MultiBird  | 0.968 | 0.863 | 0.770 | paper baseline |
| EHRMamba   | irreproducible | irreproducible | irreproducible | training did not converge |

Validation loss (report):

| Model   | Validation Loss |
|---------|------------------|
| Baseline | 5.1676 |
| Mamba    | 4.5688 |

Training cost snapshot (report):

| Model   | Hardware     | Epochs | Avg Epoch Time |
|---------|--------------|--------|----------------|
| Baseline | RTX 5000 Ada | 2      | 902.66s |
| Mamba    | RTX 6000 Ada | 2      | 16635.6s |

## Requirements
- Python 3.10+
- PyTorch
- transformers >= 4.38
- tokenizers, pandas, scikit-learn
- mamba-ssm (needed for Mamba/Mamba2 models)

## Quickstart (standalone scripts)
All training scripts expect MEDS-preprocessed sequences with an `event_tokens` column.
See `odyssey_repro/README.md` for more detailed usage.

### Mortality finetuning
```bash
python odyssey_repro/finetune_mortality.py \
  --model-type bert \
  --sequence-path /path/to/sequences.parquet \
  --vocab-dir /path/to/vocab_dir \
  --output-dir runs/bert
```

### BERT MLM pretraining (MEDS)
```bash
python odyssey_repro/meds_mlm_train.py \
  --sequence-path /path/to/sequences.parquet \
  --vocab-dir /path/to/vocab_dir \
  --output-dir runs/meds_mlm
```

### Mamba2 causal pretraining (MEDS)
```bash
python odyssey_repro/mamba2_pretrain.py \
  --sequence-path /path/to/sequences.parquet \
  --vocab-dir /path/to/vocab_dir \
  --output-dir runs/mamba2
```

## Notes on LLM assistance (from the report)
We used an LLM to help with:
- Preprocessing code for loading multiple MEDS parquet shards.
- Training loop scaffolding, AUROC computation, and checkpointing.

The generated code required several iterations to match the actual MEDS schema,
handle large data volumes, and correctly compute metrics for multi-label targets.

## Reproducibility notes
- Training was stopped early while validation losses were still decreasing.
- The reported numbers reflect partially trained checkpoints and may be underestimates.
- Optional hypotheses (MPF and long-sequence stress tests) were not completed.

## Authors
Gladwin Lee (Glee426), ZiHao Liang (Zl334)

## References
- Paper: EHRMamba: Towards Generalizable and Scalable Foundation Models for Electronic Health Records
- Official repository: https://github.com/txzhao/EHR-Mamba
- Dataset: https://physionet.org/content/mimiciv/

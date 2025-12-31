# Finetune Mortality (standalone)

Standalone fine-tuning for MEDS mortality classification with BERT or Mamba (no odyssey imports).

## Data
- Input: parquet/csv with `event_tokens` (list or space-separated tokens) and a binary label column (default: `mortality_1month`).
- Vocab: directory containing one or more `*vocab.json` files.

## Quickstart
```bash
# BERT
python finetune_mortality.py \
  --model-type bert \
  --sequence-path /path/to/sequences.parquet \
  --vocab-dir /path/to/vocab_dir \
  --output-dir runs/bert

# Mamba (requires transformers>=4.38 + mamba-ssm)
python finetune_mortality.py \
  --model-type mamba \
  --sequence-path /path/to/sequences.parquet \
  --vocab-dir /path/to/vocab_dir \
  --output-dir runs/mamba
```

## Retrain / Resume
- Re-train on new data with prior weights (keep the same vocab):  
```bash
python finetune_mortality.py \
  --model-type bert \
  --sequence-path /new/data.parquet \
  --vocab-dir /path/to/vocab_dir \
  --pretrained-ckpt runs/bert/best_bert.pt \
  --output-dir runs/bert_retrain
```
- Swap to mamba by changing `--model-type` and pointing `--pretrained-ckpt` to `best_mamba.pt` (or `last_*.pt` if you prefer the last epoch).
- If you want to continue the same experiment but with a different seed/hyperparams, change `--output-dir` to avoid overwriting and keep `--vocab-dir` consistent with the checkpoint tokenizer.

## Notes
- Key flags: `--max-len`, `--batch-size`, `--epochs`, `--lr`, `--warmup-ratio`, `--num-layers`, `--hidden-size`, `--padding-side`.
- Uses mixed precision on CUDA by default; disable with `--disable-amp` or force CPU with `--cpu`.
- Checkpoints (best/last) and the saved tokenizer are written to `--output-dir`.

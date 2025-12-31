#!/usr/bin/env python3
"""Independent mortality fine-tuning script for MEDS (no odyssey imports).

Supports two model types:
  - bert: BertForSequenceClassification
  - mamba/mamba2: HuggingFace Mamba sequence classifier (requires transformers with mamba + mamba-ssm)

Data format: parquet/csv with columns:
  - event_tokens: list or space-separated tokens
  - label column (default: mortality_1month) with 0/1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.utils.data import DataLoader, Dataset, random_split
from time import perf_counter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def build_tokenizer(vocab_dir: str, padding_side: str = "right"):
    from transformers import PreTrainedTokenizerFast

    special_tokens = ["[PAD]", "[UNK]", "[MASK]", "[BOS]", "[EOS]", "[CLS]"]
    vocab_files = sorted(glob.glob(os.path.join(vocab_dir, "*vocab.json")))
    if not vocab_files:
        raise FileNotFoundError(f"No *vocab.json files found in {vocab_dir}")

    token_pool: List[str] = []
    for fp in vocab_files:
        with open(fp, "r") as f:
            loaded = json.load(f)
        tokens = list(loaded.keys()) if isinstance(loaded, dict) else list(loaded)
        token_pool.extend(tokens)

    merged: List[str] = []
    seen = set()
    for tok in special_tokens + token_pool:
        if tok not in seen:
            merged.append(tok)
            seen.add(tok)

    vocab = {tok: idx for idx, tok in enumerate(merged)}
    tokenizer_obj = Tokenizer(
        models.WordPiece(
            vocab=vocab,
            unk_token="[UNK]",
            max_input_chars_per_word=1000,
        )
    )
    tokenizer_obj.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )
    tokenizer.padding_side = padding_side
    return tokenizer


class MortalityDataset(Dataset):
    """Sequence classification dataset."""

    def __init__(self, frame: pd.DataFrame, tokenizer, max_len: int, label_col: str) -> None:
        if "event_tokens" not in frame.columns:
            raise ValueError("Input dataframe must contain an 'event_tokens' column.")
        if label_col not in frame.columns:
            raise ValueError(f"Label column '{label_col}' not found.")
        self.df = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _normalize_tokens(raw_tokens: Sequence[str]) -> List[str]:
        if isinstance(raw_tokens, str):
            return raw_tokens.strip().split()
        if isinstance(raw_tokens, list):
            return raw_tokens
        if hasattr(raw_tokens, "tolist"):
            return list(raw_tokens.tolist())
        return list(raw_tokens)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.loc[idx]
        tokens = self._normalize_tokens(row["event_tokens"])
        tokens = tokens[: self.max_len - 2]
        tokens = [self.tokenizer.bos_token] + tokens + [self.tokenizer.eos_token]

        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            add_special_tokens=False,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row[self.label_col]), dtype=torch.long),
        }


def load_frame(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Only parquet or csv supported.")


def create_loaders(df: pd.DataFrame, tokenizer, args):
    dataset = MortalityDataset(df, tokenizer, args.max_len, args.label_col)
    val_size = int(len(dataset) * args.val_fraction)
    test_size = int(len(dataset) * args.test_fraction)
    train_size = len(dataset) - val_size - test_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("Split fractions yield empty subset.")
    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=gen)
    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True, batch_size=args.batch_size)
    return (
        DataLoader(train_set, shuffle=True, **loader_kwargs),
        DataLoader(val_set, shuffle=False, **loader_kwargs),
        DataLoader(test_set, shuffle=False, **loader_kwargs),
    )


def load_model_and_config(args, tokenizer):
    # Import BERT up front; import Mamba lazily so bert runs don't require the Mamba dependency/version.
    from transformers import BertConfig, BertForSequenceClassification

    def _resolve_mamba_classifier():
        """Return (MambaConfig, MambaForSequenceClassification) with fallbacks for older transformers."""
        try:
            from transformers import MambaConfig, MambaForSequenceClassification
            return MambaConfig, MambaForSequenceClassification
        except Exception:
            try:
                from transformers import MambaConfig
                from transformers.models.mamba.modeling_mamba import MambaForSequenceClassification
                return MambaConfig, MambaForSequenceClassification
            except Exception:
                try:
                    from transformers import MambaConfig
                    from transformers.activations import ACT2FN
                    from transformers.modeling_outputs import SequenceClassifierOutput
                    from transformers.models.mamba.modeling_mamba import MambaModel, MambaPreTrainedModel
                except Exception as exc:
                    raise ImportError(
                        "need transformers>=4.38 and install mamba-ssm， Mamba classification。"
                    ) from exc

                class FallbackMambaForSequenceClassification(MambaPreTrainedModel):
                    def __init__(self, config):
                        super().__init__(config)
                        self.num_labels = config.num_labels
                        hidden_size = getattr(config, "hidden_size", getattr(config, "d_model", None))
                        if hidden_size is None:
                            raise ValueError("MambaConfig lack of hidden_size/d_model")
                        dropout = getattr(config, "classifier_dropout", getattr(config, "hidden_dropout_prob", 0.1))
                        self.backbone = MambaModel(config)
                        self.dropout = torch.nn.Dropout(dropout)
                        self.dense = torch.nn.Linear(hidden_size, hidden_size)
                        self.act = ACT2FN[config.hidden_act]
                        self.out_proj = torch.nn.Linear(hidden_size, config.num_labels)
                        self.post_init()

                    def forward(
                        self,
                        input_ids: torch.LongTensor = None,
                        attention_mask: torch.Tensor | None = None,
                        inputs_embeds: torch.Tensor | None = None,
                        labels: torch.LongTensor | None = None,
                        output_hidden_states: bool | None = None,
                        return_dict: bool | None = None,
                    ):
                        outputs = self.backbone(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            inputs_embeds=inputs_embeds,
                            output_hidden_states=output_hidden_states,
                            return_dict=True,
                        )
                        hidden_states = outputs.last_hidden_state
                        batch_size = hidden_states.size(0)
                        if attention_mask is not None:
                            last_token_idx = attention_mask.sum(dim=1) - 1
                        else:
                            last_token_idx = torch.full(
                                (batch_size,),
                                hidden_states.size(1) - 1,
                                device=hidden_states.device,
                                dtype=torch.long,
                            )
                        pooled = hidden_states[torch.arange(batch_size, device=hidden_states.device), last_token_idx]
                        x = self.dropout(pooled)
                        x = self.act(self.dense(x))
                        x = self.dropout(x)
                        logits = self.out_proj(x)
                        loss = None
                        if labels is not None:
                            loss = torch.nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))
                        return SequenceClassifierOutput(
                            loss=loss,
                            logits=logits,
                            hidden_states=outputs.hidden_states,
                        )

                return MambaConfig, FallbackMambaForSequenceClassification

    if args.model_type == "bert":
        config = BertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            intermediate_size=args.intermediate_size,
            hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.dropout,
            pad_token_id=tokenizer.pad_token_id,
            cls_token_id=tokenizer.cls_token_id,
            num_labels=2,
            max_position_embeddings=args.max_len,
        )
        model = BertForSequenceClassification(config)
        if args.pretrained_ckpt:
            state = torch.load(args.pretrained_ckpt, map_location="cpu")["model_state"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"Loaded pretrained weights with missing={missing}, unexpected={unexpected}")
    else:
        MambaConfig, MambaForSequenceClassification = _resolve_mamba_classifier()

        config = MambaConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.hidden_size,
            n_layer=args.num_layers,
            expand=args.expand,
            conv_kernel=args.conv_kernel,
            rms_norm=args.rms_norm,
            hidden_act=args.hidden_act,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_labels=2,
            max_position_embeddings=args.max_len,
        )
        model = MambaForSequenceClassification(config)
        if args.pretrained_ckpt:
            state = torch.load(args.pretrained_ckpt, map_location="cpu")["model_state"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"Loaded pretrained weights with missing={missing}, unexpected={unexpected}")
    return model


def evaluate(model, loader, device) -> Tuple[float, float, float]:
    model.eval()
    losses, preds, labels = [], [], []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=y,
            )
            loss = outputs.loss if hasattr(outputs, "loss") else criterion(outputs.logits, y)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds.extend(probs.cpu().tolist())
            labels.extend(y.cpu().tolist())
            losses.append(loss.item())
    avg_loss = float(np.mean(losses))
    try:
        auroc = roc_auc_score(labels, preds)
    except ValueError:
        auroc = float("nan")
    ap = average_precision_score(labels, preds)
    bin_preds = [1 if p >= 0.5 else 0 for p in preds]
    f1 = f1_score(labels, bin_preds)
    return avg_loss, auroc, f1, ap


def train(args):
    set_seed(args.seed)
    try:
        from transformers import get_linear_schedule_with_warmup
    except Exception as exc:  # pragma: no cover
        raise ImportError("需要 transformers>=4.38 并安装 mamba-ssm（如果用 mamba2）。") from exc

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer = build_tokenizer(args.vocab_dir, args.padding_side)
    df = load_frame(args.sequence_path)
    train_loader, val_loader, test_loader = create_loaders(df, tokenizer, args)
    model = load_model_and_config(args, tokenizer).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not args.disable_amp))

    best_val = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
        epoch_start = perf_counter()
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % args.log_every == 0:
                print(f"Epoch {epoch} | step {step}/{len(train_loader)} | loss {running/args.log_every:.4f}")
                running = 0.0

        torch.cuda.synchronize(device) if device.type == "cuda" else None
        epoch_time = perf_counter() - epoch_start
        val_loss, val_auc, val_f1, val_ap = evaluate(model, val_loader, device)
        peak_mem = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else float("nan")
        )
        print(
            f"Epoch {epoch} | val_loss {val_loss:.4f} | AUROC {val_auc:.4f} | "
            f"F1 {val_f1:.4f} | AP {val_ap:.4f} | "
            f"epoch_time {epoch_time:.1f}s | peak_mem {peak_mem:.2f} GB"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "metrics": {"auroc": val_auc, "f1": val_f1, "ap": val_ap},
                },
                os.path.join(args.output_dir, f"best_{args.model_type}.pt"),
            )
            tokenizer.save_pretrained(args.output_dir)
            print(f"Saved best checkpoint to {args.output_dir}")

    test_loss, test_auc, test_f1, test_ap = evaluate(model, test_loader, device)
    print(f"Test | loss {test_loss:.4f} | AUROC {test_auc:.4f} | F1 {test_f1:.4f} | AP {test_ap:.4f}")
    torch.save(
        {
            "model_state": model.state_dict(),
            "test_metrics": {"loss": test_loss, "auroc": test_auc, "f1": test_f1, "ap": test_ap},
        },
        os.path.join(args.output_dir, f"last_{args.model_type}.pt"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Independent mortality finetune on MEDS.")
    parser.add_argument("--model-type", choices=["bert", "mamba", "mamba2"], required=True)
    parser.add_argument("--sequence-path", required=True, help="Parquet/CSV with event_tokens + label column.")
    parser.add_argument("--vocab-dir", required=True, help="Directory containing *vocab.json files.")
    parser.add_argument("--output-dir", default="repro_finetune_runs")
    parser.add_argument("--label-col", default="mortality_1month", help="Binary label column name.")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--expand", type=int, default=2, help="Mamba expand factor.")
    parser.add_argument("--conv-kernel", type=int, default=4, help="Mamba conv kernel size.")
    parser.add_argument("--rms-norm", action="store_true", help="Use RMSNorm for Mamba.")
    parser.add_argument("--hidden-act", type=str, default="silu")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--padding-side", choices=["left", "right"], default="right")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    parser.add_argument("--disable-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--pretrained-ckpt", default=None, help="Path to pretrain checkpoint (best.pt/last.pt).")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

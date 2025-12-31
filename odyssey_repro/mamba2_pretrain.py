#!/usr/bin/env python3
"""Independent MEDS causal pretraining script for Mamba2 (no odyssey imports).

This uses HuggingFace's Mamba implementation (requires transformers >= 4.38 and mamba-ssm).
Training objective: next-token prediction (causal LM) on MEDS event token sequences.
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
    """Build a WordPiece tokenizer from MEDS vocab files."""
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


class MedsCausalDataset(Dataset):
    """Causal LM dataset for MEDS sequences."""

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer,
        max_len: int,
    ) -> None:
        if "event_tokens" not in frame.columns:
            raise ValueError("Input dataframe must contain an 'event_tokens' column.")
        self.df = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

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
        raw = self.df.loc[idx, "event_tokens"]
        tokens = self._normalize_tokens(raw)
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
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_loaders(dataset: Dataset, batch_size: int, val_fraction: float, num_workers: int, seed: int):
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    if val_size == 0 or train_size == 0:
        raise ValueError("val_fraction produces empty train or val split.")
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=gen)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    try:
        from transformers import MambaConfig, MambaForCausalLM, get_linear_schedule_with_warmup
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "需要 transformers>=4.38 且安装 mamba-ssm。请先 `pip install 'transformers>=4.38' mamba-ssm`。"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    tokenizer = build_tokenizer(args.vocab_dir, args.padding_side)
    df = pd.read_parquet(args.sequence_path)
    dataset = MedsCausalDataset(frame=df, tokenizer=tokenizer, max_len=args.max_len)
    train_loader, val_loader = create_loaders(
        dataset=dataset,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    config = MambaConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.hidden_size,
        n_layer=args.num_layers,
        expand=args.expand,
        conv_kernel=args.conv_kernel,
        rms_norm=args.rms_norm,
        hidden_act=args.hidden_act,
        tie_word_embeddings=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_position_embeddings=args.max_len,
    )
    model = MambaForCausalLM(config).to(device)

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
        train_time = perf_counter() - epoch_start

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)
        max_mem = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else float("nan")
        )
        print(
            f"Epoch {epoch} finished | val_loss {val_loss:.4f} | "
            f"epoch_time {train_time:.1f}s | peak_mem {max_mem:.2f} GB"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config.to_dict(),
                    "val_loss": val_loss,
                },
                os.path.join(args.output_dir, "best.pt"),
            )
            tokenizer.save_pretrained(args.output_dir)
            print(f"Saved best checkpoint to {args.output_dir}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "val_loss": best_val,
        },
        os.path.join(args.output_dir, "last.pt"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Independent Mamba2 causal pretraining on MEDS.")
    parser.add_argument("--sequence-path", required=True, help="MEDS parquet with event_tokens column.")
    parser.add_argument("--vocab-dir", required=True, help="Directory containing *vocab.json files.")
    parser.add_argument("--output-dir", default="repro_mamba2_ckpts", help="Where to save checkpoints/tokenizer.")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--conv-kernel", type=int, default=4)
    parser.add_argument("--rms-norm", action="store_true", help="Use RMSNorm (recommended for Mamba).")
    parser.add_argument("--hidden-act", type=str, default="silu")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--padding-side", choices=["left", "right"], default="right")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    parser.add_argument("--disable-amp", action="store_true", help="Disable mixed precision.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

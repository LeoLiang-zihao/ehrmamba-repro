#!/usr/bin/env python3
"""Independent MEDS masked language model pretraining script.

This implementation is separated from the original odyssey training code to comply
with the BDH reproducibility challenge requirement of writing our own code. It
expects MEDS-preprocessed sequences (parquet) with an `event_tokens` column.
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
from transformers import (
    BertConfig,
    BertForMaskedLM,
    PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup,
)


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def build_tokenizer(vocab_dir: str, padding_side: str = "right") -> PreTrainedTokenizerFast:
    """
    Build a WordPiece tokenizer from MEDS vocabulary json files.

    We intentionally reimplement this instead of importing the project's tokenizer.
    """
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

    # Deduplicate while keeping order: special tokens first, then vocab tokens
    merged: List[str] = []
    seen = set()
    for token in special_tokens + token_pool:
        if token not in seen:
            merged.append(token)
            seen.add(token)

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


class MedsPretrainDataset(Dataset):
    """Lightweight dataset for MLM pretraining on MEDS sequences."""

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: PreTrainedTokenizerFast,
        max_len: int,
        mask_prob: float,
    ) -> None:
        if "event_tokens" not in frame.columns:
            raise ValueError("Input dataframe must contain an 'event_tokens' column.")
        self.df = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.special_ids = set(tokenizer.all_special_ids)

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _normalize_tokens(raw_tokens: Sequence[str]) -> List[str]:
        """Convert stored tokens to a plain list of strings."""
        if isinstance(raw_tokens, str):
            return raw_tokens.strip().split()
        if isinstance(raw_tokens, list):
            return raw_tokens
        if hasattr(raw_tokens, "tolist"):
            return list(raw_tokens.tolist())
        return list(raw_tokens)

    def _mask_tokens(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply BERT-style masking."""
        labels = input_ids.clone()
        prob = torch.full(labels.shape, self.mask_prob, device=input_ids.device)
        is_special = torch.tensor(
            [tid in self.special_ids for tid in labels.tolist()],
            device=input_ids.device,
            dtype=torch.bool,
        )
        prob[is_special] = 0

        mask = torch.bernoulli(prob).bool()
        labels[~mask] = -100

        replace_with_mask = torch.bernoulli(torch.full(labels.shape, 0.8, device=input_ids.device)).bool() & mask
        input_ids[replace_with_mask] = self.tokenizer.mask_token_id

        replace_with_rand = (
            torch.bernoulli(torch.full(labels.shape, 0.1, device=input_ids.device)).bool()
            & mask
            & ~replace_with_mask
        )
        random_ids = torch.randint(
            low=0,
            high=self.tokenizer.vocab_size,
            size=labels.shape,
            device=input_ids.device,
        )
        input_ids[replace_with_rand] = random_ids[replace_with_rand]
        return input_ids, labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        raw = self.df.loc[idx, "event_tokens"]
        tokens = self._normalize_tokens(raw)
        # Add BOS/EOS manually and truncate to fit within max_len.
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
        input_ids, labels = self._mask_tokens(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_sequences(parquet_path: str) -> pd.DataFrame:
    """Load MEDS-preprocessed sequences from parquet."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Could not find parquet file at {parquet_path}")
    return pd.read_parquet(parquet_path)


def create_loaders(
    dataset: Dataset,
    batch_size: int,
    val_fraction: float,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Split dataset into train/val and create dataloaders."""
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
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    tokenizer = build_tokenizer(args.vocab_dir, args.padding_side)
    df = load_sequences(args.sequence_path)
    dataset = MedsPretrainDataset(
        frame=df,
        tokenizer=tokenizer,
        max_len=args.max_len,
        mask_prob=args.mask_prob,
    )
    train_loader, val_loader = create_loaders(
        dataset=dataset,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        max_position_embeddings=args.max_len,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = BertForMaskedLM(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
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
        model.train()
        running_loss = 0.0
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
            running_loss += loss.item()

            if step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                print(f"Epoch {epoch} | step {step}/{len(train_loader)} | train_loss {avg_loss:.4f}")
                running_loss = 0.0

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
        print(f"Epoch {epoch} finished | val_loss {val_loss:.4f}")

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
            print(f"Saved best checkpoint with val_loss {val_loss:.4f} to {args.output_dir}")

    # Always keep the final weights too
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "val_loss": best_val,
        },
        os.path.join(args.output_dir, "last.pt"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent MEDS MLM pretraining (no odyssey imports).")
    parser.add_argument("--sequence-path", required=True, help="Path to MEDS parquet with event_tokens column.")
    parser.add_argument("--vocab-dir", required=True, help="Directory containing *vocab.json files from MEDS.")
    parser.add_argument("--output-dir", default="repro_checkpoints", help="Where to save checkpoints and tokenizer.")
    parser.add_argument("--max-len", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--mask-prob", type=float, default=0.15, help="Probability of masking tokens.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--padding-side", choices=["left", "right"], default="right")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50, help="Logging interval in training steps.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--disable-amp", action="store_true", help="Disable mixed precision.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

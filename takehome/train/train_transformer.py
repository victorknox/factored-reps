#!/usr/bin/env python3
"""Train a small attention-only transformer on the non-ergodic Mess3 dataset."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from experiments.models.attention_only import AttentionOnly, AttentionOnlyConfig


def load_data(data_dir):
    train = np.load(data_dir / "train_data.npz")
    val = np.load(data_dir / "val_data.npz")
    return train, val


def make_batches(tokens, batch_size, shuffle=True, rng=None):
    """Yield input/target pairs for next-token prediction."""
    n = len(tokens)
    idx = np.arange(n)
    if shuffle:
        (rng or np.random).shuffle(idx)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_tokens = tokens[idx[start:end]]
        inputs = torch.tensor(batch_tokens[:, :-1], dtype=torch.long)
        targets = torch.tensor(batch_tokens[:, 1:], dtype=torch.long)
        yield inputs, targets


def evaluate(model, val_tokens, batch_size, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for inputs, targets in make_batches(val_tokens, batch_size, shuffle=False):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
            total_tokens += targets.numel()
    return total_loss / total_tokens


def train(config):
    data_dir = Path(config["data_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    train_data = np.load(data_dir / "train_data.npz")
    val_data = np.load(data_dir / "val_data.npz")
    train_tokens = train_data["tokens"]
    val_tokens = val_data["tokens"]
    print(f"Train: {train_tokens.shape}, Val: {val_tokens.shape}")

    seq_len = train_tokens.shape[1]
    n_ctx = seq_len - 1  # input length (predict next token)

    # Build model
    cfg = AttentionOnlyConfig(
        d_vocab=config["d_vocab"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        n_ctx=n_ctx,
        normalization_type=config.get("normalization_type", "LN"),
        seed=config.get("seed", 42),
    )
    model = AttentionOnly(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: d_model={cfg.d_model}, n_heads={cfg.n_heads}, n_layers={cfg.n_layers}, "
          f"n_ctx={cfg.n_ctx}, params={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                  weight_decay=config.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"], eta_min=config["lr"] / 10
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    rng = np.random.RandomState(config.get("seed", 42))
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    checkpoint_epochs = config.get("checkpoint_epochs", [])

    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0

        for inputs, targets in make_batches(train_tokens, config["batch_size"], shuffle=True, rng=rng):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            if config.get("grad_clip", 0) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item() * targets.numel()
            epoch_tokens += targets.numel()

        scheduler.step()

        train_loss = epoch_loss / epoch_tokens
        val_loss = evaluate(model, val_tokens, config["batch_size"], device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        if (epoch + 1) % config.get("log_every", 5) == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                  f"best_val={best_val_loss:.4f} | lr={scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) in checkpoint_epochs:
            torch.save(model.state_dict(), output_dir / f"checkpoint_epoch{epoch+1}.pt")

    # Save final model and training history
    torch.save(model.state_dict(), output_dir / "final_model.pt")

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "config": config,
        "n_params": n_params,
    }
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Also save model config for loading
    model_config = {
        "d_vocab": cfg.d_vocab, "d_model": cfg.d_model,
        "n_heads": cfg.n_heads, "n_layers": cfg.n_layers,
        "n_ctx": cfg.n_ctx, "normalization_type": cfg.normalization_type,
    }
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Saved to {output_dir}")
    return model, history


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = {
            "d_vocab": 3,
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 4,
            "normalization_type": "LN",
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "batch_size": 512,
            "num_epochs": 200,
            "log_every": 10,
            "seed": 42,
            "checkpoint_epochs": [10, 25, 50, 100, 150, 200],
        }

    config["data_dir"] = args.data_dir or str(base_dir / "results")
    config["output_dir"] = args.output_dir or str(base_dir / "results" / "checkpoints")

    train(config)


if __name__ == "__main__":
    main()

"""
Smoke test: instantiate the paper model, optionally load a checkpoint, and run
one forward pass with a single synthetic (or provided) PPG recording.

Usage:
    python smoke_test.py                              # synthetic data, random weights
    python smoke_test.py --checkpoint path/to/ckpt   # load pretrained weights
    python smoke_test.py --input sample_data/example_ppg.npy  # custom input
"""
import argparse
import os
import sys
from functools import partial

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from model.architecture.mae_vit import MAE1DViT


# ── Paper model config (MAE1d_on_segments, 256-dim) ───────────────────────────
PAPER_CONFIG = dict(
    in_chans=3,           # SpO2, HR, PAT waveform
    input_size=15_000,    # 120 s × 125 Hz
    patch_size=125,       # 1-second patches → 120 patches total
    embed_dim=256,
    decoder_embed_dim=256,
    depth=24,
    num_heads=16,
    decoder_num_heads=16,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    loss_version="v3.3",
    loss_channels=[2],    # PAT waveform channel (relative index)
    ch_weights=None,
)


def build_model() -> MAE1DViT:
    return MAE1DViT(**PAPER_CONFIG)


def load_checkpoint(model: MAE1DViT, ckpt_path: str, device: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state.get("model", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing keys  ({len(missing)}): {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")
    print(f"  Loaded checkpoint: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="SleepFM smoke test")
    parser.add_argument("--checkpoint", type=str,
                        default="/net/mraid20/export/jafar/SleepFM/ssl_sleep/models/checkpoint-final1000steps_256D_2wNonep128b_SSL_MAE1d_on_segments_in3_out_pat_infra_s120_p125_m0.5_lr0.001_scNone_loss_v3.3_stratFalse_epoch143.pth",
                        help="Path to the pretrained checkpoint")
    parser.add_argument("--input", type=str,
                        default=os.path.join(os.path.dirname(__file__), "sample_data", "example_ppg.npy"),
                        help="Path to a .npy file of shape (1, 3, 15000)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Build model ──────────────────────────────────────────────────────────
    print("\nBuilding model…")
    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    if args.checkpoint and os.path.exists(args.checkpoint):
        print("Loading checkpoint…")
        load_checkpoint(model, args.checkpoint, device)
    elif args.checkpoint:
        print(f"  Checkpoint not found at {args.checkpoint} — running with random weights.")

    model.eval()

    # ── Load / generate input ─────────────────────────────────────────────────
    if os.path.exists(args.input):
        x = np.load(args.input).astype(np.float32)
        print(f"\nLoaded input from {args.input}  shape={x.shape}")
    else:
        print(f"\nInput file not found ({args.input}), using random synthetic data.")
        x = np.random.randn(1, 3, 15_000).astype(np.float32)

    assert x.ndim == 3 and x.shape[1] == 3 and x.shape[2] == 15_000, (
        f"Expected shape (B, 3, 15000), got {x.shape}"
    )
    x_t = torch.from_numpy(x).to(device)

    # ── Forward pass ──────────────────────────────────────────────────────────
    print("Running forward pass  (mask_ratio=0.5)…")
    with torch.no_grad():
        total_loss, pred, mask, padding_mask, latent, target, *_ = model(
            x_t, mask_ratio=0.5
        )

    # CLS token embedding (B, embed_dim)
    embedding = latent[:, 0, :]

    print("\n── Results ──────────────────────────────────────────────────────")
    print(f"  Embedding shape : {tuple(embedding.shape)}")
    print(f"  Embedding norm  : {embedding.norm(dim=-1).item():.4f}")
    print(f"  Reconstruction loss : {total_loss.item():.6f}")
    print(f"  Pred patches shape  : {tuple(pred.shape)}")
    print(f"  Masked patches      : {mask.sum().item():.0f} / {mask.numel()}")
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()

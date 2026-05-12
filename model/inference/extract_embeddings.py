#!/usr/bin/env python3
"""
Create segment-level embeddings from a checkpoint using the original (non-segmented) data path.
Uses an unshuffled DataLoader over the "preprocessed_data_SR_125Hz_120s_segments_gold" directory;
each recording file is (num_segments, C, segment_length), segments are [i, :, :].
Output: one embedding per segment, saved to HDF5 for large datasets (~7M segments).
"""

import csv
import os
from tkinter import FALSE

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

try:
    import h5py
except ImportError:
    h5py = None

try:
    import wandb
except ImportError:
    wandb = None

# Project imports
from model.utils import PARENT_DIR, PROJECT_DIR
from model.data.datasets import TemporalChannelwiseZScore

# -----------------------------
# User-editable global settings
# -----------------------------
CHECKPOINT_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "checkpoint-final1000steps_256D_2wNonep128b_SSL_MAE1d_on_segments_in3_out_pat_infra_s120_p125_m0.5_lr0.001_scNone_loss_v3.3_stratFalse_epoch143.pth",
)
DATA_DIR = os.path.join(PROJECT_DIR, "preprocessed_data_SR_125Hz_120s_segments_gold")
SPLIT = "gold_all"
SPLIT_DIR = "randomstate1"
USE_CLS_TOKEN = True
POOLING_METHOD = "all"  # "avg" | "max" | "all" # relevant if USE_CLS_TOKEN is False
OUTPUT_DIR = os.path.join(PROJECT_DIR, "embeddings")
SAVE_FILENAME = None  # if None, auto-derived from checkpoint + split
SAVE_FORMAT = "h5"  # "h5" (HDF5) for large datasets (~7M segments), "csv" for smaller
N_JOBS = 10  # DataLoader num_workers for file loading (0 = main process only)
RUN_SERIAL = False  # if True, use num_workers=0 (single-process loading)
BATCH_SIZE = 128  # model forward batch size (per recording file)
DEVICE = None  # "cuda" | "cpu" | None for auto
WANDB_PROJECT = "sleep_embeddings"
WANDB_LOG_EVERY_N_RECORDINGS = 50  # log progress to wandb every N recordings


def _parse_checkpoint_config(checkpoint_path: str):
    """Parse model config from checkpoint path (same logic as get_mae_latent)."""
    architecture = checkpoint_path.split("MAE1d_on_")[1].split("_")[0]
    architecture = "MAE1d_on_" + architecture
    in_channels = int(checkpoint_path.split("_in")[1].split("_")[0])
    patch_size = int(checkpoint_path.split("_p")[2].split("_")[0])
    loss_version = checkpoint_path.split("loss_v")[1].split("_")[0]
    embed_dim = 256
    relative_loss_channels = [2]
    input_channels = [0, 1, 2] if in_channels == 3 else [0, 1, 2, 3, 4, 5, 6, 7]
    input_size = 125 * 120
    return {
        "architecture": architecture,
        "in_channels": in_channels,
        "patch_size": patch_size,
        "loss_version": loss_version,
        "embed_dim": embed_dim,
        "relative_loss_channels": relative_loss_channels,
        "input_channels": input_channels,
        "input_size": input_size,
    }


def _read_id_column_from_csv(path: str):
    """Read CSV and return set of second part of RegistrationCode (split by '_')[1]."""
    out = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get("RegistrationCode", "")
            parts = code.split("_")
            if len(parts) >= 2:
                out.add(parts[1])
    return out


def _get_subset_ids(split: str, split_dir: str = "randomstate1"):
    """Load recording subset IDs for the given split."""
    if split == "gold_all":
        train_path = os.path.join(PARENT_DIR, "data_splits", split_dir, f"gold_train_ids_{split_dir}.csv")
        val_path = os.path.join(PARENT_DIR, "data_splits", split_dir, f"gold_val_ids_{split_dir}.csv")
        train_ids = _read_id_column_from_csv(train_path)
        val_ids = _read_id_column_from_csv(val_path)
        return train_ids | val_ids
    split_path = os.path.join(PARENT_DIR, "data_splits", split_dir, f"{split}_ids_{split_dir}.csv")
    return _read_id_column_from_csv(split_path)


def _list_recording_files(data_dir: str, split: str, split_dir: str = "randomstate1"):
    """
    List (file_path, recording_id) for all recordings in data_dir (non-segmented layout).
    Expects either:
      - Flat: data_dir/*.pt with filenames like preprocessed_<id>__<visit>__<night>.pt
      - Nested: data_dir/preprocessed_<name>/ with one or more .pt files (we use first .pt per dir)
    Each .pt file is assumed to have shape [N, C, segment_length] (one file per recording).
    Returns list of (path, recording_id) in sorted order (unshuffled).
    """
    subset_ids = _get_subset_ids(split, split_dir)
    tasks = []

    # Flat layout: data_dir/preprocessed_xxx.pt
    if os.path.isfile(data_dir):
        raise ValueError("data_dir must be a directory")
    names = sorted(os.listdir(data_dir))
    for name in names:
        path = os.path.join(data_dir, name)
        if name.startswith("preprocessed_") and name.endswith(".pt"):
            # recording_id = stem, e.g. preprocessed_123__visit__night
            recording_id = os.path.splitext(name)[0]
            try:
                pid = name.split("_")[1].split("__")[0]
            except IndexError:
                pid = name.split("_")[1] if len(name.split("_")) > 1 else ""
            if pid in subset_ids:
                tasks.append((path, recording_id))
        elif name.startswith("preprocessed_") and os.path.isdir(path):
            pt_files = sorted(f for f in os.listdir(path) if f.endswith(".pt"))
            if not pt_files:
                continue
            # Use first .pt in dir; assume it contains [N, C, L] for this recording
            first_pt = os.path.join(path, pt_files[0])
            recording_id = name
            try:
                pid = name.split("_")[1].split("__")[0]
            except IndexError:
                pid = name.split("_")[1]
            if pid in subset_ids:
                tasks.append((first_pt, recording_id))

    return tasks


class SegmentsPerRecordingDataset(Dataset):
    """
    Dataset over recordings in preprocessed_data_SR_125Hz_120s_segments_gold.
    Each .pt file has shape (num_segments, C, segment_length). __getitem__(idx) loads
    the idx-th file and returns (segments_tensor, recording_id), where segments_tensor
    is (num_segments, C, segment_length) and each segment is [i, :, :].
    """

    def __init__(self, file_tasks, channels):
        self.file_tasks = file_tasks  # list of (path, recording_id)
        self.channels = channels

    def __len__(self):
        return len(self.file_tasks)

    def __getitem__(self, idx):
        path, recording_id = self.file_tasks[idx]
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}") from e
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Expected tensor, got {type(data)}")
        if data.dim() == 2:
            data = data.unsqueeze(0)
        if data.dim() != 3:
            raise ValueError(f"Expected 3D [N,C,L], got shape {data.shape}")
        if data.shape[1] != len(self.channels):
            data = data[:, self.channels, :]
        if torch.isnan(data).any() or torch.isinf(data).any():
            data = torch.nan_to_num(data)
        return data, recording_id


def _collate_single_recording(batch):
    """Collate for batch_size=1: return (segments_tensor, recording_id) without extra batch dim."""
    assert len(batch) == 1
    return batch[0][0], batch[0][1]


def _load_one_file(args_tuple):
    """
    Load one .pt file and apply channel selection; return (recording_id, segments_tensor, error).
    Runs in a worker process. segments_tensor shape: (N, C, L). No transform (done in main).
    """
    file_path, recording_id, channels = args_tuple
    try:
        data = torch.load(file_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return recording_id, None, str(e)
    if not isinstance(data, torch.Tensor):
        return recording_id, None, f"expected tensor, got {type(data)}"
    if data.dim() == 2:
        data = data.unsqueeze(0)
    if data.dim() != 3:
        return recording_id, None, f"expected 3D [N,C,L], got shape {data.shape}"
    if data.shape[1] != len(channels):
        data = data[:, channels, :]
    if torch.isnan(data).any() or torch.isinf(data).any():
        data = torch.nan_to_num(data)
    return recording_id, data, None


def _build_transform(input_channels):
    return transforms.Compose([TemporalChannelwiseZScore(channels=input_channels)])


def _latent_to_arrays(ids, latent, use_cls_token, pooling_method):
    """
    Convert encoder latent tensor to (ids, embeddings_np).
    Same logic as get_latent_dataframe but returns (list of ids, np.ndarray) or (None, None) on error.
    """
    if use_cls_token:
        latent = latent[:, 0, :].detach().cpu().numpy()
    else:
        latent = latent[:, 1:, :]  # remove class token
        num_of_patches = latent.shape[1]
        latent = latent.permute(0, 2, 1)  # (batch_size, embed_dim, patches)
        if pooling_method == "all":
            try:
                from scipy.stats import skew, kurtosis
                stats = []
                stats.append(torch.mean(latent, dim=2))
                stats.append(torch.std(latent, dim=2))
                stats.append(torch.median(latent, dim=2).values)
                stats.append(torch.quantile(latent.float(), 0.25, dim=2))
                stats.append(torch.quantile(latent.float(), 0.75, dim=2))
                stats.append(stats[-1] - stats[-2])  # IQR
                latent_np = latent.cpu().numpy()
                dev = latent.device
                stats.append(torch.from_numpy(skew(latent_np, axis=2)).to(dev))
                stats.append(torch.abs(latent - latent.mean(dim=2, keepdim=True)).mean(dim=2))
                stats.append(torch.from_numpy(kurtosis(latent_np, axis=2)).to(dev))
                latent = torch.cat(stats, dim=1)
                latent = latent.detach().cpu().numpy()
            except Exception as e:
                print(f"Error calculating all pooling: {e}")
                return None, None
        else:
            if pooling_method == "max":
                pool = torch.nn.MaxPool1d(kernel_size=num_of_patches)
            elif pooling_method == "avg":
                pool = torch.nn.AvgPool1d(kernel_size=num_of_patches)
            else:
                return None, None
            pooled = pool(latent).squeeze(-1)
            latent = pooled.detach().cpu().numpy()
    if len(ids) != latent.shape[0]:
        return None, None
    return ids, latent.astype(np.float32)


def create_embeddings_from_checkpoint(
    checkpoint_path: str,
    data_dir: str = None,
    split: str = "gold_all",
    split_dir: str = "randomstate1",
    use_cls_token: bool = False,
    pooling_method: str = "all",
    output_dir: str = None,
    save_filename: str = None,
    save_format: str = "h5",
    n_jobs: int = 4,
    run_serial: bool = None,
    batch_size: int = 128,
    device: str = None,
    wandb_project: str = "sleep_embeddings",
    wandb_log_every_n_recordings: int = 50,
):
    """
    Create segment-level embeddings from a checkpoint using the original (non-segmented) data path.
    Uses an unshuffled DataLoader: each .pt file is (num_segments, C, segment_length), segments [i,:,:].
    One embedding per segment. For large datasets (~7M segments), saves to HDF5 (.h5); otherwise CSV.

    Args:
        checkpoint_path: Path to .pth checkpoint.
        data_dir: Directory with .pt files (e.g. preprocessed_data_SR_125Hz_120s_segments_gold).
        split: Dataset split ('gold_all', 'gold_train', 'gold_val', etc.).
        split_dir: Split directory name (e.g. 'randomstate1').
        use_cls_token: Use only class token as embedding.
        pooling_method: 'avg' / 'max' / 'all' for patch pooling.
        output_dir: Where to save outputs. Default: PROJECT_DIR + 'embeddings'.
        save_filename: Output filename (without extension). If None, derived from checkpoint and split.
        save_format: 'h5' (HDF5) or 'csv'. Use 'h5' for large datasets.
        n_jobs: DataLoader num_workers (0 when run_serial True).
        run_serial: If True, use num_workers=0.
        batch_size: Model forward batch size per recording's segments.
        device: 'cuda' or 'cpu'. Auto if None.
        wandb_project: wandb project name for logging.
        wandb_log_every_n_recordings: Log to wandb every N recordings.

    Returns:
        When save_format=='h5': None (embeddings in .h5 file).
        When save_format=='csv': (recording_ids, embeddings_array) with embeddings_array shape (n_segments, embed_dim).
    """
    import model.architecture.mae_vit as models_mae

    if data_dir is None:
        data_dir = os.path.join(PROJECT_DIR, "preprocessed_data_SR_125Hz_120s_segments_gold")
    if output_dir is None:
        output_dir = os.path.join(PROJECT_DIR, "embeddings")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    if run_serial is None:
        run_serial = RUN_SERIAL
    num_workers = 0 if run_serial else n_jobs

    if save_format == "h5" and h5py is None:
        raise ImportError("save_format='h5' requires h5py. Install with: pip install h5py")

    cfg = _parse_checkpoint_config(checkpoint_path)
    transform = _build_transform(cfg["input_channels"])

    file_tasks = _list_recording_files(data_dir, split, split_dir)
    if not file_tasks:
        raise FileNotFoundError(
            f"No recording files found in {data_dir} for split {split}. "
            "Check data_dir and that split IDs match directory naming."
        )

    dataset = SegmentsPerRecordingDataset(file_tasks, cfg["input_channels"])
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_single_recording,
        pin_memory=(device.type == "cuda"),
    )

    # Build model once
    model = models_mae.sleep_1d_original(
        input_size=cfg["input_size"],
        in_chans=cfg["in_channels"],
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        decoder_embed_dim=cfg["embed_dim"],
        loss_version=cfg["loss_version"],
        loss_channels=cfg["relative_loss_channels"],
        ch_weights=None,
        norm_pix_loss=False,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    # Optional wandb
    if wandb is not None:
        wandb.init(
            project=wandb_project,
            config={
                "checkpoint": os.path.basename(checkpoint_path),
                "data_dir": data_dir,
                "split": split,
                "use_cls_token": use_cls_token,
                "pooling_method": pooling_method,
                "batch_size": batch_size,
                "save_format": save_format,
                "num_recordings": len(file_tasks),
            },
            job_type="create_embeddings",
        )

    os.makedirs(output_dir, exist_ok=True)
    if save_filename is None:
        stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
        method = "cls" if use_cls_token else pooling_method
        save_filename = f"MAE_{method}_latent_segments_{stem}_{split}"
    out_path_csv = os.path.join(output_dir, save_filename + ".csv")
    out_path_h5 = os.path.join(output_dir, save_filename + ".h5")

    all_chunks = []  # list of (rec_ids, emb_np) for CSV output
    total_segments_processed = 0
    total_recordings_processed = 0
    embed_dim = None  # set from first chunk when save_format == "h5"
    h5_file = None
    h5_current_size = 0

    if save_format == "h5":
        pass  # create after first chunk when we know embed_dim

    def process_loaded_file(recording_id, segments_tensor):
        nonlocal total_segments_processed, total_recordings_processed, h5_current_size, embed_dim, h5_file
        if segments_tensor is None or segments_tensor.shape[0] == 0:
            return
        try:
            transformed = []
            for i in range(segments_tensor.shape[0]):
                seg = transform(segments_tensor[i])
                transformed.append(seg)
            segments_tensor = torch.stack(transformed, dim=0)
        except Exception as e:
            print(f"Skipping recording {recording_id}: transform failed: {e}")
            return
        with torch.no_grad():
            for start in range(0, segments_tensor.shape[0], batch_size):
                end = min(start + batch_size, segments_tensor.shape[0])
                try:
                    batch = segments_tensor[start:end].to(device, non_blocking=True)
                    if device.type == "cuda":
                        with torch.cuda.amp.autocast():
                            _, _, _, latent_0mask, _ = model.forward_encoder(batch, mask_ratio=0.0)
                            latent_0mask = latent_0mask.float()
                    else:
                        _, _, _, latent_0mask, _ = model.forward_encoder(batch, mask_ratio=0.0)
                        latent_0mask = latent_0mask.float()
                    ids = [recording_id] * (end - start)
                    rec_ids, emb_np = _latent_to_arrays(
                        ids, latent_0mask, use_cls_token=use_cls_token, pooling_method=pooling_method
                    )
                    if rec_ids is None or emb_np is None:
                        continue
                    n = len(rec_ids)
                    if save_format == "h5":
                        chunk_embed_dim = emb_np.shape[1]
                        if h5_file is None:
                            embed_dim = chunk_embed_dim
                            h5_file = h5py.File(out_path_h5, "w")
                            h5_file.create_dataset(
                                "embeddings",
                                shape=(0, embed_dim),
                                maxshape=(None, embed_dim),
                                dtype=np.float32,
                                chunks=(min(10000, batch_size * 10), embed_dim),
                            )
                            h5_file.create_dataset(
                                "recording_ids",
                                shape=(0,),
                                maxshape=(None,),
                                dtype=h5py.special_dtype(vlen=str),
                            )
                        try:
                            h5_file["embeddings"].resize((h5_current_size + n, embed_dim))
                            h5_file["embeddings"][h5_current_size : h5_current_size + n] = emb_np
                            h5_file["recording_ids"].resize((h5_current_size + n,))
                            h5_file["recording_ids"][h5_current_size : h5_current_size + n] = rec_ids
                            h5_current_size += n
                            total_segments_processed += n
                        except (KeyError, OSError, Exception) as e:
                            print(f"Skipping {n} segments (recording {recording_id}): HDF5 write failed: {e}")
                            continue
                    else:
                        all_chunks.append((rec_ids, emb_np))
                        total_segments_processed += n
                except Exception as e:
                    print(f"Skipping batch (recording {recording_id}, segments {start}:{end}): {e}")
                    continue
        total_recordings_processed += 1
        if wandb is not None and total_recordings_processed % wandb_log_every_n_recordings == 0:
            wandb.log({
                "embeddings/recordings_processed": total_recordings_processed,
                "embeddings/segments_processed": total_segments_processed,
                "embeddings/progress_pct_recordings": 100.0 * total_recordings_processed / len(file_tasks),
            })

    try:
        for batch in tqdm(dataloader, desc="Creating embeddings", total=len(dataloader)):
            try:
                segments_tensor, recording_id = batch
            except Exception as e:
                print(f"Skipping batch due to load error: {e}")
                continue
            process_loaded_file(recording_id, segments_tensor)
    except Exception as e:
        if h5_file is not None:
            h5_file.close()
        raise
    if h5_file is not None:
        h5_file.close()

    if save_format == "h5":
        if total_segments_processed > 0:
            print(f"Saved segment-level embeddings ({total_segments_processed} segments) to {out_path_h5}")
        else:
            print("No segments processed; no HDF5 file written.")
        if wandb is not None:
            wandb.log({
                "embeddings/total_segments": total_segments_processed,
                "embeddings/total_recordings": total_recordings_processed,
                "embeddings/output_path": out_path_h5 if total_segments_processed > 0 else None,
            })
            wandb.finish()
        return None
    else:
        if not all_chunks:
            raise RuntimeError("No embeddings produced; check data and errors above.")
        embed_dim_csv = all_chunks[0][1].shape[1]
        with open(out_path_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Recordings"] + [str(i) for i in range(embed_dim_csv)])
            for rec_ids, emb_np in all_chunks:
                for rec_id, row in zip(rec_ids, emb_np):
                    writer.writerow([rec_id] + row.tolist())
        total_rows = sum(len(ids) for ids, _ in all_chunks)
        print(f"Saved segment-level embeddings ({total_rows} rows) to {out_path_csv}")
        if wandb is not None:
            wandb.log({
                "embeddings/total_segments": total_rows,
                "embeddings/total_recordings": total_recordings_processed,
                "embeddings/output_path": out_path_csv,
            })
            wandb.finish()
        recording_ids_flat = [rec_id for rec_ids, _ in all_chunks for rec_id in rec_ids]
        embeddings_stack = np.vstack([arr for _, arr in all_chunks])
        return (recording_ids_flat, embeddings_stack)


if __name__ == "__main__":
    if CHECKPOINT_PATH is None:
        raise ValueError("Set CHECKPOINT_PATH at the top of this file before running.")

    create_embeddings_from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        data_dir=DATA_DIR,
        split=SPLIT,
        split_dir=SPLIT_DIR,
        use_cls_token=USE_CLS_TOKEN,
        pooling_method=POOLING_METHOD,
        output_dir=OUTPUT_DIR,
        save_filename=SAVE_FILENAME,
        save_format=SAVE_FORMAT,
        n_jobs=N_JOBS,
        run_serial=RUN_SERIAL,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        wandb_project=WANDB_PROJECT,
        wandb_log_every_n_recordings=WANDB_LOG_EVERY_N_RECORDINGS,
    )

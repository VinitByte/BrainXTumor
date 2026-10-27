"""
Brain tumor detection from in-vivo hyperspectral images (HELICoiD database)
using a HybridSN-style CNN (3D conv for spectra + 2D conv for spatial context).

Classes: NT (normal tissue), TT (tumor tissue), BV (blood vessel), BG (background)

Usage:
    python brain_tumor_hsi_cnn.py --dataset-root "D:/ImageXTumor/Dataset" \
                                  --work-dir "D:/ImageXTumor/Process Data" \
                                  --results-dir results

Outputs (in --results-dir):
    metrics.json, classification_report.txt, confusion_matrix.png,
    training_curves.png, preprocessing_example.png, predictions/<patient>.png
"""
import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")  # save figures to disk, no display needed
import matplotlib.pyplot as plt
import numpy as np
import spectral as spy
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

spy.settings.envi_support_nonlowercase_params = True
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RAW_FILE, WHITE_REF_FILE, DARK_REF_FILE, GT_FILE = "raw", "whiteReference", "darkReference", "gtMap"

# Ground-truth labels in HELICoiD: 0 = unlabeled, 1 = NT, 2 = TT, 3 = BV, 4 = BG
LABEL_NAMES = {1: "NT", 2: "TT", 3: "BV", 4: "BG"}
IGNORE_LABELS = (0,)
USED_LABELS = [l for l in LABEL_NAMES if l not in IGNORE_LABELS]
LABEL_TO_IDX = {l: i for i, l in enumerate(USED_LABELS)}
CLASS_NAMES = [LABEL_NAMES[l] for l in USED_LABELS]
CLASS_COLORS = {"NT": "#2ca02c", "TT": "#d62728", "BV": "#1f77b4", "BG": "#000000"}  # as in the paper

PATCH_SIZE = 15
TARGET_BANDS = 128
BATCH_SIZE = 64
EPOCHS = 10
SAMPLES_PER_EPOCH = 20000  # class-balanced samples drawn per epoch
LR = 1e-3
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", required=True, help="Folder with one sub-folder per HELICoiD image")
    p.add_argument("--work-dir", default="processed_data", help="Where preprocessed .npy files + best model go")
    p.add_argument("--results-dir", default="results", help="Where metrics and figures are saved")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--num-workers", type=int, default=0, help="Keep 0 on Windows")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Loading + preprocessing
# --------------------------------------------------------------------------- #
def get_patient_folders(dataset_root):
    folders = []
    for name in sorted(os.listdir(dataset_root)):
        full = os.path.join(dataset_root, name)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, RAW_FILE + ".hdr")):
            folders.append(full)
    return folders


def load_patient_data(folder):
    def _load(name):
        return np.array(spy.open_image(os.path.join(folder, name + ".hdr")).load(), dtype=np.float32)

    gt = None
    gt_hdr = os.path.join(folder, GT_FILE + ".hdr")
    if os.path.exists(gt_hdr):
        gt = np.squeeze(np.array(spy.open_image(gt_hdr).load()))
        gt = np.nan_to_num(gt, nan=0).round().astype(np.uint8)  # float -> integer labels
    return {"raw": _load(RAW_FILE), "white_ref": _load(WHITE_REF_FILE),
            "dark_ref": _load(DARK_REF_FILE), "gt": gt, "patient_id": os.path.basename(folder)}


def calibrate(raw, white_ref, dark_ref):
    """CI = (RI - DI) / (WI - DI)"""
    white = white_ref.mean(axis=(0, 1), keepdims=True)
    dark = dark_ref.mean(axis=(0, 1), keepdims=True)
    return np.clip((raw - dark) / (white - dark + 1e-6), 0, None)


def remove_noisy_edge_bands(cube, first_n=56, last_n=126):
    return cube[:, :, first_n: cube.shape[2] - last_n]  # 826 -> 645 bands


def smooth_spectra(cube, window=5):
    return uniform_filter1d(cube, size=window, axis=2, mode="nearest")


def decimate_bands(cube, target_bands=TARGET_BANDS):
    idx = np.linspace(0, cube.shape[2] - 1, target_bands).astype(int)
    return cube[:, :, idx]


def normalize_spectra(cube):
    c_min = cube.min(axis=2, keepdims=True)
    c_max = cube.max(axis=2, keepdims=True)
    return (cube - c_min) / (c_max - c_min + 1e-6)


def full_pipeline(raw, white_ref, dark_ref):
    cube = calibrate(raw, white_ref, dark_ref)
    cube = remove_noisy_edge_bands(cube)
    cube = smooth_spectra(cube)
    cube = decimate_bands(cube)
    return normalize_spectra(cube)


def process_all_patients(folders, work_dir, results_dir, overwrite=False):
    ok, failed = 0, []
    saved_example = False
    for i, folder in enumerate(folders, 1):
        pid = os.path.basename(folder)
        cube_path = os.path.join(work_dir, f"{pid}_cube.npy")
        gt_path = os.path.join(work_dir, f"{pid}_gt.npy")
        print(f"[{i}/{len(folders)}] {pid} ...", end=" ")
        if not overwrite and os.path.exists(cube_path) and os.path.exists(gt_path):
            print("already done"); ok += 1; continue
        try:
            d = load_patient_data(folder)
            if d["gt"] is None:
                print("no gtMap, skipped"); continue
            cube = full_pipeline(d["raw"], d["white_ref"], d["dark_ref"])
            np.save(cube_path, cube.astype(np.float32))
            np.save(gt_path, d["gt"])
            if not saved_example:
                save_preprocessing_example(d["raw"], cube, results_dir, pid)
                saved_example = True
            print("done", cube.shape); ok += 1
        except Exception as e:  # keep going if one image is broken
            print("FAILED --", e); failed.append((pid, str(e)))
    print(f"\nProcessed {ok}/{len(folders)}")
    if failed:
        print("Failed:", failed)


def save_preprocessing_example(raw, processed, results_dir, pid):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(raw[:, :, raw.shape[2] // 2], cmap="gray"); ax[0].set_title(f"Raw (one band) - {pid}")
    ax[1].imshow(processed[:, :, processed.shape[2] // 2], cmap="gray"); ax[1].set_title("Preprocessed")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "preprocessing_example.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Dataset (memory-mapped, patches cut on the fly)
# --------------------------------------------------------------------------- #
class HSIPatchDataset(Dataset):
    def __init__(self, patient_ids, processed_dir, patch_size=PATCH_SIZE, ignore_labels=IGNORE_LABELS):
        self.pids = list(patient_ids)
        self.dir = processed_dir
        self.half = patch_size // 2
        self.cubes, self.shapes = {}, {}
        lut = np.full(256, -1, dtype=np.int64)  # raw label -> class index
        for raw_l, idx in LABEL_TO_IDX.items():
            lut[raw_l] = idx

        pi, ys, xs, labs = [], [], [], []
        for k, pid in enumerate(self.pids):
            cube_mm = np.load(os.path.join(self.dir, f"{pid}_cube.npy"), mmap_mode="r")
            self.shapes[pid] = cube_mm.shape
            gt = np.squeeze(np.load(os.path.join(self.dir, f"{pid}_gt.npy"))).astype(np.int64)
            y, x = np.nonzero(~np.isin(gt, ignore_labels))
            pi.append(np.full(len(y), k, dtype=np.int32))
            ys.append(y.astype(np.int32)); xs.append(x.astype(np.int32))
            labs.append(lut[gt[y, x]])
        self.pi = np.concatenate(pi); self.ys = np.concatenate(ys)
        self.xs = np.concatenate(xs); self.labels = np.concatenate(labs)

    def _cube(self, pid):
        if pid not in self.cubes:
            self.cubes[pid] = np.load(os.path.join(self.dir, f"{pid}_cube.npy"), mmap_mode="r")
        return self.cubes[pid]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        pid = self.pids[self.pi[i]]
        y, x, h = int(self.ys[i]), int(self.xs[i]), self.half
        H, W, _ = self.shapes[pid]
        y0, y1, x0, x1 = y - h, y + h + 1, x - h, x + h + 1
        pad = ((max(0, -y0), max(0, y1 - H)), (max(0, -x0), max(0, x1 - W)), (0, 0))
        patch = np.array(self._cube(pid)[max(0, y0):min(H, y1), max(0, x0):min(W, x1), :], dtype=np.float32)
        if any(p for pr in pad for p in pr):
            patch = np.pad(patch, pad, mode="reflect")
        patch = np.ascontiguousarray(patch.transpose(2, 0, 1))  # (bands, H, W)
        return torch.from_numpy(patch), int(self.labels[i])


# --------------------------------------------------------------------------- #
# Model: small HybridSN-style (3D conv for spectra, 2D conv for spatial)
# --------------------------------------------------------------------------- #
class HybridSNLite(nn.Module):
    def __init__(self, n_bands=TARGET_BANDS, patch=PATCH_SIZE, n_classes=len(CLASS_NAMES)):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 8, (7, 3, 3)), nn.BatchNorm3d(8), nn.ReLU(),
            nn.Conv3d(8, 16, (5, 3, 3)), nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16, 32, (3, 3, 3)), nn.BatchNorm3d(32), nn.ReLU(),
        )
        with torch.no_grad():
            c, d, h, w = self.conv3d(torch.zeros(1, 1, n_bands, patch, patch)).shape[1:]
        self.conv2d = nn.Sequential(nn.Conv2d(c * d, 64, 3), nn.BatchNorm2d(64), nn.ReLU())
        flat = 64 * (h - 2) * (w - 2)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(flat, 128), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(128, n_classes))

    def forward(self, x):  # x: (B, bands, H, W)
        x = self.conv3d(x.unsqueeze(1))  # (B, C, D, H, W)
        x = x.flatten(1, 2)              # (B, C*D, H, W)
        return self.head(self.conv2d(x))


# --------------------------------------------------------------------------- #
# Train / evaluate helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(xb)
        preds.append(out.argmax(1).cpu().numpy())
        targets.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def save_confusion_matrix(y_true, y_pred, path):
    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(labels); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(labels); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (row-normalised)")
    for i in labels:
        for j in labels:
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


def save_training_curves(history, path):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], marker="o")
    ax[0].set_title("Training loss"); ax[0].set_xlabel("Epoch")
    ax[1].plot(ep, [h["val_oa"] for h in history], marker="o", label="Val OA")
    ax[1].plot(ep, [h["val_kappa"] for h in history], marker="o", label="Val kappa")
    ax[1].set_title("Validation"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


@torch.no_grad()
def predict_full_image(model, pid, processed_dir, device, num_workers):
    ds = HSIPatchDataset([pid], processed_dir, ignore_labels=())  # every pixel
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)
    model.eval()
    preds = []
    for xb, _ in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(xb.to(device, non_blocking=True))
        preds.append(out.argmax(1).cpu().numpy())
    H, W, _ = ds.shapes[pid]
    return np.concatenate(preds).reshape(H, W)


def save_prediction_figure(model, pid, processed_dir, device, num_workers, out_path):
    gt = np.squeeze(np.load(os.path.join(processed_dir, f"{pid}_gt.npy")))
    pred = predict_full_image(model, pid, processed_dir, device, num_workers)
    gt_idx = np.full(gt.shape, -1)
    for raw_l, idx in LABEL_TO_IDX.items():
        gt_idx[gt == raw_l] = idx
    pred_vis = np.where(gt_idx >= 0, pred, -1)  # show only labelled pixels

    cmap = ListedColormap([CLASS_COLORS[c] for c in CLASS_NAMES]).copy()
    cmap.set_under("white")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(gt_idx, cmap=cmap, vmin=0, vmax=len(CLASS_NAMES) - 1, interpolation="nearest")
    ax[0].set_title(f"Ground truth {pid}")
    ax[1].imshow(pred_vis, cmap=cmap, vmin=0, vmax=len(CLASS_NAMES) - 1, interpolation="nearest")
    ax[1].set_title("Prediction")
    for a in ax:
        a.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[c]) for c in CLASS_NAMES]
    fig.legend(handles, CLASS_NAMES, loc="lower center", ncol=len(CLASS_NAMES))
    plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.savefig(out_path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "predictions"), exist_ok=True)

    np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: CUDA GPU not found - training on CPU will be very slow.")
    else:
        torch.backends.cudnn.benchmark = True
    print("Device:", device, "| torch", torch.__version__)

    # 1. Preprocess
    folders = get_patient_folders(args.dataset_root)
    print(f"{len(folders)} image folders found")
    process_all_patients(folders, args.work_dir, args.results_dir)

    # 2. Patient-level split (group = patient number, so 050-01 and 050-02 stay together)
    all_pids = [os.path.basename(f) for f in folders
                if os.path.exists(os.path.join(args.work_dir, f"{os.path.basename(f)}_gt.npy"))]
    groups = [p.split("-")[0] for p in all_pids]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, va_idx = next(gss.split(all_pids, groups=groups))
    train_pids = [all_pids[i] for i in tr_idx]
    val_pids = [all_pids[i] for i in va_idx]
    print(f"Train images: {len(train_pids)} | Val images: {len(val_pids)}")

    train_ds = HSIPatchDataset(train_pids, args.work_dir)
    val_ds = HSIPatchDataset(val_pids, args.work_dir)
    counts = lambda ds: {CLASS_NAMES[i]: int((ds.labels == i).sum()) for i in range(len(CLASS_NAMES))}
    print("Train class counts:", counts(train_ds))
    print("Val class counts:  ", counts(val_ds))

    # Balanced sampling for training (handles rare tumor class)
    cnt = np.bincount(train_ds.labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    sample_w = 1.0 / np.maximum(cnt[train_ds.labels], 1)
    sampler = WeightedRandomSampler(sample_w, num_samples=SAMPLES_PER_EPOCH, replacement=True)
    dl_kw = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, **dl_kw)

    # 3. Model + training
    model = HybridSNLite().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("Params:", n_params)

    criterion = nn.CrossEntropyLoss()  # sampler already balances classes
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_kappa, best_path = -1.0, os.path.join(args.work_dir, "best_model.pt")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(yb); n += len(yb)

        yp, yt = predict(model, val_loader, device)
        oa, kappa = accuracy_score(yt, yp), cohen_kappa_score(yt, yp)
        history.append({"epoch": epoch, "train_loss": total / n, "val_oa": oa, "val_kappa": kappa})
        print(f"Epoch {epoch:02d} | train loss {total / n:.4f} | val OA {oa:.4f} | val kappa {kappa:.4f}")
        if kappa > best_kappa:
            best_kappa = kappa
            torch.save(model.state_dict(), best_path)

    # 4. Final evaluation with the best checkpoint
    model.load_state_dict(torch.load(best_path, map_location=device))
    yp, yt = predict(model, val_loader, device)
    labels = list(range(len(CLASS_NAMES)))
    report_txt = classification_report(yt, yp, labels=labels, target_names=CLASS_NAMES, zero_division=0)
    report_dict = classification_report(yt, yp, labels=labels, target_names=CLASS_NAMES,
                                        zero_division=0, output_dict=True)
    oa, kappa = accuracy_score(yt, yp), cohen_kappa_score(yt, yp)
    print(f"\n=== Best model on validation ===\nOA: {oa:.4f} | Kappa: {kappa:.4f}\n{report_txt}")

    with open(os.path.join(args.results_dir, "classification_report.txt"), "w") as f:
        f.write(f"OA: {oa:.4f} | Kappa: {kappa:.4f}\n\n{report_txt}\n")
        f.write("Confusion matrix (rows=true, cols=pred):\n")
        f.write(str(confusion_matrix(yt, yp, labels=labels)) + "\n")
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump({"overall_accuracy": oa, "cohen_kappa": kappa, "n_params": n_params,
                   "train_images": train_pids, "val_images": val_pids,
                   "per_class": report_dict, "history": history}, f, indent=2)
    save_confusion_matrix(yt, yp, os.path.join(args.results_dir, "confusion_matrix.png"))
    save_training_curves(history, os.path.join(args.results_dir, "training_curves.png"))

    # 5. Full-image predictions for every validation image
    for pid in val_pids:
        save_prediction_figure(model, pid, args.work_dir, device, args.num_workers,
                               os.path.join(args.results_dir, "predictions", f"{pid}.png"))
    print(f"\nAll results saved in: {os.path.abspath(args.results_dir)}")


if __name__ == "__main__":
    main()

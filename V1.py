# train_resnet.py

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn import metrics as skm
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm


# ----------------------
# Argparse
# ----------------------
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ResNet50 on mid-sagittal PNGs (ASD vs NON-ASD)."
    )
    p.add_argument("--data-root", required=True, type=str, help="Root folder with train/val/test splits.")
    p.add_argument("--output-dir", required=True, type=str, help="Output directory for checkpoints and reports.")
    p.add_argument("--epochs", default=50, type=int)
    p.add_argument("--batch-size", default=32, type=int)
    p.add_argument("--lr", default=3e-4, type=float)
    p.add_argument("--weight-decay", default=1e-4, type=float)
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision (autocast + GradScaler)")
    p.add_argument("--patience", default=10, type=int, help="Early stop patience (epochs) on val ROC-AUC.")
    p.add_argument("--use-imagenet-norm", action="store_true",
                   help="Use ImageNet mean/std even if norm_stats.json exists.")
    p.add_argument("--hflip", action="store_true", help="Enable RandomHorizontalFlip(p=0.5) for training.")
    p.add_argument("--freeze-until", default="none", choices=["none", "layer1", "layer2", "layer3"],
                   help="Freeze early blocks up to (and including) the given stage.")
    p.add_argument("--onecycle", action="store_true", help="Use OneCycleLR rather than warmup+cosine.")
    return p.parse_args()


# ----------------------
# Logging
# ----------------------
def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    logger = logging.getLogger("train_resnet")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(f"Logging to {log_path}")
    return logger


# ----------------------
# Repro
# ----------------------
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    # Different seed per worker, but deterministic given base seed
    s = torch.initial_seed() % 2**32
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)


# ----------------------
# JSON helpers
# ----------------------
def try_load_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def resolve_norm_stats(data_root: Path, use_imagenet: bool, logger: logging.Logger) -> Tuple[List[float], List[float]]:
    # Try <data_root>/norm_stats.json
    stats = None
    if not use_imagenet:
        stats = try_load_json(data_root / "norm_stats.json")
        if stats is None:
            # Also try parent or qc sibling
            stats = try_load_json(data_root.parent / "norm_stats.json") or try_load_json(
                data_root / "qc" / "norm_stats.json"
            )
    if stats and "mean_rgb_0_1" in stats and "std_rgb_0_1" in stats:
        mean = [float(x) for x in stats["mean_rgb_0_1"]]
        std = [float(x) for x in stats["std_rgb_0_1"]]
        logger.info(f"Using dataset normalization from norm_stats.json: mean={mean}, std={std}")
        return mean, std

    # Fallback to ImageNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if use_imagenet:
        logger.info("Using ImageNet normalization (forced by --use-imagenet-norm).")
    else:
        logger.warning("norm_stats.json not found or invalid; falling back to ImageNet mean/std.")
    return mean, std


def resolve_class_weights(data_root: Path, train_set: datasets.ImageFolder, logger: logging.Logger) -> torch.Tensor:
    # Try to load JSON weights first
    wjson = try_load_json(data_root / "class_weights.json")
    if wjson is None:
        wjson = try_load_json(data_root.parent / "class_weights.json") or try_load_json(
            data_root / "qc" / "class_weights.json"
        )

    cls_to_idx = train_set.class_to_idx
    num_classes = len(cls_to_idx)
    weights = torch.ones(num_classes, dtype=torch.float32)

    if wjson and "class_weights" in wjson:
        cw = wjson["class_weights"]
        try:
            for cls, idx in cls_to_idx.items():
                weights[idx] = float(cw.get(cls, 1.0))
            logger.info(f"Loaded class weights from class_weights.json: {dict(zip(train_set.classes, weights.tolist()))}")
            return weights
        except Exception:
            logger.warning("class_weights.json present but invalid; will compute from training set.")

    # Compute inverse-frequency weights from train_set
    labels = np.array(train_set.targets, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = (counts.sum() / (len(counts) * counts)).astype(np.float32)
    weights = torch.from_numpy(inv)
    logger.info(f"Computed inverse-frequency class weights from training set: "
                f"{dict(zip(train_set.classes, weights.tolist()))}")
    return weights


def make_sampler(train_set: datasets.ImageFolder) -> WeightedRandomSampler:
    labels = np.array(train_set.targets, dtype=np.int64)
    counts = np.bincount(labels, minlength=len(train_set.classes)).astype(np.float64)
    counts[counts == 0] = 1.0
    sample_weights = 1.0 / counts[labels]
    sample_weights = torch.from_numpy(sample_weights).float()
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)
    return sampler


# ----------------------
# Data
# ----------------------
def build_transforms(mean: List[float], std: List[float], use_hflip: bool) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = [
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=3, fill=0),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=0),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ]
    if use_hflip:
        train_tfms.append(transforms.RandomHorizontalFlip(p=0.5))
    train_tfms.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transforms.Compose(train_tfms), eval_tfms


def build_dataloaders(
    data_root: Path,
    mean: List[float],
    std: List[float],
    batch_size: int,
    num_workers: int,
    use_hflip: bool,
    logger: logging.Logger,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], datasets.ImageFolder, datasets.ImageFolder, Optional[datasets.ImageFolder]]:
    tf_train, tf_eval = build_transforms(mean, std, use_hflip)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    train_set = datasets.ImageFolder(str(train_dir), transform=tf_train)
    val_set = datasets.ImageFolder(str(val_dir), transform=tf_eval)
    test_set = datasets.ImageFolder(str(test_dir), transform=tf_eval) if test_dir.exists() else None

    logger.info(f"Classes: {train_set.classes}")
    # Print split sizes by class
    def split_counts(ds: datasets.ImageFolder, name: str):
        arr = np.array(ds.targets, dtype=np.int64)
        counts = {ds.classes[i]: int((arr == i).sum()) for i in range(len(ds.classes))}
        logger.info(f"{name} counts: {counts} | total={len(ds)}")

    split_counts(train_set, "train")
    split_counts(val_set, "val")
    if test_set is not None:
        split_counts(test_set, "test")
    else:
        logger.warning("test/ not found; final evaluation will be skipped.")

    sampler = make_sampler(train_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    return train_loader, val_loader, test_loader, train_set, val_set, test_set


# ----------------------
# Model
# ----------------------
def build_model(num_classes: int, freeze_until: str, logger: logging.Logger) -> nn.Module:
    weights = None
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
    except Exception:
        pass

    try:
        model = models.resnet50(weights=weights)
        if weights is None:
            logger.warning("Pretrained weights could not be resolved; using random init.")
    except Exception as e:
        logger.warning(f"Could not load pretrained weights ({e}); using random init.")
        model = models.resnet50(weights=None)

    # Replace fc
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Freeze early blocks if requested
    def set_requires_grad(module: nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    if freeze_until != "none":
        logger.info(f"Freezing layers up to: {freeze_until}")
        set_requires_grad(model.conv1, False)
        set_requires_grad(model.bn1, False)
        set_requires_grad(model.layer1, False)
        if freeze_until in ("layer2", "layer3"):
            set_requires_grad(model.layer2, False)
        if freeze_until == "layer3":
            set_requires_grad(model.layer3, False)
    else:
        logger.info("Training all layers (no freezing).")

    return model


# ----------------------
# Schedulers
# ----------------------
def build_scheduler(optimizer: torch.optim.Optimizer, epochs: int, onecycle: bool,
                    steps_per_epoch: int, base_lr: float, logger: logging.Logger):
    if onecycle:
        logger.info("Using OneCycleLR scheduler.")
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=base_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=5 / max(epochs, 1),
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=1e4,
        )
        step_on_batch = True
        return sched, step_on_batch

    # Warmup + cosine via LambdaLR
    warmup_epochs = min(5, max(1, epochs // 10))  # default: 5 epochs
    logger.info(f"Using linear warmup ({warmup_epochs} epochs) + cosine anneal.")

    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)
        else:
            t = (current_epoch - warmup_epochs) / max(1, (epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    step_on_batch = False
    return sched, step_on_batch


# ----------------------
# Train / Eval helpers
# ----------------------
@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    all_logits = []
    all_targets = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += ce(logits, y).item()
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())
    if len(all_logits) == 0:
        return float("nan"), float("nan"), float("nan")
    logits = torch.cat(all_logits, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    loss = total_loss / max(1, len(loader.dataset))
    preds = probs.argmax(axis=1)
    acc = (preds == targets).mean()

    auc = float("nan")
    try:
        if probs.shape[1] == 2 and len(np.unique(targets)) == 2:
            auc = skm.roc_auc_score(targets, probs[:, 1])
    except Exception:
        pass
    return loss, acc, auc


def train(
    args: argparse.Namespace,
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    out_dir: Path,
    logger: logging.Logger,
) -> Tuple[nn.Module, List[dict]]:
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    scheduler, step_on_batch = build_scheduler(optimizer, args.epochs, args.onecycle,
                                               steps_per_epoch=len(train_loader),
                                               base_lr=args.lr, logger=logger)

    metrics_rows: List[dict] = []
    best_auc = -1.0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    epochs_no_improve = 0

    logger.info("----- Training begins -----")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        n_seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(xb)
                loss = ce(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step_on_batch and scheduler is not None:
                scheduler.step()

            running_loss += loss.detach().item() * xb.size(0)
            preds = logits.detach().argmax(dim=1)
            running_correct += (preds == yb).sum().item()
            n_seen += xb.size(0)

            lr_cur = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_cur:.2e}")

        if not step_on_batch and scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(1, n_seen)
        train_acc = running_correct / max(1, n_seen)

        val_loss, val_acc, val_auc = eval_epoch(model, val_loader, device)
        logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f} "
                    f"train_acc={train_acc:.4f} | val_loss={val_loss:.4f} "
                    f"val_acc={val_acc:.4f} val_auc={val_auc:.4f}")

        metrics_rows.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Early stopping on val AUC (fallback to acc if NaN)
        score = val_auc if not (np.isnan(val_auc) or val_auc is None) else val_acc
        best_cmp = best_auc if not (np.isnan(best_auc) or best_auc is None) else -1.0
        if score > best_cmp:
            epochs_no_improve = 0
            best_auc = val_auc
            torch.save(model.state_dict(), best_path)
            logger.info(f"  ↳ Saved new BEST to {best_path} (val_auc={val_auc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(f"Early stopping triggered (no improvement for {args.patience} epochs).")
                break

    # Save last
    torch.save(model.state_dict(), last_path)
    logger.info(f"Saved LAST checkpoint to {last_path}")
    return model, metrics_rows


# ----------------------
# Test-time evaluation
# ----------------------
@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_targets = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(yb.cpu().numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)


def youden_j_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, thr = skm.roc_curve(y_true, y_score)
    j = tpr - fpr
    best = int(np.argmax(j))
    return float(thr[best])


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 1000, seed: int = 12345) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, aucs, f1s = [], [], []
    for _ in tqdm(range(n_boot), desc="Bootstrap", unit="res" \
    "le", leave=False):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        pred = (yp >= 0.5).astype(int)
        accs.append(skm.accuracy_score(yt, pred))
        try:
            aucs.append(skm.roc_auc_score(yt, yp))
        except Exception:
            pass
        f1s.append(skm.f1_score(yt, pred))
    def q(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"acc": q(accs), "auc": q(aucs) if len(aucs) > 0 else [float("nan"), float("nan")], "f1": q(f1s)}


def plot_curves(metrics_rows: List[dict], out_path: Path) -> None:
    epochs = [r["epoch"] for r in metrics_rows]
    tr_loss = [r["train_loss"] for r in metrics_rows]
    va_loss = [r["val_loss"] for r in metrics_rows]
    va_acc = [r["val_acc"] for r in metrics_rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, tr_loss, label="train_loss")
    ax[0].plot(epochs, va_loss, label="val_loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].plot(epochs, va_acc, label="val_acc")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].legend(); ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = skm.roc_curve(y_true, y_prob)
    auc = skm.roc_auc_score(y_true, y_prob)
    ax[0].plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="gray")
    ax[0].set_title("ROC"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend()

    prec, rec, _ = skm.precision_recall_curve(y_true, y_prob)
    ap = skm.average_precision_score(y_true, y_prob)
    ax[1].plot(rec, prec, label=f"AP={ap:.3f}")
    ax[1].set_title("Precision-Recall"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], out_path: Path) -> None:
    cm = skm.confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for j, (mat, title) in enumerate([(cm, "Confusion (raw)"), (cm_norm, "Confusion (normalized)")]):
        a = ax[j].imshow(mat, cmap="Blues")
        for (i, k), v in np.ndenumerate(mat):
            ax[j].text(k, i, f"{v:.2f}" if title.endswith("normalized") else str(v),
                       ha="center", va="center", color="black", fontsize=10)
        ax[j].set_xticks([0, 1]); ax[j].set_yticks([0, 1])
        ax[j].set_xticklabels(classes); ax[j].set_yticklabels(classes)
        ax[j].set_xlabel("Pred"); ax[j].set_ylabel("True"); ax[j].set_title(title)
        fig.colorbar(a, ax=ax[j], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ----------------------
# Grad-CAM
# ----------------------
class GradCAM:
    """
    Minimal Grad-CAM for ResNet layer4[-1].conv3
    """
    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self.layer = layer
        self.activations = None
        self.gradients = None
        self.hook_a = layer.register_forward_hook(self._forward_hook)
        self.hook_g = layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        # Global average pool gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # BxCx1x1
        cam = (weights * self.activations).sum(dim=1, keepdim=False)  # BxHxW
        cam = cam.relu()
        cam = cam[0].cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def close(self):
        self.hook_a.remove()
        self.hook_g.remove()


def denormalize(img: torch.Tensor, mean: List[float], std: List[float]) -> np.ndarray:
    # img: 3xHxW tensor
    mean = torch.tensor(mean, device=img.device)[:, None, None]
    std = torch.tensor(std, device=img.device)[:, None, None]
    x = img * std + mean
    x = x.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return x


def save_gradcam_samples(model: nn.Module,
                         loader: DataLoader,
                         device: torch.device,
                         mean: List[float],
                         std: List[float],
                         out_dir: Path,
                         classes: List[str],
                         logger: logging.Logger,
                         max_samples: int = 8) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Hook last conv in layer4
    try:
        target_layer = model.layer4[-1].conv3
    except Exception:
        logger.warning("Could not access layer4[-1].conv3; falling back to layer4.")
        target_layer = model.layer4
    cam = GradCAM(model, target_layer)
    model.eval()

    saved = 0
    with torch.no_grad():
        # First pass: gather predictions to select correct/incorrect
        xs, ys, probs, preds = [], [], [], []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pr = torch.softmax(logits, dim=1)[:, 1]
            pb = pr.detach().cpu().numpy()
            pd = logits.argmax(dim=1).detach().cpu().numpy()
            xs.append(xb.cpu()); ys.append(yb.cpu().numpy()); probs.append(pb); preds.append(pd)
        x_all = torch.cat(xs, dim=0)
        y_all = np.concatenate(ys, axis=0)
        p_all = np.concatenate(probs, axis=0)
        pred_all = np.concatenate(preds, axis=0)

    # Choose up to half correct and half incorrect
    correct_idx = np.where(pred_all == y_all)[0]
    incorrect_idx = np.where(pred_all != y_all)[0]
    rng = np.random.default_rng(1234)
    take_c = min(len(correct_idx), max_samples // 2)
    take_i = min(len(incorrect_idx), max_samples - take_c)
    chosen = []
    if take_c > 0:
        chosen.extend(rng.choice(correct_idx, size=take_c, replace=False).tolist())
    if take_i > 0:
        chosen.extend(rng.choice(incorrect_idx, size=take_i, replace=False).tolist())
    if len(chosen) == 0:
        chosen = list(range(min(max_samples, len(x_all))))

    for idx in chosen:
        x = x_all[idx:idx + 1].to(device)
        true = int(y_all[idx])
        prob1 = float(p_all[idx])
        pred = int(pred_all[idx])

        _ = model(x)  # forward to populate hooks
        heat = cam(x)  # HxW in [0,1]
        img = denormalize(x[0], mean, std)  # HxWx3 in [0,1]

        # Resize heatmap to image size
        H, W, _ = img.shape
        heat_img = torch.tensor(heat).unsqueeze(0).unsqueeze(0)
        heat_img = torch.nn.functional.interpolate(heat_img, size=(H, W), mode="bilinear", align_corners=False)
        heat_resized = heat_img[0, 0].cpu().numpy()
        # Overlay in matplotlib
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img)
        ax.imshow(heat_resized, cmap="jet", alpha=0.35, vmin=0.0, vmax=1.0)
        ax.axis("off")
        fname = f"idx{idx:04d}_true-{classes[true]}_pred-{classes[pred]}_p1-{prob1:.3f}.png"
        fig.savefig(out_dir / fname, bbox_inches="tight", pad_inches=0, dpi=160)
        plt.close(fig)
        saved += 1
        if saved >= max_samples:
            break

    cam.close()
    logger.info(f"Saved {saved} Grad-CAM overlays to {out_dir}")


# ----------------------
# Main
# ----------------------
def main():
    args = get_args()
    out_dir = Path(args.output_dir)
    logger = setup_logger(out_dir)
    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | torch {torch.__version__} | torchvision {torchvision_version_str()}")

    data_root = Path(args.data_root)
    mean, std = resolve_norm_stats(data_root, args.use_imagenet_norm, logger)

    # Build data
    train_loader, val_loader, test_loader, train_set, val_set, test_set = build_dataloaders(
        data_root, mean, std, args.batch_size, args.num_workers, args.hflip, logger
    )

    # Class weights for loss
    class_weights = resolve_class_weights(data_root, train_set, logger)

    # Build model
    model = build_model(num_classes=len(train_set.classes), freeze_until=args.freeze_until, logger=logger).to(device)

    # Save config
    cfg = {
        "args": vars(args),
        "classes": train_set.classes,
        "mean": mean,
        "std": std,
        "class_weights": dict(zip(train_set.classes, [float(w) for w in class_weights.tolist()])),
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision_version_str(),
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn": cudnn.version(),
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Train
    model, metrics_rows = train(args, model, device, train_loader, val_loader, class_weights, out_dir, logger)

    # Save metrics.csv and curves
    import csv as _csv
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        for r in metrics_rows:
            writer.writerow(r)
    plot_curves(metrics_rows, out_dir / "curves.png")

    # Load best checkpoint for evaluation
    best_ckpt = out_dir / "best.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        logger.info(f"Loaded best checkpoint from {best_ckpt} for evaluation.")
    else:
        logger.warning("best.pt not found; evaluating last.pt")
        last_ckpt = out_dir / "last.pt"
        if last_ckpt.exists():
            model.load_state_dict(torch.load(last_ckpt, map_location=device))

    # Expected performance note
    logger.info("Note: with clean splits, expect roughly 93–96% accuracy for a strong baseline.")

    # Evaluate on test
    if test_loader is not None:
        probs, y_true = predict_proba(model, test_loader, device)
        y_prob = probs[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc = skm.accuracy_score(y_true, y_pred)
        auc = skm.roc_auc_score(y_true, y_prob)
        f1 = skm.f1_score(y_true, y_pred)
        prec = skm.precision_score(y_true, y_pred)
        rec = skm.recall_score(y_true, y_pred)

        # Youden-J optimal threshold
        thr_opt = youden_j_threshold(y_true, y_prob)
        y_pred_opt = (y_prob >= thr_opt).astype(int)
        acc_opt = skm.accuracy_score(y_true, y_pred_opt)
        f1_opt = skm.f1_score(y_true, y_pred_opt)
        prec_opt = skm.precision_score(y_true, y_pred_opt)
        rec_opt = skm.recall_score(y_true, y_pred_opt)

        # Bootstrap CIs
        cis = bootstrap_ci(y_true, y_prob, n_boot=1000, seed=args.seed)

        # Artifacts
        plot_roc_pr(y_true, y_prob, out_dir / "roc_pr_curves.png")
        plot_confusion(y_true, y_pred, train_set.classes, out_dir / "confusion_matrix.png")

        with (out_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
            f.write(skm.classification_report(y_true, y_pred, target_names=train_set.classes))

        test_payload = {
            "point_estimates": {
                "threshold_default": 0.5,
                "accuracy": float(acc),
                "roc_auc": float(auc),
                "f1": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "threshold_youdenj": float(thr_opt),
                "accuracy_youdenj": float(acc_opt),
                "f1_youdenj": float(f1_opt),
                "precision_youdenj": float(prec_opt),
                "recall_youdenj": float(rec_opt),
            },
            "bootstrap_95ci": cis,
        }
        with (out_dir / "metrics_test.json").open("w", encoding="utf-8") as f:
            json.dump(test_payload, f, indent=2)
        logger.info(f"Test Accuracy={acc:.4f} AUC={auc:.4f} F1={f1:.4f} "
                    f"| YoudenJ thr={thr_opt:.3f} Acc_opt={acc_opt:.4f} F1_opt={f1_opt:.4f}")

        # Grad-CAM samples
        save_gradcam_samples(model, test_loader, device, mean, std, out_dir / "gradcam_samples",
                             train_set.classes, logger, max_samples=8)
    else:
        logger.warning("Skipping test evaluation (no test/ split found).")

    # Repro manifest
    with (out_dir / "repro_manifest.txt").open("w", encoding="utf-8") as f:
        f.write(f"seed={args.seed}\n")
        f.write(f"device={device}\n")
        f.write(f"torch={torch.__version__} torchvision={torchvision_version_str()}\n")
        f.write(f"cuda={torch.version.cuda if torch.cuda.is_available() else 'cpu'} "
                f"cudnn={cudnn.version()}\n")
        f.write(f"transforms=train({args.hflip=}), eval(Resize->ToTensor->Normalize)\n")
        f.write(json.dumps({"mean": mean, "std": std}, indent=2))

    logger.info("Done.")


def torchvision_version_str() -> str:
    try:
        import torchvision
        return torchvision.__version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
"""Pure participant-LOSO deep baselines for FaceEMG-11.

The outer target subject is completely held out: all 30 blocks are evaluated
only after training/epoch selection on the other 11 participants.  Epoch
selection is itself participant-grouped: for each outer fold, each of the 11
source subjects is used once as an inner validation participant while the
other ten source subjects train the model.  Normalization is fitted only on
the corresponding inner-training or final-source data.  The target subject is
never used for statistics, batch-normalization updates, early stopping, or
model selection.

This file intentionally depends on the repository's config.py and
dataset.py, so it can be copied into a clean reproduction workspace
directory without changing the release data package.
"""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config as cfg
from dataset import load_trials


ARCHITECTURES = ("compactcnn", "eegnet", "facial1dcnn", "cnntcn")
SELECTION_SEED = 11
FINAL_SEEDS = (11, 23, 37, 53, 71)
SELECTION_PROTOCOL = "outer_target_unread; inner_grouped_loso_all30; source_only_normalization"
FINAL_PROTOCOL = "outer_target_all30; epoch_from_inner_grouped_loso; source_only_normalization"

ARCHITECTURAL_PROVENANCE = {
    "compactcnn": {
        "display_name": "Raw-waveform CompactCNN (project control)",
        "status": "project_custom_control",
        "source": "this project",
        "fidelity_boundary": "custom architecture; not attributed to a published layer-exact CompactCNN",
    },
    "eegnet": {
        "display_name": "EEGNet-style (20-channel sEMG adaptation)",
        "status": "cross_modality_family_adaptation",
        "source": "Lawhern et al., 2018",
        "fidelity_boundary": "EEGNet-style temporal-spatial/depthwise topology; not a layer-exact reproduction on the original EEG task",
    },
    "facial1dcnn": {
        "display_name": "Facial 1-D CNN family adaptation (source-only z-score)",
        "status": "method_family_adaptation",
        "source": "Ramadhan and Adinandra, 2025",
        "fidelity_boundary": "project three-layer Conv1d adaptation; no claim of layer-exact reproduction",
    },
    "cnntcn": {
        "display_name": "CNN-TCN family adaptation (source-only z-score)",
        "status": "method_family_adaptation",
        "source": "Meybodi et al., 2026",
        "fidelity_boundary": "project non-causal residual dilated-TCN adaptation; replaces participant-specific MVIC normalization and is not layer-exact",
    },
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


class CompactCNN(nn.Module):
    def __init__(self, channels=cfg.CHANNELS, classes=len(cfg.LABEL_IDS)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, 64, 15, stride=3, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 64, 15, padding=7, groups=64, bias=False),
            nn.Conv1d(64, 96, 1, bias=False), nn.BatchNorm1d(96), nn.GELU(),
            nn.Dropout(0.25),
            nn.Conv1d(96, 96, 9, stride=2, padding=4, groups=96, bias=False),
            nn.Conv1d(96, 128, 1, bias=False), nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, classes)

    def forward(self, x):
        return self.classifier(self.features(x).squeeze(-1))


class EEGNet(nn.Module):
    """EEGNet-style temporal-spatial depthwise network adapted to sEMG."""

    def __init__(self, channels=cfg.CHANNELS, classes=len(cfg.LABEL_IDS)):
        super().__init__()
        f1, depth, f2 = 16, 2, 32
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * depth, (channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(0.25),
            nn.Conv2d(f1 * depth, f1 * depth, (1, 16), padding=(0, 8),
                      groups=f1 * depth, bias=False),
            nn.Conv2d(f1 * depth, f2, 1, bias=False),
            nn.BatchNorm2d(f2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(f2, classes)

    def forward(self, x):
        return self.classifier(self.features(x[:, None]).flatten(1))


class Facial1DCNN(nn.Module):
    """Project 1-D CNN in the Ramadhan facial-EMG method family."""

    def __init__(self, channels=cfg.CHANNELS, classes=len(cfg.LABEL_IDS)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, 64, 25, padding=12, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(0.20),
            nn.Conv1d(64, 96, 15, padding=7, bias=False),
            nn.BatchNorm1d(96), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(0.20),
            nn.Conv1d(96, 128, 9, padding=4, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, classes)

    def forward(self, x):
        return self.classifier(self.features(x).squeeze(-1))


class TCNBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=0.20):
        super().__init__()
        kernel = 5
        pad = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class CNNTCN(nn.Module):
    """Project non-causal CNN plus residual dilated-TCN family adaptation."""

    def __init__(self, channels=cfg.CHANNELS, classes=len(cfg.LABEL_IDS)):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv1d(channels, 64, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 96, 9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(96), nn.GELU(),
        )
        self.tcn = nn.Sequential(
            TCNBlock(96, 1), TCNBlock(96, 2), TCNBlock(96, 4), TCNBlock(96, 8)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(96, classes)

    def forward(self, x):
        x = self.tcn(self.front(x))
        return self.classifier(self.pool(x).squeeze(-1))


def make_model(name):
    return {
        "compactcnn": CompactCNN,
        "eegnet": EEGNet,
        "facial1dcnn": Facial1DCNN,
        "cnntcn": CNNTCN,
    }[name]()


def load_corpus(subjects=None):
    """Load only the explicitly requested participants.

    In selection mode ``subjects`` excludes the outer target, so the target
    file is not even opened for validation or metadata inspection.
    """
    subjects = tuple(cfg.SUBJECTS if subjects is None else subjects)
    if len(subjects) != len(set(subjects)) or not set(subjects).issubset(set(cfg.SUBJECTS)):
        raise ValueError("subjects must be unique members of cfg.SUBJECTS")
    values = {}
    for subject in subjects:
        rec = load_trials(subject)
        if rec.data.ndim != 3 or rec.data.shape[1:] != (cfg.CHANNELS, cfg.TRIAL_SAMPLES):
            raise AssertionError("%s data must have shape (trials, %d, %d)" %
                                 (subject, cfg.CHANNELS, cfg.TRIAL_SAMPLES))
        if len(rec.labels) != len(cfg.LABEL_IDS) * cfg.RECORDED_BLOCKS:
            raise AssertionError("%s must contain exactly 330 trials" % subject)
        if sorted(np.unique(rec.groups).astype(int).tolist()) != list(range(1, cfg.RECORDED_BLOCKS + 1)):
            raise AssertionError("%s blocks must be exactly 1..30" % subject)
        pair_counts = {(block, label): 0 for block in range(1, cfg.RECORDED_BLOCKS + 1)
                       for label in range(1, len(cfg.LABEL_IDS) + 1)}
        for block, label in zip(rec.groups.astype(int).tolist(), rec.labels.astype(int).tolist()):
            key = (block, label)
            if key not in pair_counts:
                raise AssertionError("%s has invalid block/label pair %r" % (subject, key))
            pair_counts[key] += 1
        if set(pair_counts.values()) != {1}:
            raise AssertionError("%s does not have one trial per block/label" % subject)
        values[subject] = {
            "data": rec.data.astype(np.float32, copy=False),
            "labels": rec.labels.astype(np.int64, copy=False) - 1,
            "blocks": rec.groups.astype(np.int64, copy=False),
            "path": str(rec.path),
        }
    return values


def concat_subjects(corpus, subjects):
    x = np.concatenate([corpus[s]["data"] for s in subjects], axis=0)
    y = np.concatenate([corpus[s]["labels"] for s in subjects], axis=0)
    return x, y


def fit_standardizer(reference):
    center = reference.mean(axis=(0, 2), keepdims=True)
    scale = np.maximum(reference.std(axis=(0, 2), keepdims=True), 1e-6)
    return center.astype(np.float32), scale.astype(np.float32)


def apply_standardizer(x, center, scale):
    return ((x - center) / scale).astype(np.float32, copy=False)


def source_target_masks(corpus, subject, blocks=None):
    blocks = set(range(1, cfg.RECORDED_BLOCKS + 1) if blocks is None else blocks)
    mask = np.isin(corpus[subject]["blocks"], list(blocks))
    return mask


def data_loader(x, y, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))),
        batch_size=batch_size, shuffle=shuffle, num_workers=0,
        pin_memory=True, generator=generator,
    )


def augment(x):
    gain = torch.empty(x.shape[0], x.shape[1], 1, device=x.device).uniform_(0.9, 1.1)
    keep = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > 0.025)
    return x * gain * keep + 0.01 * torch.randn_like(x)


def evaluate(model, x, y, batch_size, device, return_probs=False):
    model.eval()
    loader = data_loader(x, y, batch_size, False, 0)
    truth, pred, probs = [], [], []
    with torch.no_grad():
        for bx, by in loader:
            logits = model(bx.to(device, non_blocking=True))
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
            pred.append(np.argmax(p, axis=1))
            truth.append(by.numpy())
    truth = np.concatenate(truth)
    pred = np.concatenate(pred)
    prob = np.concatenate(probs)
    row = {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro")),
        "counts": {str(i + 1): int(np.sum(truth == i)) for i in range(len(cfg.LABEL_IDS))},
        "truth": (truth + 1).tolist(),
        "prediction": (pred + 1).tolist(),
    }
    if return_probs:
        row["probabilities"] = prob.tolist()
    return row


def train_model(name, x, y, epochs, schedule_horizon, batch_size, seed, device, val=None):
    if not 1 <= int(epochs) <= int(schedule_horizon):
        raise ValueError("epochs must be within 1..schedule_horizon")
    seed_all(seed)
    model = make_model(name).to(device)
    loader = data_loader(x, y, batch_size, True, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    # Selection observes prefixes of one max-epoch schedule.  Final refits must
    # use that same prefix, rather than compressing a full cosine cycle into the
    # selected number of epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(schedule_horizon), eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in loader:
            bx = augment(bx.to(device, non_blocking=True))
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if val is not None:
            metrics = evaluate(model, val[0], val[1], batch_size, device)
            curve.append({"epoch": epoch, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
    return model, curve


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp.%d" % os.getpid())
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temp), str(path))


def config_hash(args):
    text = json.dumps(vars(args), sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def script_hashes():
    return {
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "config_sha256": file_sha256(Path(cfg.__file__).resolve()),
        "dataset_sha256": file_sha256(Path(__import__("dataset").__file__).resolve()),
    }


def data_hash_manifest(corpus, manifest_path):
    """Create/read an exact SHA256 manifest once per remote experiment."""
    manifest_path = Path(manifest_path)
    if set(corpus) != set(cfg.SUBJECTS):
        raise ValueError("data hash manifest requires the complete 12-participant corpus")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = sorted(corpus[s]["path"] for s in cfg.SUBJECTS)
        if (sorted(payload.get("paths", [])) == expected and
                sorted(payload.get("sha256", {}).keys()) == expected):
            return payload
    lock = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    owned = False
    try:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii"))
            os.close(fd)
            owned = True
        except FileExistsError:
            owned = False
        if owned:
            paths = sorted(corpus[s]["path"] for s in cfg.SUBJECTS)
            payload = {"schema_version": 1, "paths": paths,
                       "sha256": {path: file_sha256(path) for path in paths}}
            atomic_json(manifest_path, payload)
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
            return payload
        for _ in range(900):
            if manifest_path.is_file():
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected = sorted(corpus[s]["path"] for s in cfg.SUBJECTS)
                if (sorted(payload.get("paths", [])) == expected and
                        sorted(payload.get("sha256", {}).keys()) == expected):
                    return payload
            time.sleep(1)
        raise RuntimeError("timed out waiting for data SHA256 manifest")
    finally:
        if owned and lock.exists():
            lock.unlink()


def inner_selection(target_index, architecture, args):
    target = cfg.SUBJECTS[target_index]
    source = [s for s in cfg.SUBJECTS if s != target]
    corpus = load_corpus(source)
    if target in corpus or set(corpus) != set(source):
        raise AssertionError("selection corpus must contain exactly the 11 source participants")
    device = torch.device("cuda")
    curves = []
    for dev in source:
        train_subjects = [s for s in source if s != dev]
        train_x, train_y = concat_subjects(corpus, train_subjects)
        val_x = corpus[dev]["data"]
        val_y = corpus[dev]["labels"]
        center, scale = fit_standardizer(train_x)
        train_x = apply_standardizer(train_x, center, scale)
        val_x = apply_standardizer(val_x, center, scale)
        model, curve = train_model(architecture, train_x, train_y, args.max_epochs,
                                   args.max_epochs, args.batch_size, SELECTION_SEED,
                                   device, (val_x, val_y))
        curves.append({
            "inner_validation_subject": dev,
            "inner_training_subjects": train_subjects,
            "normalization_fit_subjects": train_subjects,
            "curve": curve,
        })
        del model
        torch.cuda.empty_cache()
    epochs = list(range(1, args.max_epochs + 1))
    mean_acc = []
    mean_f1 = []
    for epoch in epochs:
        rows = [item["curve"][epoch - 1] for item in curves]
        mean_acc.append(float(np.mean([r["accuracy"] for r in rows])))
        mean_f1.append(float(np.mean([r["macro_f1"] for r in rows])))
    # All inner validation participants have exactly 330 trials.  Compare
    # integer numbers correct so a mathematically exact tie cannot be broken by
    # ~1e-16 floating summation noise.
    correct_by_epoch = [
        sum(int(round(item["curve"][epoch - 1]["accuracy"] * len(corpus[item["inner_validation_subject"]]["labels"])))
            for item in curves)
        for epoch in epochs
    ]
    best_epoch = max(epochs, key=lambda e: (correct_by_epoch[e - 1], -e))
    payload = {
        "schema_version": 3,
        "experiment": "faceemg11_deep_baseline_selection",
        "protocol": SELECTION_PROTOCOL,
        "architecture": architecture,
        "outer_target_subject": target,
        "source_subjects": source,
        "inner_validation_subjects": source,
        "selection_seed": SELECTION_SEED,
        "max_epochs": args.max_epochs,
        "selected_epoch": int(best_epoch),
        "inner_mean_accuracy_by_epoch": mean_acc,
        "inner_mean_macro_f1_by_epoch": mean_f1,
        "inner_total_correct_by_epoch": correct_by_epoch,
        "inner_curves": curves,
        "target_derived_statistics": [],
        "target_file_opened": False,
        "target_data_role": "not loaded",
        "selection_evidence": {"criterion": "mean participant-grouped inner validation accuracy; earliest epoch wins ties",
                               "target_used": False, "normalization_fit_subjects": "inner-train only"},
        "run_config_sha256": config_hash(args),
        **script_hashes(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    atomic_json(Path(args.selection_dir) / (architecture + "__" + target + ".json"), payload)
    print(json.dumps({"architecture": architecture, "target": target, "selected_epoch": best_epoch}, sort_keys=True), flush=True)


def validate_selection(selection, target, architecture, expected_source, max_epochs):
    """Fail closed before a selection artifact can control a final fit."""
    if int(selection.get("schema_version", -1)) != 3:
        raise ValueError("unsupported selection schema")
    if selection.get("experiment") != "faceemg11_deep_baseline_selection":
        raise ValueError("selection experiment mismatch")
    if selection.get("protocol") != SELECTION_PROTOCOL:
        raise ValueError("selection protocol mismatch")
    if selection.get("architecture") != architecture:
        raise ValueError("selection architecture mismatch")
    if selection.get("outer_target_subject") != target:
        raise ValueError("selection target mismatch")
    if selection.get("source_subjects") != expected_source:
        raise ValueError("selection source subjects mismatch")
    if target in selection.get("source_subjects", []):
        raise ValueError("outer target appears among selection sources")
    if selection.get("inner_validation_subjects") != expected_source:
        raise ValueError("inner validation subjects mismatch")
    if int(selection.get("selection_seed", -1)) != SELECTION_SEED:
        raise ValueError("selection seed mismatch")
    if int(selection.get("max_epochs", -1)) != int(max_epochs):
        raise ValueError("selection max_epochs mismatch")
    if selection.get("target_derived_statistics") != []:
        raise ValueError("selection declares target-derived statistics")
    evidence = selection.get("selection_evidence", {})
    if evidence.get("target_used") is not False:
        raise ValueError("selection does not explicitly declare target_used=false")
    if evidence.get("normalization_fit_subjects") != "inner-train only":
        raise ValueError("selection normalization boundary mismatch")
    if selection.get("target_file_opened") is not False:
        raise ValueError("selection must declare target_file_opened=false")

    curves = selection.get("inner_curves")
    if not isinstance(curves, list) or len(curves) != len(expected_source):
        raise ValueError("selection must contain one curve per source participant")
    if [row.get("inner_validation_subject") for row in curves] != expected_source:
        raise ValueError("inner curve participant ordering mismatch")
    for row in curves:
        dev = row["inner_validation_subject"]
        expected_train = [s for s in expected_source if s != dev]
        if row.get("inner_training_subjects") != expected_train:
            raise ValueError("inner training subject boundary mismatch")
        if row.get("normalization_fit_subjects") != expected_train:
            raise ValueError("inner normalization subject boundary mismatch")
        curve = row.get("curve")
        if not isinstance(curve, list) or len(curve) != int(max_epochs):
            raise ValueError("incomplete inner validation curve")
        if [int(point.get("epoch", -1)) for point in curve] != list(range(1, int(max_epochs) + 1)):
            raise ValueError("non-canonical epoch sequence")
        for point in curve:
            for metric in ("accuracy", "macro_f1"):
                value = float(point.get(metric, np.nan))
                if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError("invalid inner curve metric")

    recomputed_acc = [float(np.mean([row["curve"][e]["accuracy"] for row in curves]))
                      for e in range(int(max_epochs))]
    recomputed_f1 = [float(np.mean([row["curve"][e]["macro_f1"] for row in curves]))
                     for e in range(int(max_epochs))]
    stored_acc = np.asarray(selection.get("inner_mean_accuracy_by_epoch"), dtype=float)
    stored_f1 = np.asarray(selection.get("inner_mean_macro_f1_by_epoch"), dtype=float)
    if stored_acc.shape != (int(max_epochs),) or not np.allclose(stored_acc, recomputed_acc, atol=1e-12):
        raise ValueError("stored inner mean accuracy is inconsistent with curves")
    if stored_f1.shape != (int(max_epochs),) or not np.allclose(stored_f1, recomputed_f1, atol=1e-12):
        raise ValueError("stored inner mean macro-F1 is inconsistent with curves")
    correct_by_epoch = [
        sum(int(round(row["curve"][e]["accuracy"] * len(cfg.LABEL_IDS) * cfg.RECORDED_BLOCKS))
            for row in curves)
        for e in range(int(max_epochs))
    ]
    stored_correct = selection.get("inner_total_correct_by_epoch")
    if stored_correct is not None and [int(value) for value in stored_correct] != correct_by_epoch:
        raise ValueError("stored integer-correct selection scores are inconsistent with curves")
    recomputed_epoch = max(range(1, int(max_epochs) + 1),
                           key=lambda e: (correct_by_epoch[e - 1], -e))
    if int(selection.get("selected_epoch", -1)) != recomputed_epoch:
        raise ValueError("selected_epoch is inconsistent with the declared criterion")
    return recomputed_epoch


def final_run(target_index, architecture, seed, args):
    target = cfg.SUBJECTS[target_index]
    source = [s for s in cfg.SUBJECTS if s != target]
    selection_path = Path(args.selection_dir) / (architecture + "__" + target + ".json")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    expected_source = [s for s in cfg.SUBJECTS if s != target]
    selected_epoch = validate_selection(selection, target, architecture, expected_source,
                                        args.max_epochs)
    selection_sha256 = file_sha256(selection_path)
    corpus = load_corpus()
    source_x, source_y = concat_subjects(corpus, source)
    target_x = corpus[target]["data"]
    target_y = corpus[target]["labels"]
    target_blocks = corpus[target]["blocks"]
    if len(target_y) != len(cfg.LABEL_IDS) * cfg.RECORDED_BLOCKS:
        raise AssertionError("target must contain exactly 330 trials")
    if sorted(np.unique(target_blocks).tolist()) != list(range(1, cfg.RECORDED_BLOCKS + 1)):
        raise AssertionError("target blocks must be exactly 1..30")
    pair_counts = {(int(block), int(label)): 0 for block in range(1, cfg.RECORDED_BLOCKS + 1)
                   for label in range(len(cfg.LABEL_IDS))}
    for block, label in zip(target_blocks.tolist(), target_y.tolist()):
        key = (int(block), int(label))
        if key not in pair_counts:
            raise AssertionError("invalid target block/label pair")
        pair_counts[key] += 1
    if set(pair_counts.values()) != {1}:
        raise AssertionError("each target block/label pair must occur exactly once")
    center, scale = fit_standardizer(source_x)
    source_x = apply_standardizer(source_x, center, scale)
    target_x = apply_standardizer(target_x, center, scale)
    device = torch.device("cuda")
    model, _ = train_model(architecture, source_x, source_y, selected_epoch,
                           args.max_epochs, args.batch_size, seed, device)
    metrics = evaluate(model, target_x, target_y, args.batch_size, device, return_probs=True)
    trial_index = list(range(len(target_y)))
    payload = {
        "schema_version": 3,
        "experiment": "faceemg11_deep_baseline_evaluation",
        "protocol": FINAL_PROTOCOL,
        "model_id": architecture,
        "display_name": ARCHITECTURAL_PROVENANCE[architecture]["display_name"],
        "architecture": architecture,
        "architectural_provenance": ARCHITECTURAL_PROVENANCE[architecture],
        "outer_target_subject": target,
        "source_subjects": source,
        "seed": int(seed),
        "selected_epoch": selected_epoch,
        "target_blocks": sorted(np.unique(target_blocks).astype(int).tolist()),
        "trial_index": trial_index,
        "blocks": target_blocks.astype(int).tolist(),
        "target_trial_count": int(len(target_y)),
        "class_order": list(cfg.LABEL_IDS),
        "target_derived_statistics": [],
        "target_derived_fitting_or_selection_statistics": [],
        "target_data_role": "evaluation only after the fit and all selections are frozen",
        "selection_sha256": selection_sha256,
        "source_data_subjects": source,
        "normalization": {
            "method": "per-channel z-score over trial and time axes",
            "fit_subjects": source,
            "target_used_for_fit": False,
        },
        "training_schedule": {
            "optimizer": "AdamW",
            "cosine_horizon_epochs": int(args.max_epochs),
            "epochs_run": selected_epoch,
            "schedule_matches_selection_prefix": True,
        },
        "data_hash_manifest": data_hash_manifest(corpus, Path(args.final_dir) / "data_hash_manifest.json"),
        "selection_evidence": {"selection_file": str(selection_path), "selection_target_used": False,
                               "selection_source_subjects": source, "selection_seed": SELECTION_SEED},
        "parameter_count": int(sum(p.numel() for p in make_model(architecture).parameters())),
        "metrics": metrics,
        **script_hashes(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    atomic_json(Path(args.final_dir) / (architecture + "__" + target + "__seed" + str(seed) + ".json"), payload)
    print(json.dumps({"architecture": architecture, "target": target, "seed": seed,
                      "epoch": selected_epoch, "accuracy": metrics["accuracy"],
                      "macro_f1": metrics["macro_f1"]}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("selection", "final"), required=True)
    parser.add_argument("--target-index", type=int, required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--selection-dir", default="results/deep_selection")
    parser.add_argument("--final-dir", default="results/deep_final")
    args = parser.parse_args()
    if not 0 <= args.target_index < len(cfg.SUBJECTS):
        parser.error("--target-index must be in 0..%d" % (len(cfg.SUBJECTS) - 1))
    if args.max_epochs < 1:
        parser.error("--max-epochs must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.mode == "selection" and args.seed != SELECTION_SEED:
        parser.error("selection uses the fixed selection seed %d; do not pass another --seed" % SELECTION_SEED)
    if args.mode == "final" and args.seed not in FINAL_SEEDS:
        parser.error("final --seed must be one of %s" % (FINAL_SEEDS,))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.mode == "selection":
        inner_selection(args.target_index, args.architecture, args)
    else:
        final_run(args.target_index, args.architecture, args.seed, args)


if __name__ == "__main__":
    main()

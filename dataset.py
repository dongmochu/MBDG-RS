"""Reader for deidentified FaceEMG-11 trial files used by the final protocol."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config as cfg


REQUIRED_TRIAL_KEYS = ("data", "labels", "groups", "fs", "ch_names")


@dataclass
class TrialRecord:
    subject: str
    data: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    fs: float
    channel_names: np.ndarray
    path: Path


def trial_path(subject: str, root=None) -> Path:
    base = Path(root) if root else cfg.TRIAL_ROOT
    return base / (subject + "_trials.npz")


def load_trials(subject: str, root=None) -> TrialRecord:
    if subject not in cfg.SUBJECTS:
        raise ValueError("unknown pseudonymous subject: %s" % subject)
    path = trial_path(subject, root)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with np.load(str(path), allow_pickle=False) as payload:
        missing = sorted(set(REQUIRED_TRIAL_KEYS) - set(payload.files))
        if missing:
            raise ValueError("%s is missing keys %s" % (path, missing))
        values = {key: payload[key] for key in REQUIRED_TRIAL_KEYS}
    return TrialRecord(
        subject=subject,
        data=values["data"],
        labels=values["labels"].astype(np.int64, copy=False),
        groups=values["groups"].astype(np.int64, copy=False),
        fs=float(values["fs"]),
        channel_names=values["ch_names"],
        path=path,
    )


def validate_corpus(subjects=cfg.SUBJECTS, root=None) -> dict:
    rows = []
    for subject in subjects:
        record = load_trials(subject, root=root)
        labels, label_counts = np.unique(record.labels, return_counts=True)
        blocks, block_counts = np.unique(record.groups, return_counts=True)
        checks = {
            "shape": tuple(record.data.shape) == (330, cfg.CHANNELS, cfg.TRIAL_SAMPLES),
            "labels": labels.tolist() == list(cfg.LABEL_IDS),
            "label_balance": set(label_counts.tolist()) == {cfg.RECORDED_BLOCKS},
            "blocks": blocks.tolist() == list(range(1, cfg.RECORDED_BLOCKS + 1)),
            "block_balance": set(block_counts.tolist()) == {len(cfg.LABEL_IDS)},
            "sample_rate": record.fs == cfg.SAMPLE_RATE_HZ,
            "finite": bool(np.isfinite(record.data).all()),
        }
        rows.append({"subject": subject, "path": str(record.path), "checks": checks,
                     "valid": all(checks.values())})
    return {"schema_version": 1, "subjects": rows,
            "valid": all(row["valid"] for row in rows)}

"""Compute final paired participant-level comparisons from result aggregates."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from model_names import MBDG_RS_FULL_ABLATION_ID, MBDG_RS_NAME


def bootstrap_ci(values, seed, resamples=100_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, len(values), size=(resamples, len(values)))
    low, high = np.percentile(values[indices].mean(axis=1), [2.5, 97.5])
    return {"low": float(low), "high": float(high), "low_pp": float(100 * low),
            "high_pp": float(100 * high), "resamples": resamples,
            "unit": "outer participant"}


def exact_sign_flip(values):
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    null = [abs(float(np.mean(values * np.asarray(signs))))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def compare(left, right, label, seed):
    difference = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return {"estimand": label, "mean": float(difference.mean()),
            "mean_pp": float(100 * difference.mean()),
            "sample_sd": float(difference.std(ddof=1)),
            "sample_sd_pp": float(100 * difference.std(ddof=1)),
            "participant_values": difference.tolist(),
            "participant_values_pp": (100 * difference).tolist(),
            "participant_bootstrap_95_ci": bootstrap_ci(difference, seed),
            "exact_sign_flip_p": exact_sign_flip(difference)}


def read_mbdg_rs_f1(fold_dir, targets):
    rows = [json.loads(path.read_text(encoding="utf-8"))
            for path in Path(fold_dir).glob("fold_[0-9][0-9].json")]
    rows.sort(key=lambda row: row["protocol"]["outer_target"])
    if [row["protocol"]["outer_target"] for row in rows] != targets:
        raise ValueError("MBDG-RS fold targets do not match the aggregate")
    values = []
    for row in rows:
        methods = row["fixed_1p5s_ablation"]["methods"]
        values.append(methods[MBDG_RS_FULL_ABLATION_ID]["macro_f1"])
    return np.asarray(values, dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbdg-rs", dest="mbdg_rs", type=Path, required=True)
    parser.add_argument(
        "--mbdg-rs-fold-dir",
        dest="mbdg_rs_fold_dir", type=Path, required=True,
    )
    parser.add_argument("--deep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mbdg_rs = json.loads(args.mbdg_rs.read_text(encoding="utf-8"))
    deep = json.loads(args.deep.read_text(encoding="utf-8"))
    targets = list(mbdg_rs["outer_targets"])
    fixed = mbdg_rs["fixed_1p5s_ablation"]
    mbdg_rs_accuracy = np.asarray(
        fixed[MBDG_RS_FULL_ABLATION_ID]["per_subject_accuracy"], dtype=float
    )
    mbdg_rs_f1 = read_mbdg_rs_f1(args.mbdg_rs_fold_dir, targets)

    facial = deep["summary"]["facial1dcnn"]["per_subject"]
    if [row["subject"] for row in facial] != targets:
        raise ValueError("Facial 1-D CNN participant order does not match MBDG-RS")
    facial_accuracy = np.asarray([row["accuracy"] for row in facial], dtype=float)
    facial_f1 = np.asarray([row["macro_f1"] for row in facial], dtype=float)

    payload = {
        "schema_version": 1,
        "inference_unit": "outer participant",
        "n_outer_participants": len(targets),
        "outer_targets": targets,
        "comparisons": {
            "accuracy": {
                "MBDG-RS_minus_Facial1D-CNN": compare(mbdg_rs_accuracy, facial_accuracy, MBDG_RS_NAME + " minus Facial 1-D CNN", 2026082502),
            },
            "macro_f1": {
                "MBDG-RS_minus_Facial1D-CNN": compare(mbdg_rs_f1, facial_f1, MBDG_RS_NAME + " minus Facial 1-D CNN", 2026082504),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

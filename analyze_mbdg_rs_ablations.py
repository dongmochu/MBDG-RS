#!/usr/bin/env python3
"""Strict participant-level statistics for MBDG-RS component ablations."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from model_names import MBDG_RS_FULL_ABLATION_ID


METHODS = (
    "broadband_dual_geometry_relative_spectrum",
    "five_band_correlation_relative_spectrum",
    "five_band_covariance_relative_spectrum",
    "five_band_dual_geometry_no_spectrum",
    MBDG_RS_FULL_ABLATION_ID,
)
COMPARISONS = (
    ("full_minus_broadband", MBDG_RS_FULL_ABLATION_ID, "broadband_dual_geometry_relative_spectrum"),
    ("full_minus_correlation_only", MBDG_RS_FULL_ABLATION_ID, "five_band_correlation_relative_spectrum"),
    ("full_minus_covariance_only", MBDG_RS_FULL_ABLATION_ID, "five_band_covariance_relative_spectrum"),
    ("full_minus_no_spectrum", MBDG_RS_FULL_ABLATION_ID, "five_band_dual_geometry_no_spectrum"),
)
SUPPORTED_EXPERIMENTS = {"faceemg11_mbdg_rs_ablation_loso"}
BOOTSTRAP_RESAMPLES = 100000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def macro_f1(truth, prediction):
    truth = np.asarray(truth, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    values = []
    labels = sorted(set(truth.tolist()) | set(prediction.tolist()))
    if labels != list(range(1, 12)):
        raise ValueError("expected labels 1..11, found %r" % labels)
    for label in labels:
        tp = np.sum((truth == label) & (prediction == label))
        fp = np.sum((truth != label) & (prediction == label))
        fn = np.sum((truth == label) & (prediction != label))
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall)
                      if precision + recall else 0.0)
    return float(np.mean(values))


def metric_row(row, context):
    truth = np.asarray(row.get("truth", []), dtype=int)
    prediction = np.asarray(row.get("prediction", []), dtype=int)
    if len(truth) != 330 or len(prediction) != 330:
        raise ValueError("%s: expected 330 truth/prediction values" % context)
    accuracy = float(np.mean(truth == prediction))
    f1 = macro_f1(truth, prediction)
    if not np.isclose(accuracy, float(row.get("accuracy", np.nan)), atol=1e-12):
        raise ValueError("%s: accuracy cannot be reproduced" % context)
    if not np.isclose(f1, float(row.get("macro_f1", np.nan)), atol=1e-12):
        raise ValueError("%s: macro-F1 cannot be reproduced" % context)
    return accuracy, f1


def bootstrap(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(int(seed))
    indices = rng.randint(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "low": float(low), "high": float(high),
        "low_pp": float(100 * low), "high_pp": float(100 * high),
        "resamples": BOOTSTRAP_RESAMPLES, "seed": int(seed),
        "unit": "outer participant",
    }


def sign_flip(values):
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    enumerated = np.asarray([
        np.mean(values * np.asarray(signs, dtype=float))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ])
    p = float(np.mean(np.abs(enumerated) >= observed - 1e-15))
    return {"p_value_two_sided": p, "enumerations": int(len(enumerated))}


def holm(raw):
    ordered = sorted(raw, key=raw.get)
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * raw[name]))
        adjusted[name] = running
    return adjusted


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()), "mean_percent": float(100 * values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "sample_sd_percent": float(100 * values.std(ddof=1)),
        "per_participant": values.tolist(),
        "per_participant_percent": (100 * values).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("fold_[0-9][0-9].json"))
    if len(paths) != 12:
        raise ValueError("expected 12 folds, found %d" % len(paths))
    folds = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    targets = []
    script_hashes = set()
    values = {metric: {method: [] for method in METHODS}
              for metric in ("accuracy", "macro_f1")}
    for path, fold in zip(paths, folds):
        if fold.get("schema_version") != 3:
            raise ValueError("%s: unexpected schema" % path.name)
        if fold.get("experiment") not in SUPPORTED_EXPERIMENTS:
            raise ValueError("%s: unexpected experiment" % path.name)
        protocol = fold.get("protocol", {})
        target = protocol.get("outer_target")
        targets.append(target)
        if protocol.get("target_samples_used_for_fit_or_selection") != 0:
            raise ValueError("%s: target leakage flag" % path.name)
        script_hashes.add(fold.get("script_sha256"))
        rows = fold.get("fixed_1p5s_ablation", {}).get("methods", {})
        if set(rows) != set(METHODS):
            raise ValueError("%s: method set mismatch" % path.name)
        for method in METHODS:
            accuracy, f1 = metric_row(rows[method], "%s/%s" % (path.name, method))
            values["accuracy"][method].append(accuracy)
            values["macro_f1"][method].append(f1)
    if len(set(targets)) != 12 or len(script_hashes) != 1:
        raise ValueError("outer targets or script hashes are inconsistent")

    output = {
        "schema_version": 1,
        "experiment": "faceemg11_mbdg_rs_ablation_statistics",
        "n_outer_participants": 12,
        "outer_targets": targets,
        "inference_unit": "outer participant",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "source_script_sha256": next(iter(script_hashes)),
        "source_fold_sha256": {path.name: sha256(path) for path in paths},
        "summaries": {},
        "comparisons": {},
    }
    for metric in ("accuracy", "macro_f1"):
        output["summaries"][metric] = {
            method: summarize(values[metric][method]) for method in METHODS
        }
        raw = {}
        rows = {}
        for index, (name, left, right) in enumerate(COMPARISONS):
            difference = np.asarray(values[metric][left]) - np.asarray(values[metric][right])
            test = sign_flip(difference)
            raw[name] = test["p_value_two_sided"]
            rows[name] = {
                "estimand": "%s minus %s" % (left, right),
                **summarize(difference),
                "participant_bootstrap_95_ci": bootstrap(
                    difference, 2026082400 + 100 * (0 if metric == "accuracy" else 1) + index
                ),
                "exact_sign_flip": test,
            }
        adjusted = holm(raw)
        for name, value in adjusted.items():
            rows[name]["holm_adjusted_p"] = float(value)
        output["comparisons"][metric] = {
            "holm_family_size": len(COMPARISONS), "rows": rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "accuracy": {m: output["summaries"]["accuracy"][m]["mean_percent"] for m in METHODS},
        "accuracy_comparisons": {
            n: {"mean_pp": r["mean_percent"], "holm_p": r["holm_adjusted_p"]}
            for n, r in output["comparisons"]["accuracy"]["rows"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()

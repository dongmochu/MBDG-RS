"""Strictly validate and aggregate pure participant-LOSO deep outputs.

The primary participant value is the arithmetic mean of five independent
seed-level metrics. A probability ensemble is retained as a separately named
descriptive result; it is not silently substituted for seed aggregation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import config as cfg


ARCHITECTURES = ("compactcnn", "eegnet", "facial1dcnn", "cnntcn")
SEEDS = (11, 23, 37, 53, 71)
FINAL_PROTOCOL = "outer_target_all30; epoch_from_inner_grouped_loso; source_only_normalization"
DISPLAY_NAMES = {
    "compactcnn": "Raw-waveform CompactCNN (project control)",
    "eegnet": "EEGNet-style (20-channel sEMG adaptation)",
    "facial1dcnn": "Facial 1-D CNN family adaptation (source-only z-score)",
    "cnntcn": "CNN-TCN family adaptation (source-only z-score)",
}


def macro_f1(truth, prediction):
    values = []
    for label in cfg.LABEL_IDS:
        tp = np.sum((prediction == label) & (truth == label))
        fp = np.sum((prediction == label) & (truth != label))
        fn = np.sum((prediction != label) & (truth == label))
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall)
                      if precision + recall else 0.0)
    return float(np.mean(values))


def _assert_close(actual, expected, label, atol=1e-10):
    if not np.isfinite(float(actual)) or not np.isclose(float(actual), float(expected), atol=atol):
        raise AssertionError("%s mismatch: %r != %r" % (label, actual, expected))


def validate_payload(payload, path, architecture, subject, seed):
    expected_source = [s for s in cfg.SUBJECTS if s != subject]
    if int(payload.get("schema_version", -1)) != 3:
        raise AssertionError("%s does not use the supported result schema" % path)
    if payload.get("experiment") != "faceemg11_deep_baseline_evaluation":
        raise AssertionError("experiment mismatch in %s" % path)
    if payload.get("protocol") != FINAL_PROTOCOL:
        raise AssertionError("protocol mismatch in %s" % path)
    if payload.get("architecture") != architecture or payload.get("model_id") != architecture:
        raise AssertionError("architecture/model_id mismatch in %s" % path)
    if payload.get("display_name") != DISPLAY_NAMES[architecture]:
        raise AssertionError("display-name boundary mismatch in %s" % path)
    if payload.get("outer_target_subject") != subject:
        raise AssertionError("target mismatch in %s" % path)
    if int(payload.get("seed", -1)) != seed:
        raise AssertionError("seed mismatch in %s" % path)
    if payload.get("source_subjects") != expected_source or payload.get("source_data_subjects") != expected_source:
        raise AssertionError("source participant boundary mismatch in %s" % path)
    if subject in payload.get("source_subjects", []):
        raise AssertionError("target appears in source participants in %s" % path)
    if payload.get("target_derived_statistics") != []:
        raise AssertionError("target-derived fitting statistics declared in %s" % path)
    if payload.get("target_derived_fitting_or_selection_statistics") != []:
        raise AssertionError("target-derived fitting/selection statistics declared in %s" % path)
    if payload.get("target_data_role") != "evaluation only after the fit and all selections are frozen":
        raise AssertionError("target data role mismatch in %s" % path)
    normalization = payload.get("normalization", {})
    if normalization.get("fit_subjects") != expected_source or normalization.get("target_used_for_fit") is not False:
        raise AssertionError("normalization boundary mismatch in %s" % path)
    evidence = payload.get("selection_evidence", {})
    if (evidence.get("selection_target_used") is not False or
            evidence.get("selection_source_subjects") != expected_source or
            int(evidence.get("selection_seed", -1)) != 11):
        raise AssertionError("selection evidence mismatch in %s" % path)
    epoch = int(payload.get("selected_epoch", -1))
    schedule = payload.get("training_schedule", {})
    if (not 1 <= epoch <= 60 or int(schedule.get("cosine_horizon_epochs", -1)) != 60 or
            int(schedule.get("epochs_run", -1)) != epoch or
            schedule.get("schedule_matches_selection_prefix") is not True):
        raise AssertionError("training schedule mismatch in %s" % path)
    if payload.get("target_blocks") != list(range(1, 31)) or int(payload.get("target_trial_count", -1)) != 330:
        raise AssertionError("target must include all 30 blocks / 330 trials in %s" % path)
    if payload.get("class_order") != list(cfg.LABEL_IDS):
        raise AssertionError("class order mismatch in %s" % path)

    metrics = payload.get("metrics", {})
    truth = np.asarray(metrics.get("truth"), dtype=np.int64)
    prediction = np.asarray(metrics.get("prediction"), dtype=np.int64)
    trial_index = np.asarray(payload.get("trial_index"), dtype=np.int64)
    blocks = np.asarray(payload.get("blocks"), dtype=np.int64)
    probs = np.asarray(metrics.get("probabilities"), dtype=np.float64)
    if truth.shape != (330,) or prediction.shape != (330,) or trial_index.shape != (330,) or blocks.shape != (330,):
        raise AssertionError("trial metadata must contain 330 rows in %s" % path)
    if not np.array_equal(trial_index, np.arange(330)):
        raise AssertionError("trial_index is not canonical in %s" % path)
    if probs.shape != (330, len(cfg.LABEL_IDS)):
        raise AssertionError("probability shape must be 330 x 11 in %s" % path)
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise AssertionError("invalid probability rows in %s" % path)
    if not np.array_equal(np.argmax(probs, axis=1) + 1, prediction):
        raise AssertionError("predictions do not match probabilities in %s" % path)
    if set(truth.tolist()) != set(cfg.LABEL_IDS) or set(prediction.tolist()) - set(cfg.LABEL_IDS):
        raise AssertionError("label range mismatch in %s" % path)
    if any(int(np.sum(truth == label)) != 30 for label in cfg.LABEL_IDS):
        raise AssertionError("target truth is not 30 trials per class in %s" % path)
    if set(blocks.tolist()) != set(range(1, 31)) or any(int(np.sum(blocks == block)) != 11 for block in range(1, 31)):
        raise AssertionError("target blocks are not class-balanced in %s" % path)
    if len(set(zip(blocks.tolist(), truth.tolist()))) != 330:
        raise AssertionError("target block/class pairs are not unique in %s" % path)
    accuracy = float(np.mean(prediction == truth))
    f1 = macro_f1(truth, prediction)
    _assert_close(metrics.get("accuracy"), accuracy, "%s accuracy" % path)
    _assert_close(metrics.get("macro_f1"), f1, "%s macro_f1" % path)
    if metrics.get("counts") != {str(label): 30 for label in cfg.LABEL_IDS}:
        raise AssertionError("metric counts mismatch in %s" % path)
    if not isinstance(payload.get("selection_sha256"), str) or len(payload["selection_sha256"]) != 64:
        raise AssertionError("missing selection SHA256 in %s" % path)
    for key in ("script_sha256", "config_sha256", "dataset_sha256"):
        if not isinstance(payload.get(key), str) or len(payload[key]) != 64:
            raise AssertionError("missing %s in %s" % (key, path))
    manifest = payload.get("data_hash_manifest", {})
    if (int(manifest.get("schema_version", -1)) != 1 or
            len(manifest.get("paths", [])) != len(cfg.SUBJECTS) or
            sorted(manifest.get("paths", [])) != sorted(manifest.get("sha256", {}).keys())):
        raise AssertionError("invalid data hash manifest in %s" % path)
    provenance = payload.get("architectural_provenance", {})
    if provenance.get("display_name") != DISPLAY_NAMES[architecture]:
        raise AssertionError("architectural provenance mismatch in %s" % path)
    return truth, prediction, blocks, probs


def read_rows(final_dir, architecture):
    rows = []
    for subject in cfg.SUBJECTS:
        paths = [Path(final_dir) / (architecture + "__" + subject + "__seed" + str(seed) + ".json")
                 for seed in SEEDS]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing final outputs: " + ", ".join(missing))
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        validated = [validate_payload(payload, path, architecture, subject, seed)
                     for payload, path, seed in zip(payloads, paths, SEEDS)]
        ref_truth, _, ref_blocks, _ = validated[0]
        for truth, _, blocks, _ in validated[1:]:
            if not np.array_equal(truth, ref_truth) or not np.array_equal(blocks, ref_blocks):
                raise AssertionError("truth/block ordering differs across seeds for %s/%s" %
                                     (architecture, subject))
        if len({p["selection_sha256"] for p in payloads}) != 1:
            raise AssertionError("selection artifact differs across seeds for %s/%s" %
                                 (architecture, subject))
        if len({int(p["selected_epoch"]) for p in payloads}) != 1:
            raise AssertionError("selected epoch differs across seeds for %s/%s" %
                                 (architecture, subject))
        rows.append((subject, payloads))
    flat = [payload for _, payloads in rows for payload in payloads]
    for key in ("script_sha256", "config_sha256", "dataset_sha256"):
        if len({payload[key] for payload in flat}) != 1:
            raise AssertionError("mixed %s values for %s" % (key, architecture))
    if len({int(payload["parameter_count"]) for payload in flat}) != 1:
        raise AssertionError("parameter count differs across tasks for %s" % architecture)
    manifests = {json.dumps(payload["data_hash_manifest"], sort_keys=True, separators=(",", ":"))
                 for payload in flat}
    if len(manifests) != 1:
        raise AssertionError("data manifests differ across tasks for %s" % architecture)
    return rows


def aggregate_architecture(final_dir, architecture):
    rows = read_rows(final_dir, architecture)
    first_payload = rows[0][1][0]
    manifest_text = json.dumps(first_payload["data_hash_manifest"], sort_keys=True,
                               separators=(",", ":"))
    per_subject = []
    for subject, payloads in rows:
        seed_rows = [
            {"seed": int(payload["seed"]),
             "accuracy": float(payload["metrics"]["accuracy"]),
             "macro_f1": float(payload["metrics"]["macro_f1"]),
             "selected_epoch": int(payload["selected_epoch"])}
            for payload in payloads
        ]
        seed_acc = np.asarray([row["accuracy"] for row in seed_rows], dtype=float)
        seed_f1 = np.asarray([row["macro_f1"] for row in seed_rows], dtype=float)
        probs = np.asarray([p["metrics"]["probabilities"] for p in payloads], dtype=np.float64)
        truth = np.asarray(payloads[0]["metrics"]["truth"], dtype=np.int64)
        ensemble_prediction = np.argmax(probs.mean(axis=0), axis=1) + 1
        per_subject.append({
            "subject": subject,
            "accuracy": float(seed_acc.mean()),
            "macro_f1": float(seed_f1.mean()),
            "accuracy_seed_std": float(seed_acc.std(ddof=1)),
            "macro_f1_seed_std": float(seed_f1.std(ddof=1)),
            "selected_epoch": int(seed_rows[0]["selected_epoch"]),
            "seed_rows": seed_rows,
            "ensemble_accuracy_descriptive": float(np.mean(ensemble_prediction == truth)),
            "ensemble_macro_f1_descriptive": macro_f1(truth, ensemble_prediction),
            "target_trial_count": 330,
            "class_order": list(cfg.LABEL_IDS),
            "target_derived_statistics": [],
        })

    acc = np.asarray([row["accuracy"] for row in per_subject], dtype=float)
    f1 = np.asarray([row["macro_f1"] for row in per_subject], dtype=float)
    seed_level = []
    for seed in SEEDS:
        seed_acc = [next(item for item in row["seed_rows"] if item["seed"] == seed)["accuracy"]
                    for row in per_subject]
        seed_f1 = [next(item for item in row["seed_rows"] if item["seed"] == seed)["macro_f1"]
                   for row in per_subject]
        seed_level.append({
            "seed": seed,
            "mean_accuracy": float(np.mean(seed_acc)),
            "subject_std_accuracy": float(np.std(seed_acc, ddof=1)),
            "mean_macro_f1": float(np.mean(seed_f1)),
            "subject_std_macro_f1": float(np.std(seed_f1, ddof=1)),
        })
    return {
        "display_name": DISPLAY_NAMES[architecture],
        "parameter_count": int(first_payload["parameter_count"]),
        "artifact_hashes": {key: first_payload[key]
                            for key in ("script_sha256", "config_sha256", "dataset_sha256")},
        "data_manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        "aggregation_unit": "participant; each participant metric is the arithmetic mean of five seed-level metrics",
        "mean_accuracy": float(acc.mean()),
        "subject_std_accuracy": float(acc.std(ddof=1)),
        "mean_macro_f1": float(f1.mean()),
        "subject_std_macro_f1": float(f1.std(ddof=1)),
        "seed_level_summary": seed_level,
        "per_subject": per_subject,
    }


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.%d" % os.getpid())
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", default="results/deep_final")
    parser.add_argument("--output", default="results/deep/aggregate.json")
    args = parser.parse_args()
    summary = {architecture: aggregate_architecture(args.final_dir, architecture)
               for architecture in ARCHITECTURES}
    for key in ("script_sha256", "config_sha256", "dataset_sha256"):
        if len({summary[name]["artifact_hashes"][key] for name in ARCHITECTURES}) != 1:
            raise AssertionError("mixed %s values across architectures" % key)
    if len({summary[name]["data_manifest_sha256"] for name in ARCHITECTURES}) != 1:
        raise AssertionError("mixed data manifests across architectures")
    payload = {
        "schema_version": 3,
        "experiment": "faceemg11_deep_baseline_aggregate",
        "protocol": "12 outer targets x all 30 blocks x five independent final seeds",
        "primary_seed_aggregation": "mean seed-level metric within participant, then participant macro mean/SD",
        "architectures": list(ARCHITECTURES),
        "display_names": DISPLAY_NAMES,
        "seeds": list(SEEDS),
        "expected_final_gpu_tasks": len(ARCHITECTURES) * len(cfg.SUBJECTS) * len(SEEDS),
        "target_derived_statistics": [],
        "summary": summary,
    }
    atomic_json(args.output, payload)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

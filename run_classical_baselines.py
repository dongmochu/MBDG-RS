#!/usr/bin/env python3
"""Pure nested-LOSO classical/geometry benchmark for FaceEMG-11.

For one outer fold,
all 30 blocks from the eleven source subjects are used for fitting and all 30
blocks from the held-out subject are evaluated.  A candidate that has a
hyper-parameter is selected only by inner LOSO over the eleven source subjects;
the outer target is never used for fitting, selection, normalization, or
reference estimation.

The script writes one auditable JSON per outer fold.  It deliberately keeps
per-trial truth/prediction arrays so that the manuscript table can be rebuilt
from machine-readable evidence rather than copied from logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import config as cfg
from dataset import load_trials
from classical_features import build_subject_features, views
from model_names import MBDG_RS_ID, MBDG_RS_NAME


ALL_BLOCKS = tuple(range(1, int(cfg.RECORDED_BLOCKS) + 1))
METHODS = (
    "TD4-LDA",
    "TD4-linear-SVM",
    "AIRM-Tangent-LDA",
    "AIRM-MDM",
    "RandomForest",
    "ExtraTrees",
    MBDG_RS_NAME,
)
SVM_CANDIDATES = (0.1, 1.0, 10.0)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.%d" % os.getpid())
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mask(blocks, selected=ALL_BLOCKS):
    return np.isin(np.asarray(blocks), selected)


def _relative_groups(values, group_width):
    """Within-trial channel normalization used by the TD/RF baselines.

    The operation uses only the current trial's channels.  It therefore does
    not estimate a target-subject statistic and is applied identically to
    source and target trials.
    """
    values = np.asarray(values, dtype=np.float64)
    groups = values.reshape(len(values), -1, group_width)
    center = groups.mean(axis=2, keepdims=True)
    scale = np.maximum(groups.std(axis=2, keepdims=True), 1e-8)
    return ((groups - center) / scale).reshape(values.shape)


def _load_cache(cache_root):
    cache_root = Path(cache_root)
    cache = {}
    data_hashes = {}
    for subject in cfg.SUBJECTS:
        record = load_trials(subject)
        feature_path = cache_root / (subject + ".npz")
        facial = build_subject_features(record, feature_path)
        cache[subject] = {
            "record": record,
            "labels": np.asarray(record.labels, dtype=int),
            "blocks": np.asarray(record.groups, dtype=int),
            "td4": np.asarray(facial["td_relative"], dtype=np.float64),
            MBDG_RS_ID: np.asarray(views(facial)["filter_bank_dual_geometry_spectrum"], dtype=np.float64),
            "data_path": str(cfg.TRIAL_ROOT / (subject + "_trials.npz")),
        }
        data_hashes[subject] = sha256_file(cache[subject]["data_path"])
        print("loaded %s: %d trials" % (subject, len(record.labels)), flush=True)
    return cache, data_hashes


def _stack(cache, subjects, feature_name, blocks=ALL_BLOCKS):
    features, labels = [], []
    for subject in subjects:
        item = cache[subject]
        mask = _mask(item["blocks"], blocks)
        features.append(item[feature_name][mask])
        labels.append(item["labels"][mask])
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def _truth(cache, subject, blocks=ALL_BLOCKS):
    item = cache[subject]
    mask = _mask(item["blocks"], blocks)
    return item["labels"][mask], mask


def _standard_lda():
    return make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )


def _linear_svm(c_value):
    return make_pipeline(
        StandardScaler(),
        LinearSVC(C=float(c_value), dual=False, max_iter=10000,
                  random_state=int(cfg.DEFAULT_SEED)),
    )


def _fit_predict(estimator, train_x, train_y, test_x):
    estimator.fit(train_x, train_y)
    return estimator.predict(test_x)


def _summarize(truth, prediction):
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "n_trials": int(len(truth)),
        "truth": np.asarray(truth, dtype=int).tolist(),
        "prediction": np.asarray(prediction, dtype=int).tolist(),
    }


def _select_svm(cache, development):
    """Select C using only complete-source-subject inner LOSO."""
    candidates = []
    for c_value in SVM_CANDIDATES:
        inner_rows = []
        for validation_subject in development:
            inner_train = tuple(s for s in development if s != validation_subject)
            train_x, train_y = _stack(cache, inner_train, "td4", ALL_BLOCKS)
            val_truth, val_mask = _truth(cache, validation_subject, ALL_BLOCKS)
            val_prediction = _fit_predict(
                _linear_svm(c_value), train_x, train_y,
                cache[validation_subject]["td4"][val_mask],
            )
            inner_rows.append({
                "validation_subject": validation_subject,
                "fit_subjects": list(inner_train),
                "fit_blocks": list(ALL_BLOCKS),
                "validation_blocks": list(ALL_BLOCKS),
                "n_fit_trials": int(len(train_y)),
                "n_validation_trials": int(len(val_truth)),
                "accuracy": float(accuracy_score(val_truth, val_prediction)),
                "macro_f1": float(f1_score(val_truth, val_prediction, average="macro")),
            })
        values = np.asarray([row["accuracy"] for row in inner_rows], dtype=float)
        candidates.append({
            "C": float(c_value),
            "mean_inner_accuracy": float(values.mean()),
            "std_inner_accuracy": float(values.std(ddof=1)),
            "per_inner_subject": inner_rows,
        })
    selected = max(candidates, key=lambda row: (row["mean_inner_accuracy"], -row["C"]))
    return selected, candidates


def _sym_function(matrix, function, clip=True):
    values, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    if clip:
        values = np.maximum(values, 1e-10)
    return (vectors * function(values)) @ vectors.T


def _matrix_log(matrix):
    return _sym_function(matrix, np.log)


def _matrix_exp(matrix):
    # Tangent matrices are symmetric but need not be SPD.  Clipping their
    # negative eigenvalues would corrupt the Riemannian mean update.
    return _sym_function(matrix, np.exp, clip=False)


def _matrix_sqrt(matrix):
    return _sym_function(matrix, np.sqrt)


def _matrix_invsqrt(matrix):
    return _sym_function(matrix, lambda x: 1.0 / np.sqrt(x))


def _covariances(data, regularization=0.05):
    data = np.asarray(data, dtype=np.float64)
    centered = data - data.mean(axis=2, keepdims=True)
    covariance = centered @ np.swapaxes(centered, 1, 2) / (data.shape[2] - 1)
    trace = np.trace(covariance, axis1=1, axis2=2)
    covariance = covariance / np.maximum(trace[:, None, None], 1e-12) * data.shape[1]
    identity = np.eye(data.shape[1])[None]
    return (1.0 - regularization) * covariance + regularization * identity


def _riemannian_mean(matrices, max_iter=12, tolerance=1e-7):
    """Affine-invariant mean with a log-Euclidean initialization."""
    current = _matrix_exp(np.mean([_matrix_log(x) for x in matrices], axis=0))
    for _ in range(max_iter):
        root = _matrix_sqrt(current)
        invroot = _matrix_invsqrt(current)
        tangent = np.mean(
            [_matrix_log(invroot @ x @ invroot) for x in matrices], axis=0
        )
        norm = float(np.linalg.norm(tangent, ord="fro"))
        current = root @ _matrix_exp(tangent) @ root
        current = (current + current.T) * 0.5
        if norm < tolerance:
            break
    return current


def _tangent_features(matrices, reference):
    invroot = _matrix_invsqrt(reference)
    upper = np.triu_indices(reference.shape[0])
    off = upper[0] != upper[1]
    result = np.empty((len(matrices), len(upper[0])), dtype=np.float64)
    for index, matrix in enumerate(matrices):
        tangent = _matrix_log(invroot @ matrix @ invroot)
        vector = tangent[upper].copy()
        vector[off] *= np.sqrt(2.0)
        result[index] = vector
    return result


def _airm_distance(left, right):
    invroot = _matrix_invsqrt(left)
    return float(np.linalg.norm(_matrix_log(invroot @ right @ invroot), ord="fro"))


def _source_covariances(cache, subjects):
    arrays, labels = [], []
    for subject in subjects:
        record = cache[subject]["record"]
        arrays.append(_covariances(record.data))
        labels.append(cache[subject]["labels"])
    return np.concatenate(arrays), np.concatenate(labels)


def _run_target(target, cache, methods, job_id, code_hash, data_hashes, seed):
    development = tuple(subject for subject in cfg.SUBJECTS if subject != target)
    selection_records = {}
    selected_c = None
    if "TD4-linear-SVM" in methods:
        selected, candidates = _select_svm(cache, development)
        selected_c = float(selected["C"])
        selection_records["TD4-linear-SVM"] = {
            "selection_protocol": "inner_LOSO_over_11_source_subjects_all_30_blocks",
            "selected": selected,
            "candidates": candidates,
        }
    else:
        selection_records["TD4-linear-SVM"] = {"skipped": True}

    target_truth, target_mask = _truth(cache, target, ALL_BLOCKS)
    outputs = {}

    if "TD4-LDA" in methods:
        train_x, train_y = _stack(cache, development, "td4", ALL_BLOCKS)
        pred = _fit_predict(_standard_lda(), train_x, train_y,
                            cache[target]["td4"][target_mask])
        outputs["TD4-LDA"] = _summarize(target_truth, pred)

    if "TD4-linear-SVM" in methods:
        train_x, train_y = _stack(cache, development, "td4", ALL_BLOCKS)
        pred = _fit_predict(_linear_svm(selected_c), train_x, train_y,
                            cache[target]["td4"][target_mask])
        outputs["TD4-linear-SVM"] = _summarize(target_truth, pred)

    if MBDG_RS_NAME in methods:
        train_x, train_y = _stack(cache, development, MBDG_RS_ID, ALL_BLOCKS)
        pred = _fit_predict(_standard_lda(), train_x, train_y,
                            cache[target][MBDG_RS_ID][target_mask])
        outputs[MBDG_RS_NAME] = _summarize(target_truth, pred)

    if "RandomForest" in methods or "ExtraTrees" in methods:
        train_x, train_y = _stack(cache, development, "td4", ALL_BLOCKS)
        test_x = cache[target]["td4"][target_mask]
        if "RandomForest" in methods:
            model = RandomForestClassifier(
                n_estimators=300, max_features="sqrt", min_samples_leaf=1,
                class_weight=None, random_state=int(seed), n_jobs=8,
            )
            pred = _fit_predict(model, train_x, train_y, test_x)
            outputs["RandomForest"] = _summarize(target_truth, pred)
        if "ExtraTrees" in methods:
            model = ExtraTreesClassifier(
                n_estimators=300, max_features="sqrt", min_samples_leaf=1,
                class_weight=None, random_state=int(seed), n_jobs=8,
            )
            pred = _fit_predict(model, train_x, train_y, test_x)
            outputs["ExtraTrees"] = _summarize(target_truth, pred)

    if "AIRM-Tangent-LDA" in methods or "AIRM-MDM" in methods:
        train_cov, train_y = _source_covariances(cache, development)
        target_cov = _covariances(cache[target]["record"].data)[target_mask]
        reference = _riemannian_mean(train_cov)
        if "AIRM-Tangent-LDA" in methods:
            train_x = _tangent_features(train_cov, reference)
            test_x = _tangent_features(target_cov, reference)
            pred = _fit_predict(_standard_lda(), train_x, train_y, test_x)
            outputs["AIRM-Tangent-LDA"] = _summarize(target_truth, pred)
        if "AIRM-MDM" in methods:
            class_means = {
                label: _riemannian_mean(train_cov[train_y == label])
                for label in cfg.LABEL_IDS
            }
            distances = np.asarray([
                [_airm_distance(class_means[label], trial)
                 for label in cfg.LABEL_IDS]
                for trial in target_cov
            ])
            pred = np.asarray(cfg.LABEL_IDS)[distances.argmin(axis=1)]
            outputs["AIRM-MDM"] = _summarize(target_truth, pred)

    row = {
        "schema_version": 2,
        "experiment": "faceemg11_classical_baselines_loso",
        "target_subject": target,
        "fit_subjects": list(development),
        "fit_blocks": list(ALL_BLOCKS),
        "validation_subjects": list(development),
        "validation_blocks": list(ALL_BLOCKS),
        "test_subject": target,
        "test_blocks": list(ALL_BLOCKS),
        "trial_counts": {
            "source_subjects": len(development),
            "source_trials": int(330 * len(development)),
            "target_test_trials": int(len(target_truth)),
            "inner_fit_trials": int(330 * (len(development) - 1)),
            "inner_validation_trials": int(len(target_truth)),
        },
        "target_samples_used_for_fit_or_selection": 0,
        "target_derived_statistics": [],
        "methods": outputs,
        "selection_records": selection_records,
        "protocol": {
            "outer": "target subject held out; all 30 blocks tested",
            "inner": "complete subject LOSO over 11 source subjects; all 30 blocks",
            "n_outer_folds": len(cfg.SUBJECTS),
            "n_classes": len(cfg.LABEL_IDS),
            "window_seconds": 1.5,
            "target_labels_seen": False,
            "target_unlabeled_global_statistics_seen": False,
        },
        "model_config": {
            "TD4-LDA": "within-trial channel-relative Hudgins TD features + shrinkage LDA",
            "TD4-linear-SVM": "within-trial channel-relative Hudgins TD features + LinearSVC; C inner-selected",
            "AIRM-Tangent-LDA": "source-only affine-invariant mean reference + tangent-space shrinkage LDA",
            "AIRM-MDM": "source-only class Riemannian means + affine-invariant minimum distance",
            "RandomForest": "TD4-relative features; 300 trees; sqrt features",
            "ExtraTrees": "TD4-relative features; 300 trees; sqrt features",
            MBDG_RS_NAME: "fixed 1.5-s five-band covariance/correlation dual geometry + spectrum + shrinkage LDA",
        },
        "seed": int(seed),
        "python": sys.version,
        "platform": platform.platform(),
        "code_sha256": code_hash,
        "data_sha256": data_hashes,
        "created_unix": time.time(),
    }
    return row


def _aggregate(output_dir, aggregate_path, required_methods):
    output_dir = Path(output_dir)
    paths = sorted(output_dir.glob("outer_sub-*.json"))
    if len(paths) != len(cfg.SUBJECTS):
        raise RuntimeError("expected %d outer JSON files, found %d" %
                           (len(cfg.SUBJECTS), len(paths)))
    folds = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    targets = [fold.get("target_subject") for fold in folds]
    if len(set(targets)) != len(cfg.SUBJECTS) or set(targets) != set(cfg.SUBJECTS):
        raise RuntimeError("outer folds must contain each subject exactly once; got %r" % targets)
    for method in required_methods:
        count = sum(method in fold.get("methods", {}) for fold in folds)
        if count != len(cfg.SUBJECTS):
            raise RuntimeError("method %s appears in %d/%d folds" %
                               (method, count, len(cfg.SUBJECTS)))
    summary = {}
    for method in required_methods:
        rows = []
        for fold in folds:
            if method not in fold["methods"]:
                continue
            metric = fold["methods"][method]
            rows.append({
                "subject": fold["target_subject"],
                "accuracy": metric["accuracy"],
                "macro_f1": metric["macro_f1"],
                "n_trials": metric["n_trials"],
            })
        if not rows:
            continue
        acc = np.asarray([row["accuracy"] for row in rows], dtype=float)
        f1 = np.asarray([row["macro_f1"] for row in rows], dtype=float)
        summary[method] = {
            "mean_subject_accuracy": float(acc.mean()),
            "std_subject_accuracy": float(acc.std(ddof=1)),
            "mean_subject_macro_f1": float(f1.mean()),
            "std_subject_macro_f1": float(f1.std(ddof=1)),
            "per_subject": rows,
        }
    payload = {
        "schema_version": 2,
        "experiment": "faceemg11_classical_baselines_loso",
        "required_methods": list(required_methods),
        "protocol": folds[0]["protocol"],
        "outer_fold_files": [str(path) for path in paths],
        "summary": summary,
        "folds": folds,
    }
    atomic_json(aggregate_path, payload)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _self_test():
    probe = np.diag(np.asarray([-2.0, 1.0]))
    expected = np.diag(np.exp(np.asarray([-2.0, 1.0])))
    actual = _matrix_exp(probe)
    if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
        raise AssertionError("matrix_exp self-test failed: %r != %r" %
                             (actual, expected))
    print("matrix_exp self-test passed", flush=True)


def main():
    _self_test()
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--output-dir", default=str(cfg.RESULT_ROOT / "classical"))
    parser.add_argument("--cache-dir", default=str(cfg.RESULT_ROOT / "classical_feature_cache"))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--aggregate-output", default=str(cfg.RESULT_ROOT / "classical" / "aggregate.json"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--required-methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=int(cfg.DEFAULT_SEED))
    args = parser.parse_args()
    if args.aggregate:
        _aggregate(args.output_dir, args.aggregate_output, tuple(args.required_methods))
        return
    if args.target_index is None or not (0 <= args.target_index < len(cfg.SUBJECTS)):
        raise ValueError("--target-index must be in [0, %d]" % (len(cfg.SUBJECTS) - 1))
    started = time.time()
    cache, data_hashes = _load_cache(args.cache_dir)
    code_hash = sha256_file(Path(__file__))
    target = cfg.SUBJECTS[args.target_index]
    row = _run_target(
        target, cache, tuple(args.methods),
        os.environ.get("SLURM_JOB_ID", "local"), code_hash, data_hashes, args.seed,
    )
    row["runtime_seconds"] = float(time.time() - started)
    output_path = Path(args.output_dir) / ("outer_%s.json" % target)
    atomic_json(output_path, row)
    print(json.dumps({
        "target": target,
        "runtime_seconds": row["runtime_seconds"],
        "methods": {name: {"accuracy": val["accuracy"], "macro_f1": val["macro_f1"]}
                    for name, val in row["methods"].items()},
        "output": str(output_path),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

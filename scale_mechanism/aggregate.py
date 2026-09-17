#!/usr/bin/env python3
"""Unified participant-level aggregation for channel-gain robustness.

The deterministic methods are averaged over 20 gains within each held-out
participant.  Facial CNN is averaged over 20 gains within each training seed,
then over five seeds within participant.  Bootstrap resampling always uses
only the resulting twelve participant-level values.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np

# ``aggregate.py`` runs from a subdirectory, so make the repository root
# importable without requiring package installation.
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_names import MBDG_RS_ID, MBDG_RS_NAME


OUTPUT_TOKEN = "scale_mechanism"
EXPECTED_FOLDS = 12
EXPECTED_TRIALS = 330
EXPECTED_DRAWS = 20
EXPECTED_SIGMAS = ("0.25", "0.5", "0.75")
BOOTSTRAP_DRAWS = 100000
GAIN_SEED_BASE = 2026082400
METHODS = (
    "five_band_covariance_relative_spectrum",
    "five_band_correlation_relative_spectrum",
    MBDG_RS_ID,
    "broadband_dual_geometry_relative_spectrum",
    "AIRM-Tangent-LDA",
    "TD4-LDA",
)
FACIAL_CNN_ID = "facial_cnn"
FACIAL_CNN_LEGACY_METHOD = "Facial-CNN"
FACIAL_CNN_SEEDS = (11, 23, 37, 53, 71)
METRICS = ("accuracy", "macro_f1")

# Frozen before result inspection.  A positive difference means the method on
# the right has the smaller clean-to-perturbed drop (greater scale robustness).
PRIMARY_COMPARISONS = (
    (
        "covariance_drop_minus_correlation_drop",
        "five_band_covariance_relative_spectrum",
        "five_band_correlation_relative_spectrum",
    ),
    ("AIRM_drop_minus_MBDG-RS_drop", "AIRM-Tangent-LDA", MBDG_RS_ID),
    ("TD4_drop_minus_MBDG-RS_drop", "TD4-LDA", MBDG_RS_ID),
)


def require_scale_path(path, purpose):
    resolved = Path(path).expanduser().resolve()
    if OUTPUT_TOKEN.lower() not in str(resolved).lower():
        raise ValueError(
            "%s path must contain %r as an overwrite guard: %s"
            % (purpose, OUTPUT_TOKEN, resolved)
        )
    return resolved


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value):
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value)).tobytes(order="C")
    ).hexdigest()


def atomic_json(path, payload, overwrite=False):
    path = require_scale_path(path, "aggregate output")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing aggregate; pass --overwrite: %s" % path
        )
    temporary = path.with_suffix(path.suffix + ".tmp.%d" % os.getpid())
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temporary), str(path))


def accuracy(truth, prediction):
    return float(np.mean(np.asarray(truth) == np.asarray(prediction)))


def macro_f1(truth, prediction):
    truth = np.asarray(truth, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    values = []
    for label in sorted(np.unique(truth).tolist()):
        tp = int(np.sum((truth == label) & (prediction == label)))
        fp = int(np.sum((truth != label) & (prediction == label)))
        fn = int(np.sum((truth == label) & (prediction != label)))
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else (2.0 * tp / denominator))
    return float(np.mean(values))


def validate_metric_row(row, truth, context):
    prediction = np.asarray(row.get("prediction", []), dtype=int)
    if len(prediction) != EXPECTED_TRIALS:
        raise ValueError("%s: prediction length is not 330" % context)
    if int(row.get("n_test_trials", -1)) != EXPECTED_TRIALS:
        raise ValueError("%s: n_test_trials is not 330" % context)
    observed = {
        "accuracy": accuracy(truth, prediction),
        "macro_f1": macro_f1(truth, prediction),
    }
    for metric in METRICS:
        if not np.isclose(
            float(row.get(metric, np.nan)), observed[metric], rtol=0.0, atol=1e-12
        ):
            raise ValueError("%s: %s cannot be reproduced" % (context, metric))
    if row.get("truth_ref") != "canonical_target_trials.truth":
        raise ValueError("%s: invalid truth reference" % context)
    return observed


def validate_fold(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    context = Path(path).name
    if payload.get("schema_version") != 2:
        raise ValueError("%s: unexpected schema" % context)
    if payload.get("experiment") not in {
        "faceemg11_channel_gain_robustness_loso",
        "faceemg11_mbdg_rs_channel_gain_robustness_loso",
    }:
        raise ValueError("%s: unexpected experiment" % context)
    if payload.get("smoke"):
        raise ValueError("%s: smoke output cannot enter full aggregation" % context)
    protocol = payload.get("protocol", {})
    fold_index = int(protocol.get("outer_fold_index", -1))
    if Path(path).stem != "fold_%02d" % fold_index:
        raise ValueError("%s: filename does not match outer fold index" % context)
    target = protocol.get("outer_target")
    sources = protocol.get("source_subjects", [])
    if len(sources) != 11 or len(set(sources)) != 11 or target in sources:
        raise ValueError("%s: outer source/target partition is invalid" % context)
    if protocol.get("target_samples_used_for_fit_or_selection") != 0:
        raise ValueError("%s: target leakage flag" % context)
    if protocol.get("target_derived_statistics") != []:
        raise ValueError("%s: target-derived-statistics field is not empty" % context)
    if payload.get("target_derived_statistics") != []:
        raise ValueError("%s: top-level target-derived-statistics field is not empty" % context)
    if protocol.get("source_blocks") != list(range(1, 31)):
        raise ValueError("%s: source blocks are not 1..30" % context)
    if protocol.get("target_blocks") != list(range(1, 31)):
        raise ValueError("%s: target blocks are not 1..30" % context)
    if protocol.get("source_fit_trials") != 3630:
        raise ValueError("%s: source trial count is not 3630" % context)
    if protocol.get("target_evaluation_trials") != EXPECTED_TRIALS:
        raise ValueError("%s: target trial count is not 330" % context)
    if protocol.get("same_draw_shared_across_all_methods") is not True:
        raise ValueError("%s: gains are not declared shared across methods" % context)
    if protocol.get("same_gain_reused_for_all_target_trials") is not True:
        raise ValueError("%s: gains are not declared fixed over target trials" % context)
    if tuple(payload.get("methods", [])) != METHODS:
        raise ValueError("%s: method set/order mismatch" % context)

    canonical = payload.get("canonical_target_trials", {})
    truth = np.asarray(canonical.get("truth", []), dtype=int)
    blocks = np.asarray(canonical.get("blocks", []), dtype=int)
    indices = np.asarray(canonical.get("trial_index", []), dtype=int)
    if len(truth) != EXPECTED_TRIALS or len(blocks) != EXPECTED_TRIALS:
        raise ValueError("%s: canonical target vectors are not length 330" % context)
    if not np.array_equal(indices, np.arange(EXPECTED_TRIALS)):
        raise ValueError("%s: canonical trial indices mismatch" % context)
    labels = sorted(np.unique(truth).tolist())
    if len(labels) != 11 or sorted(np.unique(blocks).tolist()) != list(range(1, 31)):
        raise ValueError("%s: target does not have 30 blocks and 11 labels" % context)
    for block in range(1, 31):
        for label in labels:
            if int(np.sum((blocks == block) & (truth == label))) != 1:
                raise ValueError("%s: target block-label grid is incomplete" % context)

    clean_rows = payload.get("clean", {}).get("methods", {})
    if set(clean_rows) != set(METHODS):
        raise ValueError("%s: clean method set mismatch" % context)
    clean_metrics = {
        method: validate_metric_row(
            clean_rows[method], truth, "%s clean %s" % (context, method)
        )
        for method in METHODS
    }

    sigmas = payload.get("scale_perturbation", {}).get("sigmas", {})
    if tuple(sigmas) != EXPECTED_SIGMAS:
        raise ValueError("%s: sigma set/order mismatch" % context)
    seeds = set()
    gain_hashes = set()
    for sigma in EXPECTED_SIGMAS:
        sigma_row = sigmas[sigma]
        expected_role = "primary" if sigma == "0.5" else "exploratory"
        if sigma_row.get("role") != expected_role:
            raise ValueError("%s sigma %s: severity role mismatch" % (context, sigma))
        draws = sigma_row.get("draws", [])
        if int(sigma_row.get("n_draws", -1)) != EXPECTED_DRAWS:
            raise ValueError("%s sigma %s: n_draws is not 20" % (context, sigma))
        if len(draws) != EXPECTED_DRAWS:
            raise ValueError("%s sigma %s: draw list is not length 20" % (context, sigma))
        if [draw.get("draw_index") for draw in draws] != list(range(EXPECTED_DRAWS)):
            raise ValueError("%s sigma %s: draw indices mismatch" % (context, sigma))
        for draw in draws:
            draw_context = "%s sigma %s draw %02d" % (
                context, sigma, int(draw["draw_index"])
            )
            gains = np.asarray(draw.get("gain_vector_float64", []), dtype=np.float64)
            if gains.shape != (20,) or not np.all(np.isfinite(gains)) or not np.all(gains > 0):
                raise ValueError("%s: invalid gain vector" % draw_context)
            if sha256_array(gains) != draw.get("gain_sha256_float64_c_order"):
                raise ValueError("%s: gain hash mismatch" % draw_context)
            if draw.get("reused_across_methods") is not True:
                raise ValueError("%s: gain not shared across methods" % draw_context)
            if draw.get("reused_across_target_trials") is not True:
                raise ValueError("%s: gain not fixed across trials" % draw_context)
            if draw.get("applied_to_target_trials") != EXPECTED_TRIALS:
                raise ValueError("%s: gain was not applied to 330 trials" % draw_context)
            seed = int(draw.get("seed"))
            expected_seed = (
                GAIN_SEED_BASE + fold_index * 10000
                + int(round(float(sigma) * 100.0)) * 100
                + int(draw["draw_index"])
            )
            if seed != expected_seed:
                raise ValueError("%s: deterministic seed mismatch" % draw_context)
            gain_hash = draw.get("gain_sha256_float64_c_order")
            if seed in seeds or gain_hash in gain_hashes:
                raise ValueError("%s: duplicate seed or gain within fold" % draw_context)
            seeds.add(seed)
            gain_hashes.add(gain_hash)
            invariance = draw.get("correlation_matrix_invariance", {})
            if invariance.get("passed") is not True:
                raise ValueError("%s: raw correlation invariance failed" % draw_context)
            if float(invariance.get("atol", np.nan)) != 1e-8:
                raise ValueError("%s: correlation invariance atol mismatch" % draw_context)
            if float(invariance.get("rtol", np.nan)) != 1e-7:
                raise ValueError("%s: correlation invariance rtol mismatch" % draw_context)
            if int(invariance.get("n_tolerance_violations", -1)) != 0:
                raise ValueError("%s: correlation invariance violations" % draw_context)
            feature_invariance = draw.get("correlation_feature_invariance", {})
            if feature_invariance.get("passed") is not True:
                raise ValueError("%s: correlation feature invariance failed" % draw_context)
            if float(feature_invariance.get("atol", np.nan)) != 1e-8:
                raise ValueError("%s: correlation feature invariance atol mismatch" % draw_context)
            if float(feature_invariance.get("rtol", np.nan)) != 1e-7:
                raise ValueError("%s: correlation feature invariance rtol mismatch" % draw_context)
            if int(feature_invariance.get("n_tolerance_violations", -1)) != 0:
                raise ValueError("%s: correlation feature invariance violations" % draw_context)
            method_rows = draw.get("methods", {})
            if set(method_rows) != set(METHODS):
                raise ValueError("%s: perturbed method set mismatch" % draw_context)
            for method in METHODS:
                observed = validate_metric_row(
                    method_rows[method], truth, "%s %s" % (draw_context, method)
                )
                for metric in METRICS:
                    key = "clean_to_perturbed_%s_drop" % metric
                    expected = clean_metrics[method][metric] - observed[metric]
                    if not np.isclose(
                        float(method_rows[method].get(key, np.nan)), expected,
                        rtol=0.0, atol=1e-12,
                    ):
                        raise ValueError("%s: stored %s is wrong" % (draw_context, key))
    return payload


def stable_seed(label):
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def participant_bootstrap(values, label):
    values = np.asarray(values, dtype=float)
    seed = stable_seed(label)
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    estimates = values[indices].mean(axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {
        "resamples": BOOTSTRAP_DRAWS,
        "seed": int(seed),
        "unit": "outer participant",
        "ci_percentile": 95.0,
        "low": float(low),
        "high": float(high),
        "low_pp": float(low * 100.0),
        "high_pp": float(high * 100.0),
    }


def exact_sign_flip(values):
    values = np.asarray(values, dtype=float)
    observed = float(values.mean())
    signed_means = np.asarray([
        np.mean(values * np.asarray(signs, dtype=float))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ])
    threshold = abs(observed) - 1e-15
    p_value = float(np.mean(np.abs(signed_means) >= threshold))
    return {
        "test": "exact two-sided participant sign-flip of mean",
        "n_participants": int(len(values)),
        "enumerations": int(2 ** len(values)),
        "observed_mean": observed,
        "observed_mean_pp": float(observed * 100.0),
        "p_value_two_sided": p_value,
    }


def summarize_participants(values, label):
    values = np.asarray(values, dtype=float)
    return {
        "n_participants": int(len(values)),
        "participant_values": values.tolist(),
        "participant_values_pp": (values * 100.0).tolist(),
        "mean": float(values.mean()),
        "mean_pp": float(values.mean() * 100.0),
        "sample_sd": float(values.std(ddof=1)),
        "sample_sd_pp": float(values.std(ddof=1) * 100.0),
        "participant_bootstrap_95_ci": participant_bootstrap(values, label),
        "exact_sign_flip": exact_sign_flip(values),
    }


def holm_adjust(named_p_values):
    ordered = sorted(named_p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, (total - rank) * float(p_value))
        adjusted[name] = min(1.0, running)
    return adjusted


def aggregate(folds, fold_paths):
    targets = [fold["protocol"]["outer_target"] for fold in folds]
    if len(set(targets)) != EXPECTED_FOLDS:
        raise ValueError("outer targets are not unique")
    if sorted(fold["protocol"]["outer_fold_index"] for fold in folds) != list(range(12)):
        raise ValueError("outer fold indices are not exactly 0..11")

    hash_keys = ("script_sha256", "config_sha256", "dataset_sha256")
    common_hashes = {}
    for key in hash_keys:
        values = {fold["hashes"][key] for fold in folds}
        if len(values) != 1:
            raise ValueError("folds mix different %s values" % key)
        common_hashes[key] = next(iter(values))
    input_manifests = {
        json.dumps(fold["hashes"]["input_sha256"], sort_keys=True) for fold in folds
    }
    if len(input_manifests) != 1:
        raise ValueError("folds mix different input data manifests")

    # First average clean-to-perturbed drops across the 20 draws inside each
    # participant.  Only these twelve participant values enter inference.
    participant_drops = {
        sigma: {metric: {method: [] for method in METHODS} for metric in METRICS}
        for sigma in EXPECTED_SIGMAS
    }
    per_participant = {sigma: [] for sigma in EXPECTED_SIGMAS}
    for fold in folds:
        target = fold["protocol"]["outer_target"]
        clean = fold["clean"]["methods"]
        for sigma in EXPECTED_SIGMAS:
            draws = fold["scale_perturbation"]["sigmas"][sigma]["draws"]
            subject_row = {"participant": target, "methods": {}}
            for method in METHODS:
                subject_row["methods"][method] = {}
                for metric in METRICS:
                    draw_drops = np.asarray([
                        float(clean[method][metric])
                        - float(draw["methods"][method][metric])
                        for draw in draws
                    ])
                    mean_drop = float(draw_drops.mean())
                    participant_drops[sigma][metric][method].append(mean_drop)
                    subject_row["methods"][method][metric] = {
                        "clean": float(clean[method][metric]),
                        "draw_drops": draw_drops.tolist(),
                        "mean_draw_drop": mean_drop,
                        "mean_draw_drop_pp": float(mean_drop * 100.0),
                    }
            per_participant[sigma].append(subject_row)

    summaries = {}
    for sigma in EXPECTED_SIGMAS:
        summaries[sigma] = {
            "role": "primary" if sigma == "0.5" else "exploratory",
            "aggregation_order": "mean 20 draws within participant, then summarize 12 participants",
            "per_participant": per_participant[sigma],
            "metrics": {},
        }
        for metric in METRICS:
            summaries[sigma]["metrics"][metric] = {}
            for method in METHODS:
                label = "drop|%s|%s|%s" % (sigma, metric, method)
                summaries[sigma]["metrics"][metric][method] = summarize_participants(
                    participant_drops[sigma][metric][method], label
                )

    primary = {metric: {"comparisons": {}} for metric in METRICS}
    for metric in METRICS:
        raw_p = {}
        for name, left, right in PRIMARY_COMPARISONS:
            values = (
                np.asarray(participant_drops["0.5"][metric][left], dtype=float)
                - np.asarray(participant_drops["0.5"][metric][right], dtype=float)
            )
            row = summarize_participants(
                values, "comparison|0.5|%s|%s" % (metric, name)
            )
            row.update({
                "left_drop_method": left,
                "right_drop_method": right,
                "estimand": "left clean-to-perturbed drop minus right clean-to-perturbed drop",
                "positive_value_interpretation": "right method has smaller drop",
            })
            primary[metric]["comparisons"][name] = row
            raw_p[name] = row["exact_sign_flip"]["p_value_two_sided"]
        adjusted = holm_adjust(raw_p)
        for name, value in adjusted.items():
            primary[metric]["comparisons"][name]["holm_adjusted_p"] = float(value)
        primary[metric]["holm_family"] = {
            "severity_sigma": 0.5,
            "metric": metric,
            "n_predefined_comparisons": len(PRIMARY_COMPARISONS),
            "procedure": "Holm step-down adjustment of exact two-sided participant sign-flip p-values",
            "comparison_names": [item[0] for item in PRIMARY_COMPARISONS],
        }

    return {
        "schema_version": 2,
        "experiment": "faceemg11_mbdg_rs_channel_gain_robustness_aggregate",
        "n_outer_participants": EXPECTED_FOLDS,
        "outer_targets": targets,
        "methods": list(METHODS),
        "method_display_names": {MBDG_RS_ID: MBDG_RS_NAME},
        "severity_roles": {"0.5": "primary", "0.25": "exploratory", "0.75": "exploratory"},
        "inference_unit": "outer participant",
        "draw_aggregation": "within each participant, arithmetic mean of 20 clean-to-perturbed drops",
        "bootstrap_resamples": BOOTSTRAP_DRAWS,
        "summaries": summaries,
        "primary_sigma_0p5_predefined_holm": primary,
        "common_hashes": common_hashes,
        "input_sha256": json.loads(next(iter(input_manifests))),
        "source_fold_files": [str(path) for path in fold_paths],
        "source_fold_sha256": {
            Path(path).name: sha256_file(path) for path in fold_paths
        },
    }


def facial_method_rows(payload, container, context):
    """Return a canonical Facial CNN method mapping from new or old files."""
    rows = payload.get(container, {}).get("methods", {})
    if FACIAL_CNN_LEGACY_METHOD in rows:
        if FACIAL_CNN_ID in rows:
            raise ValueError("%s: duplicate Facial CNN method keys" % context)
        rows[FACIAL_CNN_ID] = rows.pop(FACIAL_CNN_LEGACY_METHOD)
    if set(rows) != {FACIAL_CNN_ID}:
        raise ValueError("%s: Facial CNN method set mismatch" % context)
    payload[container]["methods"] = rows
    return rows


def validate_facial_cnn_run(path, expected_fold, expected_seed, reference_fold):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    context = Path(path).name
    if payload.get("schema_version") not in (1, 2):
        raise ValueError("%s: unexpected Facial CNN schema" % context)
    if payload.get("experiment") not in {
        "faceemg11_facial1dcnn_channel_gain_robustness_loso",
        "faceemg11_facial_cnn_channel_gain_robustness_loso",
    }:
        raise ValueError("%s: unexpected Facial CNN experiment" % context)
    if payload.get("smoke"):
        raise ValueError("%s: smoke output cannot enter aggregation" % context)
    protocol = payload.get("protocol", {})
    if int(protocol.get("outer_fold_index", -1)) != int(expected_fold):
        raise ValueError("%s: Facial CNN fold mismatch" % context)
    if int(payload.get("training", {}).get("seed", -1)) != int(expected_seed):
        raise ValueError("%s: Facial CNN training seed mismatch" % context)
    if protocol.get("outer_target") != reference_fold["protocol"]["outer_target"]:
        raise ValueError("%s: target differs from deterministic fold" % context)
    if protocol.get("source_subjects") != reference_fold["protocol"]["source_subjects"]:
        raise ValueError("%s: source participants differ from deterministic fold" % context)
    if int(protocol.get("target_samples_used_for_fit_or_selection", -1)) != 0:
        raise ValueError("%s: target leakage flag" % context)
    if payload.get("target_derived_statistics") != []:
        raise ValueError("%s: target-derived statistics are not empty" % context)

    canonical = payload.get("canonical_target_trials", {})
    reference_canonical = reference_fold["canonical_target_trials"]
    for key in ("trial_index", "blocks", "truth"):
        if canonical.get(key) != reference_canonical.get(key):
            raise ValueError("%s: canonical target %s mismatch" % (context, key))
    truth = np.asarray(canonical["truth"], dtype=int)
    clean = facial_method_rows(payload, "clean", context)[FACIAL_CNN_ID]
    clean_metrics = validate_metric_row(clean, truth, context + " clean")

    sigmas = payload.get("scale_perturbation", {}).get("sigmas", {})
    if tuple(sigmas) != EXPECTED_SIGMAS:
        raise ValueError("%s: Facial CNN sigma set/order mismatch" % context)
    reference_sigmas = reference_fold["scale_perturbation"]["sigmas"]
    for sigma in EXPECTED_SIGMAS:
        draws = sigmas[sigma].get("draws", [])
        reference_draws = reference_sigmas[sigma]["draws"]
        if len(draws) != EXPECTED_DRAWS:
            raise ValueError("%s sigma %s: expected 20 draws" % (context, sigma))
        for draw_index, (draw, reference_draw) in enumerate(
            zip(draws, reference_draws)
        ):
            draw_context = "%s sigma %s draw %02d" % (
                context, sigma, draw_index
            )
            if int(draw.get("draw_index", -1)) != draw_index:
                raise ValueError("%s: draw index mismatch" % draw_context)
            if int(draw.get("seed", -1)) != int(reference_draw["seed"]):
                raise ValueError("%s: gain seed differs from deterministic fold" % draw_context)
            gains = np.asarray(draw.get("gain_vector_float64", []), dtype=np.float64)
            reference_gains = np.asarray(
                reference_draw.get("gain_vector_float64", []), dtype=np.float64
            )
            if gains.shape != (20,) or not np.array_equal(gains, reference_gains):
                raise ValueError("%s: gain vector differs from deterministic fold" % draw_context)
            if sha256_array(gains) != draw.get("gain_sha256_float64_c_order"):
                raise ValueError("%s: gain hash mismatch" % draw_context)
            method_rows = draw.get("methods", {})
            if FACIAL_CNN_LEGACY_METHOD in method_rows:
                method_rows[FACIAL_CNN_ID] = method_rows.pop(
                    FACIAL_CNN_LEGACY_METHOD
                )
            if set(method_rows) != {FACIAL_CNN_ID}:
                raise ValueError("%s: Facial CNN draw method mismatch" % draw_context)
            row = method_rows[FACIAL_CNN_ID]
            observed = validate_metric_row(row, truth, draw_context)
            for metric in METRICS:
                key = "clean_to_perturbed_%s_drop" % metric
                expected = clean_metrics[metric] - observed[metric]
                if not np.isclose(
                    float(row.get(key, np.nan)), expected, rtol=0.0, atol=1e-12
                ):
                    raise ValueError("%s: stored %s is wrong" % (draw_context, key))
    return payload


def find_facial_cnn_paths(input_dir):
    """Resolve exactly one new or legacy file for every fold/seed pair."""
    input_dir = require_scale_path(input_dir, "Facial CNN fold input")
    grouped = []
    paths = []
    for fold_index in range(EXPECTED_FOLDS):
        fold_paths = []
        for seed in FACIAL_CNN_SEEDS:
            candidates = [
                input_dir / ("facial_cnn__fold_%02d__seed%d.json" % (fold_index, seed)),
                input_dir / ("facial1dcnn__fold_%02d__seed%d.json" % (fold_index, seed)),
            ]
            existing = [path for path in candidates if path.is_file()]
            if len(existing) != 1:
                raise FileNotFoundError(
                    "expected one Facial CNN output for fold %02d seed %d; found %r"
                    % (fold_index, seed, [str(path) for path in existing])
                )
            fold_paths.append(existing[0])
            paths.append(existing[0])
        grouped.append(fold_paths)
    return grouped, paths


def merge_facial_cnn(result, facial_runs, facial_paths):
    """Add seed-aware Facial CNN participant summaries to one aggregate."""
    participant_drops = {
        sigma: {metric: [] for metric in METRICS}
        for sigma in EXPECTED_SIGMAS
    }
    clean_values = {metric: [] for metric in METRICS}
    per_participant = {sigma: [] for sigma in EXPECTED_SIGMAS}

    for fold_index, runs in enumerate(facial_runs):
        target = runs[0]["protocol"]["outer_target"]
        seed_clean = {metric: [] for metric in METRICS}
        for run in runs:
            clean = run["clean"]["methods"][FACIAL_CNN_ID]
            for metric in METRICS:
                seed_clean[metric].append(float(clean[metric]))
        for metric in METRICS:
            clean_values[metric].append(float(np.mean(seed_clean[metric])))

        for sigma in EXPECTED_SIGMAS:
            subject_row = {
                "participant": target,
                "training_seeds": list(FACIAL_CNN_SEEDS),
                "metrics": {},
            }
            for metric in METRICS:
                seed_mean_drops = []
                seed_draw_drops = []
                for run in runs:
                    clean = float(run["clean"]["methods"][FACIAL_CNN_ID][metric])
                    drops = [
                        clean - float(draw["methods"][FACIAL_CNN_ID][metric])
                        for draw in run["scale_perturbation"]["sigmas"][sigma]["draws"]
                    ]
                    seed_draw_drops.append(drops)
                    seed_mean_drops.append(float(np.mean(drops)))
                participant_drop = float(np.mean(seed_mean_drops))
                participant_drops[sigma][metric].append(participant_drop)
                subject_row["metrics"][metric] = {
                    "clean_accuracy_or_f1_by_seed": seed_clean[metric],
                    "draw_drops_by_seed": seed_draw_drops,
                    "mean_20_draws_by_seed": seed_mean_drops,
                    "participant_mean_over_five_seeds": participant_drop,
                    "participant_mean_over_five_seeds_pp": 100.0 * participant_drop,
                }
            per_participant[sigma].append(subject_row)

    result["methods"].append(FACIAL_CNN_ID)
    result.setdefault("method_display_names", {})[FACIAL_CNN_ID] = "Facial CNN"
    result["facial_cnn_seed_aggregation"] = (
        "mean 20 gain draws within each training seed, then mean five seeds "
        "within participant; bootstrap the twelve participant values"
    )
    result["facial_cnn_training_seeds"] = list(FACIAL_CNN_SEEDS)
    result["facial_cnn_clean"] = {
        metric: summarize_participants(
            clean_values[metric], "facial_cnn|clean|%s" % metric
        )
        for metric in METRICS
    }
    result["facial_cnn_per_participant"] = per_participant

    for sigma in EXPECTED_SIGMAS:
        for metric in METRICS:
            result["summaries"][sigma]["metrics"][metric][FACIAL_CNN_ID] = (
                summarize_participants(
                    participant_drops[sigma][metric],
                    "drop|%s|%s|%s" % (sigma, metric, FACIAL_CNN_ID),
                )
            )

    comparisons = {}
    for metric in METRICS:
        mbdg_drop = np.asarray(
            result["summaries"]["0.5"]["metrics"][metric][MBDG_RS_ID][
                "participant_values"
            ],
            dtype=float,
        )
        facial_drop = np.asarray(
            participant_drops["0.5"][metric], dtype=float
        )
        difference = facial_drop - mbdg_drop
        row = summarize_participants(
            difference, "comparison|0.5|%s|facial_cnn_minus_mbdg_rs" % metric
        )
        row.update({
            "estimand": "Facial CNN drop minus MBDG-RS drop",
            "positive_value_interpretation": "MBDG-RS has the smaller drop",
            "severity_sigma": 0.5,
        })
        comparisons[metric] = row
    result["primary_sigma_0p5_facial_cnn_comparison"] = comparisons
    result["source_facial_cnn_files"] = [str(path) for path in facial_paths]
    result["source_facial_cnn_sha256"] = {
        Path(path).name: sha256_file(path) for path in facial_paths
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate deterministic and optional Facial CNN robustness runs."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--facial-cnn-input-dir", type=Path,
        help=(
            "directory containing 12 folds x 5 Facial CNN seeds; when omitted, "
            "only deterministic methods are aggregated"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    input_dir = require_scale_path(args.input_dir, "fold input")
    paths = sorted(input_dir.glob("fold_[0-9][0-9].json"))
    if len(paths) != EXPECTED_FOLDS:
        raise ValueError(
            "expected exactly 12 full fold JSON files, found %d" % len(paths)
        )
    folds = [validate_fold(path) for path in paths]
    payload = aggregate(folds, paths)
    if args.facial_cnn_input_dir is not None:
        grouped_paths, facial_paths = find_facial_cnn_paths(
            args.facial_cnn_input_dir
        )
        facial_runs = []
        for fold_index, fold_paths in enumerate(grouped_paths):
            facial_runs.append([
                validate_facial_cnn_run(path, fold_index, seed, folds[fold_index])
                for path, seed in zip(fold_paths, FACIAL_CNN_SEEDS)
            ])
        payload = merge_facial_cnn(payload, facial_runs, facial_paths)
    atomic_json(args.output, payload, overwrite=bool(args.overwrite))
    print(json.dumps({
        "n_outer_participants": payload["n_outer_participants"],
        "outer_targets": payload["outer_targets"],
        "bootstrap_resamples": payload["bootstrap_resamples"],
        "facial_cnn_included": args.facial_cnn_input_dir is not None,
        "output": str(args.output),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

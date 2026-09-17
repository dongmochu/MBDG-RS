#!/usr/bin/env python3
"""Unified positive channel-gain robustness runner for FaceEMG-11.

Each outer fold fits fixed LDA pipelines on all 30 blocks (330 trials each)
from eleven source participants.  The held-out participant is evaluated only;
no target sample or statistic enters feature fitting, scaling, model fitting,
selection, or the source-only AIRM reference.  Target perturbations use one
fixed positive diagonal 20-channel gain vector for all 330 target trials.  A
draw is shared by every method.

``--model deterministic`` fits the six fixed feature/LDA methods once per
outer fold. ``--model facial_cnn`` fits one Facial CNN training seed and uses
the same gain vectors. Full mode evaluates 20 deterministic draws at sigma
0.25, 0.5, and 0.75. Smoke mode evaluates sigma 0.5 with two draws. Outputs
and caches are restricted to paths containing ``scale_mechanism``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

# This runner lives in a subdirectory; use the repository root by default.
PROJECT_ROOT = Path(os.environ.get("FACEEMG_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg
from dataset import load_trials
from model_names import MBDG_RS_ID, MBDG_RS_NAME


ALL_BLOCKS = tuple(range(1, int(cfg.RECORDED_BLOCKS) + 1))
BANDS = ((2, 20), (20, 60), (60, 120), (120, 250), (250, 450))
WINDOW_MS = 1500
CHANNELS = 20
EXPECTED_TRIALS = 330
REGULARIZATION = 0.05
MBDG_RS_LOG_EIGEN_FLOOR = 1e-8
PRIMARY_SIGMAS = (0.25, 0.5, 0.75)
PRIMARY_DRAWS = 20
SMOKE_SIGMAS = (0.5,)
SMOKE_DRAWS = 2
GAIN_SEED_BASE = 2026082400
INVARIANCE_ATOL = 1e-8
INVARIANCE_RTOL = 1e-7
OUTPUT_TOKEN = "scale_mechanism"

METHODS = (
    "five_band_covariance_relative_spectrum",
    "five_band_correlation_relative_spectrum",
    MBDG_RS_ID,
    "broadband_dual_geometry_relative_spectrum",
    "AIRM-Tangent-LDA",
    "TD4-LDA",
)
RUN_MODELS = ("deterministic", "facial_cnn")
FACIAL_CNN_ID = "facial_cnn"
FACIAL_CNN_ARCHITECTURE = "facial1dcnn"
FINAL_TRAINING_SEEDS = (11, 23, 37, 53, 71)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value):
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def require_scale_path(path, purpose):
    resolved = Path(path).expanduser().resolve()
    if OUTPUT_TOKEN.lower() not in str(resolved).lower():
        raise ValueError(
            "%s path must contain %r as an overwrite guard: %s"
            % (purpose, OUTPUT_TOKEN, resolved)
        )
    return resolved


def atomic_json(path, payload, overwrite=False):
    path = require_scale_path(path, "output")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing output; pass --overwrite: %s" % path
        )
    temporary = path.with_suffix(path.suffix + ".tmp.%d" % os.getpid())
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temporary), str(path))


def td_features(data, threshold=1e-5):
    data = np.asarray(data, dtype=np.float64)
    mav = np.mean(np.abs(data), axis=2)
    wl = np.sum(np.abs(np.diff(data, axis=2)), axis=2)
    sign_change = np.diff(np.signbit(data), axis=2)
    zc_amp = np.abs(np.diff(data, axis=2)) > threshold
    zc = np.sum(sign_change & zc_amp, axis=2)
    slopes = np.diff(data, axis=2)
    slope_change = np.diff(np.signbit(slopes), axis=2)
    ssc_amp = (
        (np.abs(slopes[:, :, :-1]) > threshold)
        & (np.abs(slopes[:, :, 1:]) > threshold)
    )
    ssc = np.sum(slope_change & ssc_amp, axis=2)
    return np.concatenate([mav, wl, zc, ssc], axis=1)


def relative_groups(values, group_width):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1] % int(group_width):
        raise ValueError("feature width is not divisible by group width")
    groups = values.reshape(len(values), -1, int(group_width))
    center = groups.mean(axis=2, keepdims=True)
    scale = np.maximum(groups.std(axis=2, keepdims=True), 1e-8)
    return ((groups - center) / scale).reshape(values.shape)


def symmetric_function(matrix, function, floor=None):
    values, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    if floor is not None:
        values = np.maximum(values, floor)
    return (vectors * function(values)) @ vectors.T


def matrix_log(matrix, floor=1e-10):
    return symmetric_function(matrix, np.log, floor=floor)


def matrix_exp(matrix):
    return symmetric_function(matrix, np.exp)


def matrix_sqrt(matrix):
    return symmetric_function(matrix, np.sqrt, floor=1e-10)


def matrix_invsqrt(matrix):
    return symmetric_function(matrix, lambda x: 1.0 / np.sqrt(x), floor=1e-10)


def matrix_exp_diagnostic():
    probe = np.diag(np.asarray([-2.0, 1.0]))
    if not np.allclose(
        np.diag(matrix_exp(probe)), np.exp(np.asarray([-2.0, 1.0])),
        rtol=1e-12, atol=1e-12,
    ):
        raise AssertionError("matrix_exp clipped a negative eigenvalue")


def regularized_covariances(data):
    centered = data - data.mean(axis=2, keepdims=True)
    covariance = centered @ np.swapaxes(centered, 1, 2) / max(data.shape[2] - 1, 1)
    trace = np.trace(covariance, axis1=1, axis2=2)
    return covariance / np.maximum(trace[:, None, None], 1e-12) * data.shape[1]


def airm_covariances(data, regularization=REGULARIZATION):
    covariance = regularized_covariances(data)
    identity = np.eye(data.shape[1])[None]
    return (1.0 - regularization) * covariance + regularization * identity


def matrix_log_vector(
    matrices, regularization=REGULARIZATION,
    log_floor=MBDG_RS_LOG_EIGEN_FLOOR,
):
    matrices = np.asarray(matrices, dtype=np.float64)
    channels = matrices.shape[1]
    identity = np.eye(channels)
    upper = np.triu_indices(channels)
    off = upper[0] != upper[1]
    output = np.empty((len(matrices), len(upper[0])), dtype=np.float64)
    for index, matrix in enumerate(matrices):
        shrunk = (1.0 - regularization) * matrix + regularization * identity
        logged = matrix_log(shrunk, floor=log_floor)
        vector = logged[upper].copy()
        vector[off] *= np.sqrt(2.0)
        output[index] = vector
    return output


def correlation_matrices(data):
    """Scale-equivariant correlation as a Gram matrix of centered unit vectors."""
    data = np.asarray(data, dtype=np.float64)
    centered = data - data.mean(axis=2, keepdims=True)
    norm = np.linalg.norm(centered, axis=2, keepdims=True)
    if not np.all(np.isfinite(norm)) or np.any(norm <= 0.0):
        raise AssertionError("correlation geometry encountered a zero-variance channel")
    normalized = centered / norm
    correlation = normalized @ np.swapaxes(normalized, 1, 2)
    return (correlation + np.swapaxes(correlation, 1, 2)) * 0.5


def filter_bank(data, fs):
    result = []
    for low, high in BANDS:
        sos = butter(4, (low, high), btype="bandpass", fs=fs, output="sos")
        padlen = min(data.shape[2] - 1, 3 * (2 * len(sos) + 1))
        result.append(sosfiltfilt(sos, data, axis=2, padlen=padlen))
    return result


def bandpower(data, fs):
    spectrum = np.fft.rfft(data, axis=2)
    power = np.square(np.abs(spectrum)) / data.shape[2]
    frequencies = np.fft.rfftfreq(data.shape[2], 1.0 / fs)
    output = []
    for low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            raise ValueError("empty frequency band %s-%s" % (low, high))
        output.append(np.log(np.maximum(power[:, :, mask].mean(axis=2), 1e-20)))
    return np.concatenate(output, axis=1)


def extract_features(raw, fs):
    """Extract the fixed feature definitions used in this mechanism test."""
    samples = int(round(float(fs) * WINDOW_MS / 1000.0))
    data = np.asarray(raw[:, :CHANNELS, :samples], dtype=np.float64)
    if data.shape[1] != CHANNELS:
        raise AssertionError("expected exactly 20 channels")

    td = np.log1p(np.maximum(td_features(data), 0.0))
    td_rel = relative_groups(td, CHANNELS)

    spectral = bandpower(data, fs)
    spec_channel_rel = relative_groups(spectral, CHANNELS)
    cube = spectral.reshape(len(spectral), len(BANDS), CHANNELS).transpose(0, 2, 1)
    spec_band_rel = relative_groups(cube.reshape(len(cube), -1), len(BANDS))
    spec_band_rel = (
        spec_band_rel.reshape(len(cube), CHANNELS, len(BANDS))
        .transpose(0, 2, 1).reshape(len(cube), -1)
    )
    relative_spectrum = np.concatenate([spec_channel_rel, spec_band_rel], axis=1)

    covariance = regularized_covariances(data)
    covariance_tangent = matrix_log_vector(covariance)
    correlation = correlation_matrices(data)
    correlation_tangent = matrix_log_vector(correlation)
    fb_cov_parts = []
    fb_corr_parts = []
    fb_corr_matrix_parts = []
    for filtered in filter_bank(data, fs):
        fb_cov_parts.append(matrix_log_vector(regularized_covariances(filtered)))
        band_correlation = correlation_matrices(filtered)
        fb_corr_parts.append(matrix_log_vector(band_correlation))
        fb_corr_matrix_parts.append(band_correlation.reshape(len(filtered), -1))
    fb_cov = np.concatenate(fb_cov_parts, axis=1)
    fb_corr = np.concatenate(fb_corr_parts, axis=1)

    return {
        "td4": td_rel,
        "airm_covariance": airm_covariances(data),
        "fb_corr_component": fb_corr,
        "fb_corr_scale_equivariant_matrices": np.concatenate(
            fb_corr_matrix_parts, axis=1
        ),
        "methods": {
            "five_band_covariance_relative_spectrum": np.concatenate(
                [fb_cov, relative_spectrum], axis=1
            ),
            "five_band_correlation_relative_spectrum": np.concatenate(
                [fb_corr, relative_spectrum], axis=1
            ),
            MBDG_RS_ID: np.concatenate(
                [fb_cov, fb_corr, relative_spectrum], axis=1
            ),
            "broadband_dual_geometry_relative_spectrum": np.concatenate(
                [covariance_tangent, correlation_tangent, relative_spectrum], axis=1
            ),
        },
    }


def estimator():
    return make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )


def geometric_mean(matrices, max_iter=12, tolerance=1e-7):
    current = matrix_exp(np.mean([matrix_log(x) for x in matrices], axis=0))
    for _ in range(max_iter):
        root = matrix_sqrt(current)
        invroot = matrix_invsqrt(current)
        tangent = np.mean(
            [matrix_log(invroot @ x @ invroot) for x in matrices], axis=0
        )
        norm = float(np.linalg.norm(tangent, ord="fro"))
        current = root @ matrix_exp(tangent) @ root
        current = (current + current.T) * 0.5
        if norm < tolerance:
            break
    return current


def tangent_features(matrices, reference):
    invroot = matrix_invsqrt(reference)
    upper = np.triu_indices(reference.shape[0])
    off = upper[0] != upper[1]
    output = np.empty((len(matrices), len(upper[0])), dtype=np.float64)
    for index, matrix in enumerate(matrices):
        tangent = matrix_log(invroot @ matrix @ invroot)
        vector = tangent[upper].copy()
        vector[off] *= np.sqrt(2.0)
        output[index] = vector
    return output


def metric_row(truth, prediction):
    prediction = np.asarray(prediction, dtype=int)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "n_test_trials": int(len(truth)),
        "prediction": prediction.tolist(),
        "truth_ref": "canonical_target_trials.truth",
    }


def validate_records(records):
    if len(cfg.SUBJECTS) != 12:
        raise AssertionError("experiment requires exactly 12 participants")
    if int(cfg.CHANNELS) != CHANNELS:
        raise AssertionError("experiment requires exactly 20 channels")
    for subject, record in records.items():
        labels = np.asarray(record.labels, dtype=int)
        blocks = np.asarray(record.groups, dtype=int)
        if len(labels) != 330:
            raise AssertionError("%s does not contain 330 trials" % subject)
        if sorted(np.unique(blocks).tolist()) != list(ALL_BLOCKS):
            raise AssertionError("%s blocks are not exactly 1..30" % subject)
        for block in ALL_BLOCKS:
            for label in cfg.LABEL_IDS:
                if int(np.sum((blocks == block) & (labels == label))) != 1:
                    raise AssertionError(
                        "%s must contain one trial for block=%d label=%d"
                        % (subject, block, label)
                    )


def load_records(subjects):
    records = {subject: load_trials(subject) for subject in subjects}
    validate_records(records)
    return records


def build_clean_store(records, cache_root):
    """Cache target-independent fixed features; cache identity includes code/data."""
    cache_root = require_scale_path(cache_root, "cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    script_hash = sha256_file(Path(__file__))
    store = {}
    for subject, record in records.items():
        path = cache_root / ("%s_clean.npz" % subject)
        manifest = {
            "schema_version": 1,
            "subject": subject,
            "data_sha256": sha256_file(record.path),
            "script_sha256": script_hash,
            "config_sha256": sha256_file(Path(cfg.__file__)),
            "dataset_sha256": sha256_file(Path(sys.modules["dataset"].__file__)),
            "window_ms": WINDOW_MS,
            "channels": CHANNELS,
            "bands_hz": [list(value) for value in BANDS],
        }
        required_arrays = (
            "td4", "airm_covariance", "fb_corr_component",
        ) + METHODS[:4]
        try:
            with np.load(str(path), allow_pickle=False) as payload:
                cached = json.loads(str(payload["cache_manifest"].item()))
                if cached == manifest and all(
                    key in payload.files
                    for key in required_arrays + ("cache_manifest",)
                ):
                    store[subject] = {
                        key: payload[key] for key in required_arrays
                    }
                    continue
        except (FileNotFoundError, KeyError, ValueError, OSError):
            pass

        features = extract_features(record.data, record.fs)
        values = {
            "td4": features["td4"],
            "airm_covariance": features["airm_covariance"],
            "fb_corr_component": features["fb_corr_component"],
            "cache_manifest": np.asarray(json.dumps(manifest, sort_keys=True)),
        }
        values.update(features["methods"])
        temporary = path.with_name(path.name + ".tmp.%d.npz" % os.getpid())
        np.savez_compressed(str(temporary), **values)
        os.replace(str(temporary), str(path))
        store[subject] = {key: values[key] for key in required_arrays}
    return store


def stack_sources(records, store, sources, key):
    x = np.concatenate([store[subject][key] for subject in sources], axis=0)
    y = np.concatenate([records[subject].labels for subject in sources], axis=0)
    if len(y) != 3630:
        raise AssertionError("source fit must use 11 x 330 = 3630 trials")
    return x, y


def fit_source_models(records, store, sources):
    source_y = np.concatenate([records[subject].labels for subject in sources])
    models = {}
    for method in METHODS[:4]:
        source_x, checked_y = stack_sources(records, store, sources, method)
        if not np.array_equal(source_y, checked_y):
            raise AssertionError("source label order mismatch")
        models[method] = estimator().fit(source_x, source_y)

    source_td4, _ = stack_sources(records, store, sources, "td4")
    models["TD4-LDA"] = estimator().fit(source_td4, source_y)

    source_cov = np.concatenate(
        [store[subject]["airm_covariance"] for subject in sources], axis=0
    )
    reference = geometric_mean(source_cov)
    source_airm = tangent_features(source_cov, reference)
    models["AIRM-Tangent-LDA"] = estimator().fit(source_airm, source_y)
    return models, reference


def features_for_models(features, airm_reference):
    values = dict(features["methods"])
    values["TD4-LDA"] = features["td4"]
    values["AIRM-Tangent-LDA"] = tangent_features(
        features["airm_covariance"], airm_reference
    )
    if set(values) != set(METHODS):
        raise AssertionError("method feature set mismatch")
    return values


def deterministic_gain(fold_index, sigma, draw_index):
    # Encoding severity in the seed makes smoke draws 0/1 identical to the
    # corresponding first two production draws at sigma=0.5.
    sigma_code = int(round(float(sigma) * 100.0))
    seed = (
        GAIN_SEED_BASE + int(fold_index) * 10000
        + sigma_code * 100 + int(draw_index)
    )
    rng = np.random.RandomState(seed)
    gains = np.exp(rng.normal(0.0, float(sigma), size=CHANNELS)).astype(np.float64)
    return seed, gains


def invariance_row(clean, perturbed):
    clean = np.asarray(clean, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)
    absolute = np.abs(clean - perturbed)
    allowed = INVARIANCE_ATOL + INVARIANCE_RTOL * np.abs(clean)
    passed = bool(np.all(absolute <= allowed))
    row = {
        "feature": "five-band raw correlation matrices before fixed floors, regularization, and log mapping",
        "mathematical_scope": "positive diagonal channel gains with scale-equivariant zero-variance handling",
        "atol": float(INVARIANCE_ATOL),
        "rtol": float(INVARIANCE_RTOL),
        "maximum_absolute_error": float(absolute.max(initial=0.0)),
        "maximum_allowed_elementwise_error": float(allowed.max(initial=0.0)),
        "n_elements_checked": int(clean.size),
        "n_tolerance_violations": int(np.sum(absolute > allowed)),
        "passed": passed,
    }
    if not passed:
        raise AssertionError("raw five-band correlation invariance tolerance failed: %r" % row)
    return row


def feature_invariance_row(clean, perturbed):
    """Hard-gate invariance after identity regularization and log-svec mapping."""
    clean = np.asarray(clean, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)
    absolute = np.abs(perturbed - clean)
    allowed = INVARIANCE_ATOL + INVARIANCE_RTOL * np.abs(clean)
    passed = bool(np.all(absolute <= allowed))
    row = {
        "feature": "evaluated five-band correlation log-map component",
        "scope": "after identity regularization and log-svec mapping; before relative-spectrum concatenation",
        "atol": float(INVARIANCE_ATOL),
        "rtol": float(INVARIANCE_RTOL),
        "maximum_absolute_error": float(absolute.max(initial=0.0)),
        "maximum_allowed_elementwise_error": float(allowed.max(initial=0.0)),
        "n_elements_checked": int(clean.size),
        "n_tolerance_violations": int(np.sum(absolute > allowed)),
        "passed": passed,
    }
    if not passed:
        raise AssertionError("correlation log-map invariance tolerance failed: %r" % row)
    return row


def run_fold(fold_index, output, cache_root, smoke=False, overwrite=False):
    started = time.time()
    matrix_exp_diagnostic()
    target = cfg.SUBJECTS[int(fold_index)]
    sources = tuple(subject for subject in cfg.SUBJECTS if subject != target)
    if len(sources) != 11:
        raise AssertionError("outer fold must have exactly eleven source participants")

    # Do not even open the outer-target file until every scaler, LDA, and the
    # AIRM reference has been fitted from the eleven source participants.
    source_records = load_records(sources)
    store = build_clean_store(source_records, cache_root)
    models, airm_reference = fit_source_models(source_records, store, sources)
    target_record = load_trials(target)
    validate_records({target: target_record})
    records = dict(source_records)
    records[target] = target_record
    truth = np.asarray(target_record.labels, dtype=int)

    clean_features = extract_features(target_record.data, target_record.fs)
    clean_values = features_for_models(clean_features, airm_reference)
    clean_rows = {
        method: metric_row(truth, models[method].predict(clean_values[method]))
        for method in METHODS
    }

    sigmas = SMOKE_SIGMAS if smoke else PRIMARY_SIGMAS
    n_draws = SMOKE_DRAWS if smoke else PRIMARY_DRAWS
    sigma_rows = {}
    for sigma in sigmas:
        draws = []
        for draw_index in range(n_draws):
            seed, gains = deterministic_gain(fold_index, sigma, draw_index)
            perturbed_raw = target_record.data * gains[None, :, None]
            perturbed_features = extract_features(perturbed_raw, target_record.fs)
            corr_check = invariance_row(
                clean_features["fb_corr_scale_equivariant_matrices"],
                perturbed_features["fb_corr_scale_equivariant_matrices"],
            )
            corr_feature_check = feature_invariance_row(
                clean_features["fb_corr_component"],
                perturbed_features["fb_corr_component"],
            )
            perturbed_values = features_for_models(
                perturbed_features, airm_reference
            )
            methods = {
                method: metric_row(
                    truth, models[method].predict(perturbed_values[method])
                )
                for method in METHODS
            }
            for method in METHODS:
                methods[method]["clean_to_perturbed_accuracy_drop"] = float(
                    clean_rows[method]["accuracy"] - methods[method]["accuracy"]
                )
                methods[method]["clean_to_perturbed_macro_f1_drop"] = float(
                    clean_rows[method]["macro_f1"] - methods[method]["macro_f1"]
                )
            draws.append({
                "draw_index": int(draw_index),
                "seed": int(seed),
                "gain_distribution": "exp(Normal(0, sigma^2))",
                "gain_vector_float64": gains.tolist(),
                "gain_sha256_float64_c_order": sha256_array(gains),
                "gain_scope": "one fixed 20-channel vector shared by all methods and all 330 target trials",
                "applied_to_target_trials": 330,
                "reused_across_methods": True,
                "reused_across_target_trials": True,
                "correlation_matrix_invariance": corr_check,
                "correlation_feature_invariance": corr_feature_check,
                "methods": methods,
            })
        sigma_rows[format(float(sigma), "g")] = {
            "sigma": float(sigma),
            "role": "primary" if float(sigma) == 0.5 else "exploratory",
            "n_draws": int(n_draws),
            "draws": draws,
        }

    payload = {
        "schema_version": 2,
        "experiment": "faceemg11_mbdg_rs_channel_gain_robustness_loso",
        "protocol": {
            "outer": "participant-LOSO",
            "outer_fold_index": int(fold_index),
            "outer_target": target,
            "source_subjects": list(sources),
            "source_blocks": list(ALL_BLOCKS),
            "target_blocks": list(ALL_BLOCKS),
            "source_fit_trials": 3630,
            "target_evaluation_trials": 330,
            "target_samples_used_for_fit_or_selection": 0,
            "target_labels_used_for_fit_or_selection": False,
            "target_unlabeled_statistics_used_for_fit_or_selection": False,
            "target_derived_statistics": [],
            "perturbation": "target-only fixed positive diagonal channel gain",
            "same_draw_shared_across_all_methods": True,
            "same_gain_reused_for_all_target_trials": True,
            "window_ms": WINDOW_MS,
            "channels": CHANNELS,
            "bands_hz": [list(value) for value in BANDS],
            "regularization": REGULARIZATION,
            "mbdg_rs_log_eigen_floor": MBDG_RS_LOG_EIGEN_FLOOR,
            "method_display_names": {MBDG_RS_ID: MBDG_RS_NAME},
        },
        "target_derived_statistics": [],
        "methods": list(METHODS),
        "canonical_target_trials": {
            "trial_index": np.arange(len(truth), dtype=int).tolist(),
            "blocks": np.asarray(target_record.groups, dtype=int).tolist(),
            "truth": truth.tolist(),
        },
        "clean": {"methods": clean_rows},
        "scale_perturbation": {
            "gain_seed_base": GAIN_SEED_BASE,
            "sigmas": sigma_rows,
        },
        "smoke": bool(smoke),
        "counts": {
            "n_source_participants": 11,
            "n_target_participants": 1,
            "source_trials": 3630,
            "target_trials": 330,
        },
        "hashes": {
            "script_sha256": sha256_file(Path(__file__)),
            "config_sha256": sha256_file(Path(cfg.__file__)),
            "dataset_sha256": sha256_file(Path(sys.modules["dataset"].__file__)),
            "input_sha256": {
                subject: sha256_file(records[subject].path) for subject in cfg.SUBJECTS
            },
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "job": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "runtime_seconds": float(time.time() - started),
    }
    atomic_json(output, payload, overwrite=overwrite)
    return payload


def load_deep_stack():
    """Import the GPU stack only for ``--model facial_cnn``."""
    import dataset as dataset_module
    import run_deep_baselines as deep
    import torch

    if FACIAL_CNN_ARCHITECTURE not in deep.ARCHITECTURES:
        raise AssertionError("run_deep_baselines.py does not define facial1dcnn")
    if tuple(int(value) for value in deep.FINAL_SEEDS) != FINAL_TRAINING_SEEDS:
        raise AssertionError("deep final-training seed list has changed")
    return dataset_module, deep, torch


def validate_deep_target_grid(target_row):
    labels = np.asarray(target_row["labels"], dtype=int) + 1
    blocks = np.asarray(target_row["blocks"], dtype=int)
    if len(labels) != EXPECTED_TRIALS:
        raise AssertionError("target must contain exactly 330 trials")
    if sorted(np.unique(labels).tolist()) != list(cfg.LABEL_IDS):
        raise AssertionError("target labels do not match the configured classes")
    if sorted(np.unique(blocks).tolist()) != list(ALL_BLOCKS):
        raise AssertionError("target blocks must be exactly 1..30")
    for block in ALL_BLOCKS:
        for label in cfg.LABEL_IDS:
            if int(np.sum((blocks == block) & (labels == label))) != 1:
                raise AssertionError("target block/label grid is incomplete")
    return labels, blocks


def deep_metric_row(evaluation, expected_truth):
    truth = np.asarray(evaluation["truth"], dtype=int)
    prediction = np.asarray(evaluation["prediction"], dtype=int)
    if not np.array_equal(truth, np.asarray(expected_truth, dtype=int)):
        raise AssertionError("deep evaluation returned a non-canonical truth vector")
    return {
        "accuracy": float(evaluation["accuracy"]),
        "macro_f1": float(evaluation["macro_f1"]),
        "n_test_trials": EXPECTED_TRIALS,
        "prediction": prediction.tolist(),
        "truth_ref": "canonical_target_trials.truth",
    }


def load_gain_plan(fold_index, target, reference_scale_dir, smoke=False):
    """Generate gains and, when available, verify/reuse the LDA fold vectors."""
    sigmas = SMOKE_SIGMAS if smoke else PRIMARY_SIGMAS
    n_draws = SMOKE_DRAWS if smoke else PRIMARY_DRAWS
    generated = {
        format(float(sigma), "g"): [
            deterministic_gain(fold_index, sigma, draw_index)
            for draw_index in range(n_draws)
        ]
        for sigma in sigmas
    }
    reference_path = require_scale_path(
        Path(reference_scale_dir) / ("fold_%02d.json" % fold_index),
        "reference scale fold",
    )
    if not reference_path.is_file():
        return generated, None, None

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("experiment") not in {
        "faceemg11_channel_gain_robustness_loso",
        "faceemg11_mbdg_rs_channel_gain_robustness_loso",
    }:
        raise ValueError("reference scale fold has the wrong experiment id")
    protocol = reference.get("protocol", {})
    if int(protocol.get("outer_fold_index", -1)) != int(fold_index):
        raise ValueError("reference scale fold index mismatch")
    if protocol.get("outer_target") != target:
        raise ValueError("reference scale outer target mismatch")

    reference_sigmas = reference.get("scale_perturbation", {}).get("sigmas", {})
    verified = {}
    for sigma_key, expected_rows in generated.items():
        draws = reference_sigmas.get(sigma_key, {}).get("draws", [])
        if len(draws) < n_draws:
            raise ValueError("reference fold lacks draws for sigma %s" % sigma_key)
        verified_rows = []
        for draw_index, (expected_seed, expected_gains) in enumerate(expected_rows):
            draw = draws[draw_index]
            observed = np.asarray(
                draw.get("gain_vector_float64", []), dtype=np.float64
            )
            if int(draw.get("seed", -1)) != expected_seed:
                raise ValueError("reference gain seed mismatch")
            if observed.shape != (CHANNELS,) or not np.allclose(
                observed, expected_gains, rtol=1e-14, atol=0.0
            ):
                raise ValueError("reference gain vector mismatch")
            if sha256_array(observed) != draw.get(
                "gain_sha256_float64_c_order"
            ):
                raise ValueError("reference gain hash mismatch")
            verified_rows.append((expected_seed, observed.copy()))
        verified[sigma_key] = verified_rows
    return verified, reference_path, reference


def inspect_facial_cnn_run(fold_index, training_seed, selection_dir,
                           reference_scale_dir, max_epochs, smoke=False):
    _, deep, _ = load_deep_stack()
    target = cfg.SUBJECTS[int(fold_index)]
    sources = [subject for subject in cfg.SUBJECTS if subject != target]
    selection_path = Path(selection_dir).expanduser().resolve() / (
        FACIAL_CNN_ARCHITECTURE + "__" + target + ".json"
    )
    if not selection_path.is_file():
        raise FileNotFoundError(
            "missing Facial CNN inner-LOSO selection artifact: %s"
            % selection_path
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_epoch = deep.validate_selection(
        selection, target, FACIAL_CNN_ARCHITECTURE, sources, max_epochs
    )
    gain_plan, reference_path, reference = load_gain_plan(
        fold_index, target, reference_scale_dir, smoke=smoke
    )
    return {
        "target": target,
        "sources": sources,
        "selection_path": selection_path,
        "selected_epoch": selected_epoch,
        "gain_plan": gain_plan,
        "reference_path": reference_path,
        "reference": reference,
        "training_seed": int(training_seed),
    }


def run_facial_cnn_fold(fold_index, training_seed, output, selection_dir,
                        reference_scale_dir, max_epochs=60, batch_size=128,
                        smoke=False, overwrite=False):
    """Fit one source-only Facial CNN seed and evaluate all shared gains."""
    started = time.time()
    dataset_module, deep, torch = load_deep_stack()
    plan = inspect_facial_cnn_run(
        fold_index, training_seed, selection_dir, reference_scale_dir,
        max_epochs, smoke=smoke,
    )
    output = require_scale_path(output, "Facial CNN output")
    if output.exists() and not overwrite:
        raise FileExistsError(
            "refusing to spend GPU time because output exists; pass --overwrite: %s"
            % output
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen deep-baseline protocol")
    device = torch.device("cuda")
    target = plan["target"]
    sources = plan["sources"]

    # Fit every learned quantity before opening the held-out participant.
    source_corpus = deep.load_corpus(sources)
    source_paths = {subject: source_corpus[subject]["path"] for subject in sources}
    source_x, source_y = deep.concat_subjects(source_corpus, sources)
    center, scale = deep.fit_standardizer(source_x)
    source_x = deep.apply_standardizer(source_x, center, scale)
    del source_corpus
    gc.collect()
    model, _ = deep.train_model(
        FACIAL_CNN_ARCHITECTURE, source_x, source_y, plan["selected_epoch"],
        max_epochs, batch_size, int(training_seed), device,
    )
    del source_x, source_y
    gc.collect()

    target_row = deep.load_corpus([target])[target]
    truth, blocks = validate_deep_target_grid(target_row)
    target_raw = target_row["data"]
    if plan["reference"] is not None:
        canonical = plan["reference"].get("canonical_target_trials", {})
        if canonical.get("truth") != truth.tolist():
            raise ValueError("reference fold has a different target truth order")
        if canonical.get("blocks") != blocks.tolist():
            raise ValueError("reference fold has a different target block order")

    clean_x = deep.apply_standardizer(target_raw, center, scale)
    clean = deep_metric_row(
        deep.evaluate(model, clean_x, target_row["labels"], batch_size, device),
        truth,
    )
    del clean_x

    sigma_rows = {}
    for sigma_key, gain_rows in plan["gain_plan"].items():
        draws = []
        for draw_index, (gain_seed, gains) in enumerate(gain_rows):
            perturbed_raw = np.multiply(
                target_raw, gains.astype(np.float32)[None, :, None],
                dtype=np.float32,
            )
            perturbed_x = deep.apply_standardizer(perturbed_raw, center, scale)
            row = deep_metric_row(
                deep.evaluate(
                    model, perturbed_x, target_row["labels"], batch_size, device
                ),
                truth,
            )
            for metric in ("accuracy", "macro_f1"):
                row["clean_to_perturbed_%s_drop" % metric] = float(
                    clean[metric] - row[metric]
                )
            draws.append({
                "draw_index": int(draw_index),
                "seed": int(gain_seed),
                "gain_distribution": "exp(Normal(0, sigma^2))",
                "gain_vector_float64": gains.tolist(),
                "gain_sha256_float64_c_order": sha256_array(gains),
                "gain_application_dtype": "float32 network input",
                "gain_scope": "one fixed 20-channel vector reused for all 330 target trials",
                "applied_to_target_trials": EXPECTED_TRIALS,
                "reused_across_target_trials": True,
                "same_as_reference_scale_mechanism": plan["reference_path"] is not None,
                "methods": {FACIAL_CNN_ID: row},
            })
            del perturbed_raw, perturbed_x
        sigma = float(sigma_key)
        sigma_rows[sigma_key] = {
            "sigma": sigma,
            "role": "primary" if sigma == 0.5 else "exploratory",
            "n_draws": len(draws),
            "draws": draws,
        }

    payload = {
        "schema_version": 2,
        "experiment": "faceemg11_facial_cnn_channel_gain_robustness_loso",
        "model_id": FACIAL_CNN_ID,
        "display_name": "Facial CNN",
        "architecture_id": FACIAL_CNN_ARCHITECTURE,
        "architectural_provenance": deep.ARCHITECTURAL_PROVENANCE[
            FACIAL_CNN_ARCHITECTURE
        ],
        "methods": [FACIAL_CNN_ID],
        "protocol": {
            "outer": "participant-LOSO",
            "outer_fold_index": int(fold_index),
            "outer_target": target,
            "source_subjects": sources,
            "source_blocks": list(ALL_BLOCKS),
            "target_blocks": list(ALL_BLOCKS),
            "source_fit_trials": 3630,
            "target_evaluation_trials": EXPECTED_TRIALS,
            "target_samples_used_for_fit_or_selection": 0,
            "target_labels_used_for_fit_or_selection": False,
            "target_unlabeled_statistics_used_for_fit_or_selection": False,
            "target_derived_statistics": [],
            "perturbation": "target-only fixed positive diagonal channel gain",
            "same_gain_reused_for_all_target_trials": True,
            "normalization_after_perturbation": (
                "fixed source-only per-channel z-score; never refitted on target"
            ),
            "channels": CHANNELS,
            "classes": len(cfg.LABEL_IDS),
            "recorded_blocks": int(cfg.RECORDED_BLOCKS),
        },
        "training": {
            "seed": int(training_seed),
            "selection_seed": int(deep.SELECTION_SEED),
            "selected_epoch": int(plan["selected_epoch"]),
            "max_epochs": int(max_epochs),
            "batch_size": int(batch_size),
            "optimizer": "AdamW",
            "schedule_matches_selection_prefix": True,
            "parameter_count": int(sum(
                parameter.numel()
                for parameter in deep.make_model(FACIAL_CNN_ARCHITECTURE).parameters()
            )),
        },
        "normalization": {
            "method": "per-channel z-score over trial and time axes",
            "fit_subjects": sources,
            "target_used_for_fit": False,
            "center_sha256_float32_c_order": sha256_array(center),
            "scale_sha256_float32_c_order": sha256_array(scale),
        },
        "selection_evidence": {
            "selection_file": str(plan["selection_path"]),
            "selection_sha256": sha256_file(plan["selection_path"]),
            "selection_target_used": False,
            "selection_source_subjects": sources,
        },
        "canonical_target_trials": {
            "trial_index": list(range(EXPECTED_TRIALS)),
            "blocks": blocks.tolist(),
            "truth": truth.tolist(),
        },
        "clean": {"methods": {FACIAL_CNN_ID: clean}},
        "scale_perturbation": {
            "gain_seed_base": GAIN_SEED_BASE,
            "reference_scale_fold": (
                None if plan["reference_path"] is None else str(plan["reference_path"])
            ),
            "reference_scale_fold_sha256": (
                None if plan["reference_path"] is None
                else sha256_file(plan["reference_path"])
            ),
            "sigmas": sigma_rows,
        },
        "target_derived_statistics": [],
        "smoke": bool(smoke),
        "hashes": {
            "script_sha256": sha256_file(Path(__file__)),
            "config_sha256": sha256_file(Path(cfg.__file__)),
            "dataset_sha256": sha256_file(Path(dataset_module.__file__)),
            "deep_baseline_script_sha256": sha256_file(Path(deep.__file__)),
        },
        "data_paths": {"source": source_paths, "target": target_row["path"]},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
        },
        "job": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "runtime_seconds": float(time.time() - started),
    }
    atomic_json(output, payload, overwrite=overwrite)
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Run deterministic or Facial CNN channel-gain robustness."
    )
    parser.add_argument(
        "--model", choices=RUN_MODELS, default="deterministic",
        help="deterministic runs the six fixed pipelines; facial_cnn runs one GPU seed",
    )
    parser.add_argument(
        "--fold", type=int, required=True, choices=range(len(cfg.SUBJECTS)),
        help="outer target index (0..11)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-root", type=Path,
        default=Path(cfg.RESULT_ROOT) / OUTPUT_TOKEN / "feature_cache",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="one outer fold, primary sigma 0.5, exactly two shared gain draws",
    )
    parser.add_argument("--seed", type=int, choices=FINAL_TRAINING_SEEDS)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--selection-dir", type=Path,
        default=PROJECT_ROOT / "results" / "deep_selection",
    )
    parser.add_argument(
        "--reference-scale-dir", type=Path,
        default=PROJECT_ROOT / "results" / OUTPUT_TOKEN,
        help="directory containing deterministic fold_XX.json gain plans",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="for facial_cnn, validate selection and gain inputs without training",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_epochs < 1 or args.batch_size < 1:
        parser.error("--max-epochs and --batch-size must be positive")

    if args.model == "deterministic":
        if args.seed is not None:
            parser.error("--seed is only valid with --model facial_cnn")
        if args.validate_only:
            parser.error("--validate-only is only valid with --model facial_cnn")
        payload = run_fold(
            args.fold, args.output, args.cache_root, smoke=bool(args.smoke),
            overwrite=bool(args.overwrite),
        )
        summary = {
            "model": args.model,
            "target": payload["protocol"]["outer_target"],
            "smoke": payload["smoke"],
            "sigma_draw_counts": {
                sigma: row["n_draws"]
                for sigma, row in payload["scale_perturbation"]["sigmas"].items()
            },
            "output": str(args.output),
        }
    else:
        if args.seed is None:
            parser.error("--seed is required with --model facial_cnn")
        plan = inspect_facial_cnn_run(
            args.fold, args.seed, args.selection_dir,
            args.reference_scale_dir, args.max_epochs, smoke=bool(args.smoke),
        )
        if args.validate_only:
            summary = {
                "model": args.model,
                "mode": "validate-only",
                "target": plan["target"],
                "training_seed": args.seed,
                "selected_epoch": plan["selected_epoch"],
                "selection_file": str(plan["selection_path"]),
                "reference_scale_fold": (
                    None if plan["reference_path"] is None
                    else str(plan["reference_path"])
                ),
                "sigma_draw_counts": {
                    sigma: len(rows) for sigma, rows in plan["gain_plan"].items()
                },
                "planned_output": str(args.output),
            }
        else:
            payload = run_facial_cnn_fold(
                args.fold, args.seed, args.output, args.selection_dir,
                args.reference_scale_dir, max_epochs=args.max_epochs,
                batch_size=args.batch_size, smoke=bool(args.smoke),
                overwrite=bool(args.overwrite),
            )
            summary = {
                "model": args.model,
                "target": payload["protocol"]["outer_target"],
                "training_seed": payload["training"]["seed"],
                "selected_epoch": payload["training"]["selected_epoch"],
                "clean_accuracy": payload["clean"]["methods"][FACIAL_CNN_ID]["accuracy"],
                "smoke": payload["smoke"],
                "sigma_draw_counts": {
                    sigma: row["n_draws"]
                    for sigma, row in payload["scale_perturbation"]["sigmas"].items()
                },
                "output": str(args.output),
            }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

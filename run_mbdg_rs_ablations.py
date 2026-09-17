"""FaceEMG-11 pure-LOSO MBDG-RS ablation with scale-equivariant correlation.

This script is additive to the frozen benchmark.  It uses all 30 blocks from
eleven source participants for fitting and all 30 blocks from one outer target
participant for testing.  No target sample, target statistic, or target
prediction is used for model selection.  The output is one auditable JSON per
outer fold and contains trial-level predictions for every requested analysis.

Sections
--------
* fixed 1.5-s mechanism ablation (five MBDG-RS views);
* observation-duration sensitivity (0.5/0.75/1.0/1.5 s);
* pre-specified bilateral channel-budget sensitivity (20/16/12/8 channels);
* target-only subject-level diagonal scale perturbation (TD4,
  source-reference Riemannian, and MBDG-RS).

The script is deliberately self-contained apart from the frozen project
modules ``config.py`` and ``dataset.py``. It writes no paper files.
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
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg
from dataset import load_trials
from model_names import MBDG_RS_FULL_ABLATION_ID, MBDG_RS_ID, MBDG_RS_NAME


ALL_BLOCKS = tuple(range(1, cfg.RECORDED_BLOCKS + 1))
BANDS = ((2, 20), (20, 60), (60, 120), (120, 250), (250, 450))
WINDOWS_MS = (500, 750, 1000, 1500)
CHANNEL_COUNTS = (20, 16, 12, 8)
REGULARIZATION = 0.05
MBDG_RS_LOG_EIGEN_FLOOR = 1e-8
BASE_SEED = 20260824
SCALE_SEED = 2026082401

# Pre-specified nested bilateral sets.  These use the verified metadata order
# in channels.tsv and cover periocular/eye, brow, temporal and jaw regions even
# at the 8-channel budget.  The right-side index is always the mirror of the
# left-side index; no outer-target data are used to select a set.
PAIR_SITES_BY_COUNT = {
    8: (0, 2, 7, 9),
    12: (0, 1, 2, 4, 7, 9),
    16: (0, 1, 2, 3, 4, 7, 8, 9),
    20: tuple(range(10)),
}
PAIR_LANDMARKS = {
    0: "medial canthus / upper nasal sidewall",
    1: "medial supraorbital / medial brow",
    2: "lateral supraorbital / lateral brow",
    3: "lateral canthus",
    4: "medial infraorbital",
    5: "lateral infraorbital",
    6: "anterior tragus / superior preauricular",
    7: "temporal region superior to auricle",
    8: "inferior preauricular / posterior cheek",
    9: "posteroinferior auricular / mandibular angle vicinity",
}
CHANNEL_INDICES = {
    count: np.asarray(list(pairs) + [10 + index for index in pairs], dtype=int)
    for count, pairs in PAIR_SITES_BY_COUNT.items()
}

ABLATION_NAMES = (
    "broadband_dual_geometry_relative_spectrum",
    "five_band_covariance_relative_spectrum",
    "five_band_correlation_relative_spectrum",
    "five_band_dual_geometry_no_spectrum",
    MBDG_RS_FULL_ABLATION_ID,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value):
    value = np.asarray(value)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
    ssc_amp = (np.abs(slopes[:, :, :-1]) > threshold) & (np.abs(slopes[:, :, 1:]) > threshold)
    ssc = np.sum(slope_change & ssc_amp, axis=2)
    return np.concatenate([mav, wl, zc, ssc], axis=1)


def relative_groups(values, group_width):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1] % group_width:
        raise ValueError("feature width is not divisible by group width")
    groups = values.reshape(len(values), -1, group_width)
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
    # Do not clip eigenvalues before exponentiation.  This keeps the matrix
    # exponential mathematically correct for symmetric inputs and catches
    # accidental use of log/exp on non-SPD objects in the diagnostic below.
    return symmetric_function(matrix, np.exp)


def matrix_sqrt(matrix):
    return symmetric_function(matrix, np.sqrt, floor=1e-10)


def matrix_invsqrt(matrix):
    return symmetric_function(matrix, lambda x: 1.0 / np.sqrt(x), floor=1e-10)


def matrix_exp_diagnostic():
    probe = np.diag(np.asarray([-2.0, 1.0]))
    observed = np.diag(matrix_exp(probe))
    expected = np.exp(np.asarray([-2.0, 1.0]))
    if not np.allclose(observed, expected, rtol=1e-12, atol=1e-12):
        raise AssertionError("matrix_exp clipped a negative eigenvalue")


def regularized_covariances(data):
    centered = data - data.mean(axis=2, keepdims=True)
    covariance = centered @ np.swapaxes(centered, 1, 2) / max(data.shape[2] - 1, 1)
    trace = np.trace(covariance, axis1=1, axis2=2)
    covariance = covariance / np.maximum(trace[:, None, None], 1e-12) * data.shape[1]
    return covariance


def airm_covariances(data, regularization=REGULARIZATION):
    """Classical AIRM covariance: trace normalization plus 0.05 I shrinkage."""
    covariance = regularized_covariances(data)
    identity = np.eye(data.shape[1])[None]
    return (1.0 - regularization) * covariance + regularization * identity


def matrix_log_vector(matrices, regularization=REGULARIZATION, log_floor=MBDG_RS_LOG_EIGEN_FLOOR):
    matrices = np.asarray(matrices, dtype=np.float64)
    channels = matrices.shape[1]
    identity = np.eye(channels)
    upper = np.triu_indices(channels)
    off = upper[0] != upper[1]
    output = np.empty((len(matrices), len(upper[0])), dtype=np.float64)
    for index, matrix in enumerate(matrices):
        matrix = (1.0 - regularization) * matrix + regularization * identity
        # The fixed representation uses a 1e-8 eigenvalue floor. AIRM
        # tangent/geometric-mean code below deliberately keeps its independent
        # 1e-10 matrix-log floor.
        logged = matrix_log(matrix, floor=log_floor)
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
        # All tested windows are >= 500 samples; explicit padlen makes the
        # transform deterministic across SciPy releases.
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


def extract_features(raw, fs, window_ms, channels):
    """Return all representations needed by the four experiment sections."""
    if window_ms not in WINDOWS_MS:
        raise ValueError(window_ms)
    indices = CHANNEL_INDICES[int(channels)]
    samples = int(round(float(fs) * window_ms / 1000.0))
    data = np.asarray(raw[:, indices, :samples], dtype=np.float64)
    nchan = data.shape[1]
    td = np.log1p(np.maximum(td_features(data), 0.0))
    td_rel = relative_groups(td, nchan)
    spectral = bandpower(data, fs)
    spec_channel_rel = relative_groups(spectral, nchan)
    cube = spectral.reshape(len(spectral), len(BANDS), nchan).transpose(0, 2, 1)
    spec_band_rel = relative_groups(cube.reshape(len(cube), -1), len(BANDS))
    spec_band_rel = spec_band_rel.reshape(len(cube), nchan, len(BANDS)).transpose(0, 2, 1).reshape(len(cube), -1)
    relative_spectrum = np.concatenate([spec_channel_rel, spec_band_rel], axis=1)

    covariance = regularized_covariances(data)
    airm_covariance = airm_covariances(data)
    covariance_tangent = matrix_log_vector(covariance)
    correlation = correlation_matrices(data)
    correlation_tangent = matrix_log_vector(correlation)
    fb_cov_parts, fb_corr_parts = [], []
    for filtered in filter_bank(data, fs):
        cov = regularized_covariances(filtered)
        fb_cov_parts.append(matrix_log_vector(cov))
        corr = correlation_matrices(filtered)
        fb_corr_parts.append(matrix_log_vector(corr))
    fb_cov = np.concatenate(fb_cov_parts, axis=1)
    fb_corr = np.concatenate(fb_corr_parts, axis=1)

    return {
        "td_rel": td_rel,
        "broad_cov": covariance_tangent,
        "broad_corr": correlation_tangent,
        "fb_cov": fb_cov,
        "fb_corr": fb_corr,
        "relative_spectrum": relative_spectrum,
        "broad_covariance": covariance,
        "airm_covariance": airm_covariance,
        "ablation": {
            ABLATION_NAMES[0]: np.concatenate([covariance_tangent, correlation_tangent, relative_spectrum], axis=1),
            ABLATION_NAMES[1]: np.concatenate([fb_cov, relative_spectrum], axis=1),
            ABLATION_NAMES[2]: np.concatenate([fb_corr, relative_spectrum], axis=1),
            ABLATION_NAMES[3]: np.concatenate([fb_cov, fb_corr], axis=1),
            ABLATION_NAMES[4]: np.concatenate([fb_cov, fb_corr, relative_spectrum], axis=1),
        },
    }


def estimator():
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def score_row(truth, prediction, blocks=None, trial_indices=None, extra=None):
    row = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "n_test_trials": int(len(truth)),
        "truth": np.asarray(truth, dtype=int).tolist(),
        "prediction": np.asarray(prediction, dtype=int).tolist(),
    }
    if blocks is not None:
        row["blocks"] = np.asarray(blocks, dtype=int).tolist()
    if trial_indices is not None:
        row["trial_indices"] = np.asarray(trial_indices, dtype=int).tolist()
    if extra:
        row.update(extra)
    return row


def concatenate_sources(records, source_subjects, feature_key, feature_store):
    x, y = [], []
    for subject in source_subjects:
        record = records[subject]
        x.append(feature_store[subject][feature_key])
        y.append(record.labels)
    return np.concatenate(x), np.concatenate(y)


def fit_predict(source_x, source_y, target_x):
    model = estimator()
    model.fit(source_x, source_y)
    return model.predict(target_x)


def geometric_mean(matrices, max_iter=12, tolerance=1e-7):
    current = matrix_exp(np.mean([matrix_log(x) for x in matrices], axis=0))
    for _ in range(max_iter):
        root = matrix_sqrt(current)
        invroot = matrix_invsqrt(current)
        tangent = np.mean([matrix_log(invroot @ x @ invroot) for x in matrices], axis=0)
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


def run_fixed_ablation(records, sources, target, store):
    item = records[target]
    truth = item.labels
    rows = {}
    for name in ABLATION_NAMES:
        sx = np.concatenate([store[s]["ablation"][name] for s in sources])
        sy = np.concatenate([records[s].labels for s in sources])
        tx = store[target]["ablation"][name]
        pred = fit_predict(sx, sy, tx)
        rows[name] = score_row(truth, pred, item.groups, np.arange(len(truth)))
    return rows


def run_feature_sensitivity(records, sources, target, cache_by_config, key, values):
    item = records[target]
    rows = {}
    for value in values:
        store = cache_by_config[(value, key[1])]
        sx = np.concatenate([store[s]["ablation"][ABLATION_NAMES[4]] for s in sources])
        sy = np.concatenate([records[s].labels for s in sources])
        pred = fit_predict(sx, sy, store[target]["ablation"][ABLATION_NAMES[4]])
        rows[str(value)] = score_row(item.labels, pred, item.groups, np.arange(len(item.labels)))
    return rows


def run_rsystem_scale(records, sources, target, base_store, fold_index):
    """Target-only gain stress test; source representations are fitted once."""
    source_y = np.concatenate([records[s].labels for s in sources])
    source_td = np.concatenate([base_store[s]["td_rel"] for s in sources])
    source_mbdg_rs = np.concatenate([base_store[s]["ablation"][ABLATION_NAMES[4]] for s in sources])
    source_cov = np.concatenate([base_store[s]["airm_covariance"] for s in sources])
    reference = geometric_mean(source_cov)
    source_airm = tangent_features(source_cov, reference)
    models = {
        "TD4": estimator().fit(source_td, source_y),
        MBDG_RS_ID: estimator().fit(source_mbdg_rs, source_y),
        "source_reference_Riemannian": estimator().fit(source_airm, source_y),
    }
    target_raw = records[target].data
    rows = {}
    for sigma_index, sigma in enumerate((0.0, 0.25, 0.5, 0.75)):
        # One fixed channel-gain vector per outer target and sigma.  The same
        # vector is reused for all 330 target trials in this outer fold.
        seed = SCALE_SEED + int(fold_index) * 100 + sigma_index
        rng = np.random.RandomState(seed)
        gains = np.exp(rng.normal(0.0, sigma, size=(cfg.CHANNELS,)))
        perturbed = target_raw * gains[None, :, None]
        features = extract_features(perturbed, records[target].fs, 1500, 20)
        test_airm = tangent_features(features["airm_covariance"], reference)
        predictions = {
            "TD4": models["TD4"].predict(features["td_rel"]),
            MBDG_RS_ID: models[MBDG_RS_ID].predict(features["ablation"][ABLATION_NAMES[4]]),
            "source_reference_Riemannian": models["source_reference_Riemannian"].predict(test_airm),
        }
        rows[str(sigma)] = {
            "seed": int(seed),
            "gain_log_mean": float(np.log(gains).mean()),
            "gain_log_std": float(np.log(gains).std()),
            "gain_sha256": sha256_array(gains.astype(np.float64)),
            "gain_vector": gains.astype(float).tolist(),
            "gain_scope": "one fixed 20-channel vector reused across all target trials",
            "applied_gain_shape": [int(cfg.CHANNELS)],
            "reused_across_target_trials": True,
            "methods": {
                name: score_row(records[target].labels, prediction, records[target].groups, np.arange(len(target_raw)))
                for name, prediction in predictions.items()
            },
        }
    baseline = {name: rows["0.0"]["methods"][name]["accuracy"] for name in models}
    for sigma_row in rows.values():
        for name, method_row in sigma_row["methods"].items():
            method_row["drop_pp_from_sigma0"] = float((baseline[name] - method_row["accuracy"]) * 100.0)
    return {"protocol": "target_trial_fixed_positive_diagonal_gain_only", "fold_index": fold_index, "rows": rows}


def load_records():
    records = {}
    for subject in cfg.SUBJECTS:
        record = load_trials(subject)
        if len(record.labels) != cfg.RECORDED_BLOCKS * len(cfg.LABEL_IDS):
            raise AssertionError("%s does not contain 330 trials" % subject)
        if sorted(np.unique(record.groups).astype(int).tolist()) != list(ALL_BLOCKS):
            raise AssertionError("%s blocks are not exactly 1..30" % subject)
        for block in ALL_BLOCKS:
            for label in cfg.LABEL_IDS:
                count = int(np.sum((record.groups == block) & (record.labels == label)))
                if count != 1:
                    raise AssertionError("%s block %d label %d count=%d" % (subject, block, label, count))
        records[subject] = record
    return records


def build_store(records, window_ms, channels, cache_root):
    """Build/cache all source-independent features for one configuration."""
    store = {}
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    indices = CHANNEL_INDICES[channels]
    for subject, record in records.items():
        path = cache_root / ("%s_w%d_c%d.npz" % (subject, window_ms, channels))
        cache_meta = {
            "schema_version": 3,
            "subject": subject,
            "data_sha256": sha256_file(record.path),
            "script_sha256": sha256_file(Path(__file__)),
            "config_sha256": sha256_file(Path(cfg.__file__)),
            "window_ms": int(window_ms),
            "channels": int(channels),
            "channel_indices": indices.tolist(),
            "bands_hz": [list(band) for band in BANDS],
            "regularization": float(REGULARIZATION),
            "mbdg_rs_log_eigen_floor": float(MBDG_RS_LOG_EIGEN_FLOOR),
        }
        required = ("td_rel", "broad_cov", "broad_corr", "fb_cov", "fb_corr", "relative_spectrum", "broad_covariance", "airm_covariance", "cache_manifest")
        try:
            with np.load(str(path), allow_pickle=False) as payload:
                cached_meta = json.loads(str(payload["cache_manifest"].item())) if "cache_manifest" in payload.files else None
                if cached_meta == cache_meta and all(key in payload.files for key in required + ABLATION_NAMES):
                    store[subject] = {key: payload[key] for key in required}
                    store[subject].pop("cache_manifest", None)
                    store[subject]["ablation"] = {name: payload[name] for name in ABLATION_NAMES}
                    continue
        except (FileNotFoundError, ValueError, OSError):
            pass
        features = extract_features(record.data, record.fs, window_ms, channels)
        values = {key: features[key] for key in required if key != "cache_manifest"}
        values.update(features["ablation"])
        values["cache_manifest"] = np.asarray(json.dumps(cache_meta, sort_keys=True))
        temporary = path.with_name(path.name + ".tmp.%d.npz" % os.getpid())
        np.savez_compressed(str(temporary), **values)
        os.replace(str(temporary), str(path))
        store[subject] = {key: features[key] for key in required if key != "cache_manifest"}
        store[subject]["ablation"] = features["ablation"]
    return store


def run_fold(fold_index, output, smoke=False):
    started = time.time()
    records = load_records()
    target = cfg.SUBJECTS[int(fold_index)]
    sources = tuple(subject for subject in cfg.SUBJECTS if subject != target)
    result_root = Path(cfg.RESULT_ROOT) / "mbdg_rs"
    cache_root = result_root / "feature_cache"
    base_store = build_store(records, 1500, 20, cache_root)

    fixed = run_fixed_ablation(records, sources, target, base_store)
    if smoke:
        # Smoke also executes the scale section so the fixed subject-level
        # gain invariant can be checked before the array is submitted.
        scale = run_rsystem_scale(records, sources, target, base_store, fold_index)
        payload = make_payload(records, sources, target, fold_index, fixed=fixed, smoke=True, started=started)
        payload["scale_robustness"] = scale
        atomic_json(output, payload)
        return payload

    window_stores = {}
    for window in WINDOWS_MS:
        window_stores[(window, 20)] = build_store(records, window, 20, cache_root)
    window_rows = {}
    for window in WINDOWS_MS:
        store = window_stores[(window, 20)]
        sx = np.concatenate([store[s]["ablation"][ABLATION_NAMES[4]] for s in sources])
        sy = np.concatenate([records[s].labels for s in sources])
        pred = fit_predict(sx, sy, store[target]["ablation"][ABLATION_NAMES[4]])
        window_rows[str(window)] = score_row(records[target].labels, pred, records[target].groups, np.arange(len(pred)), {"observation_duration_ms": window})

    channel_stores = {}
    channel_rows = {}
    for channels in CHANNEL_COUNTS:
        channel_stores[channels] = build_store(records, 1500, channels, cache_root)
        store = channel_stores[channels]
        sx = np.concatenate([store[s]["ablation"][ABLATION_NAMES[4]] for s in sources])
        sy = np.concatenate([records[s].labels for s in sources])
        pred = fit_predict(sx, sy, store[target]["ablation"][ABLATION_NAMES[4]])
        channel_rows[str(channels)] = score_row(records[target].labels, pred, records[target].groups, np.arange(len(pred)), {
            "channel_count": channels,
            "channel_indices_zero_based": CHANNEL_INDICES[channels].tolist(),
        })

    scale = run_rsystem_scale(records, sources, target, base_store, fold_index)
    payload = make_payload(records, sources, target, fold_index, fixed=fixed, smoke=False, started=started)
    payload["observation_duration_sensitivity"] = {
        "durations_ms": list(WINDOWS_MS),
        "label": "observation-duration sensitivity",
        "methods": {MBDG_RS_ID: window_rows},
    }
    payload["channel_budget_sensitivity"] = {
        "channel_counts": list(CHANNEL_COUNTS),
        "selection": "predefined first P mirror pairs; target-independent",
        "methods": {MBDG_RS_ID: channel_rows},
    }
    payload["scale_robustness"] = scale
    atomic_json(output, payload)
    return payload


def make_payload(records, sources, target, fold_index, fixed, smoke, started):
    target_record = records[target]
    input_hashes = {subject: sha256_file(records[subject].path) for subject in cfg.SUBJECTS}
    return {
        "schema_version": 3,
        "experiment": "faceemg11_mbdg_rs_ablation_loso",
        "protocol": {
            "outer": "leave-one-subject-out",
            "outer_fold_index": int(fold_index),
            "outer_target": target,
            "source_subjects": list(sources),
            "source_blocks": list(ALL_BLOCKS),
            "target_blocks": list(ALL_BLOCKS),
            "target_samples_used_for_fit_or_selection": 0,
            "window_sensitivity_definition": "observation-duration sensitivity",
            "channel_sets": {str(n): CHANNEL_INDICES[n].tolist() for n in CHANNEL_COUNTS},
            "channel_pair_sites": {str(n): list(PAIR_SITES_BY_COUNT[n]) for n in CHANNEL_COUNTS},
            "channel_pair_landmarks": {str(n): [PAIR_LANDMARKS[i] for i in PAIR_SITES_BY_COUNT[n]] for n in CHANNEL_COUNTS},
            "bands_hz": [list(band) for band in BANDS],
            "regularization": float(REGULARIZATION),
            "mbdg_rs_log_eigen_floor": float(MBDG_RS_LOG_EIGEN_FLOOR),
            "method_id": MBDG_RS_ID,
            "method_display_name": MBDG_RS_NAME,
            "correlation_definition": "Gram matrix of centered per-channel unit vectors; zero-variance channels fail closed",
            "scale_perturbation": "one target-subject-level positive diagonal channel gain vector per sigma, reused across all 330 target trials; log gain ~ Normal(0,sigma^2)",
        },
        "counts": {
            "n_source_subjects": len(sources),
            "n_target_subjects": 1,
            "source_trials": int(sum(len(records[s].labels) for s in sources)),
            "target_trials": int(len(target_record.labels)),
            "target_blocks": sorted(np.unique(target_record.groups).astype(int).tolist()),
            "target_labels": sorted(np.unique(target_record.labels).astype(int).tolist()),
        },
        "subjects": {"target": target, "sources": list(sources)},
        "fixed_1p5s_ablation": {"window_ms": 1500, "channels": 20, "methods": fixed},
        "smoke": bool(smoke),
        "seeds": {"base_seed": BASE_SEED, "scale_seed_base": SCALE_SEED},
        "input_sha256": input_hashes,
        "script_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(Path(cfg.__file__)),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(len(cfg.SUBJECTS)))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="run only fixed 1.5-s ablation for one fold")
    args = parser.parse_args()
    matrix_exp_diagnostic()
    payload = run_fold(args.fold, args.output, smoke=args.smoke)
    print(json.dumps({
        "target": payload["protocol"]["outer_target"],
        "smoke": payload["smoke"],
        "output": str(args.output),
        "fixed": {name: row["accuracy"] for name, row in payload["fixed_1p5s_ablation"]["methods"].items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

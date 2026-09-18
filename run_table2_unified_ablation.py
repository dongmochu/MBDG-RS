#!/usr/bin/env python3
"""Unified Table-2 runner and aggregator for FaceEMG-11.

This additive file keeps the existing frequency-band ablation untouched.  A
single per-fold run includes its fixed single-band and nested source-selected
Best-k dual-geometry analyses, the existing Table-2 component ablations, and
two same-band representation controls:

* BP-LDA: five-band channel log-power (100 dimensions);
* Cov-LDA: five-band log-covariance without relative spectrum (1050 dims).

Every outer model uses the same 11-source/1-target pure LOSO split, all 30
blocks, source-only StandardScaler, and LSQR LDA with automatic shrinkage.
Best-k selection remains 11-fold inner source-participant LOSO.  Cached arrays
are deterministic within-trial transforms only; no scaler, classifier, or
cross-trial statistic is cached.

Modes
-----
describe   Print the frozen protocol without loading data.
run        Produce one auditable outer-fold JSON.
aggregate  Validate and summarize exactly twelve outer-fold JSON files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from model_names import MBDG_RS_FULL_ABLATION_ID, MBDG_RS_NAME


SUBJECTS = tuple("sub-%02d" % index for index in range(1, 13))
LABEL_IDS = tuple(range(1, 12))
ALL_BLOCKS = tuple(range(1, 31))
BANDS_HZ = ((2, 20), (20, 60), (60, 120), (120, 250), (250, 450))
BAND_IDS = tuple(range(1, 6))
CHANNELS = 20
WINDOW_SAMPLES = 1500
TRIALS_PER_SUBJECT = 330
REGULARIZATION = 0.05
TIE_ATOL = 1e-12
SCHEMA_VERSION = 1
EXPERIMENT = "faceemg11_table2_unified_nested_loso"
AGGREGATE_EXPERIMENT = "faceemg11_table2_unified_nested_loso_aggregate"

CONTROL_IDS = (
    "bp_lda",
    "cov_lda",
    "broadband_dual_geometry_relative_spectrum",
    "five_band_correlation_relative_spectrum",
    "five_band_covariance_relative_spectrum",
    "five_band_dual_geometry_no_spectrum",
    MBDG_RS_FULL_ABLATION_ID,
)

CONTROL_LABELS = {
    "bp_lda": "Five-band channel log-power",
    "cov_lda": "Five-band covariance, no spectrum",
    "broadband_dual_geometry_relative_spectrum": (
        "Broadband dual geometry + relative spectrum"
    ),
    "five_band_correlation_relative_spectrum": (
        "Five-band correlation + relative spectrum"
    ),
    "five_band_covariance_relative_spectrum": (
        "Five-band covariance + relative spectrum"
    ),
    "five_band_dual_geometry_no_spectrum": (
        "Five-band dual geometry, no spectrum"
    ),
    MBDG_RS_FULL_ABLATION_ID: "Full " + MBDG_RS_NAME,
}

CONTROL_WIDTHS = {
    "bp_lda": 100,
    "cov_lda": 1050,
    "broadband_dual_geometry_relative_spectrum": 620,
    "five_band_correlation_relative_spectrum": 1250,
    "five_band_covariance_relative_spectrum": 1250,
    "five_band_dual_geometry_no_spectrum": 2100,
    MBDG_RS_FULL_ABLATION_ID: 2300,
}

SINGLE_MODEL_IDS = tuple("single_b%d" % index for index in BAND_IDS)
BEST_MODEL_IDS = tuple("best_%d" % index for index in BAND_IDS)
ALL_MODEL_IDS = SINGLE_MODEL_IDS + BEST_MODEL_IDS + CONTROL_IDS
MODEL_CHOICES = ("all",) + ALL_MODEL_IDS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def atomic_json_new(path: Path, payload) -> None:
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError("refusing to overwrite existing output: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.%d" % os.getpid())
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    if path.exists():
        temporary.unlink()
        raise FileExistsError("output appeared while writing: %s" % path)
    os.replace(str(temporary), str(path))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def import_frequency_runner(project_root: Path):
    project_root = Path(project_root).expanduser().resolve()
    path = project_root / "run_frequency_band_ablation.py"
    if not path.is_file():
        raise FileNotFoundError("missing existing frequency runner: %s" % path)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    importlib.invalidate_caches()
    module = importlib.import_module("run_frequency_band_ablation")
    if Path(module.__file__).resolve() != path:
        raise RuntimeError(
            "run_frequency_band_ablation imported from unexpected path: %s"
            % module.__file__
        )
    return module


def self_check(old) -> None:
    old.self_check()
    require(tuple(old.SUBJECTS) == SUBJECTS, "subject definitions disagree")
    require(tuple(old.LABEL_IDS) == LABEL_IDS, "label definitions disagree")
    require(tuple(tuple(x) for x in old.BANDS_HZ) == BANDS_HZ,
            "band definitions disagree")
    require(int(old.CHANNELS) == CHANNELS, "channel definitions disagree")
    require(int(old.WINDOW_SAMPLES) == WINDOW_SAMPLES,
            "window definitions disagree")
    require(float(old.REGULARIZATION) == REGULARIZATION,
            "regularization definitions disagree")
    require(CONTROL_WIDTHS["bp_lda"] == CHANNELS * len(BANDS_HZ),
            "BP feature width self-check failed")
    triangle = CHANNELS * (CHANNELS + 1) // 2
    require(CONTROL_WIDTHS["cov_lda"] == 5 * triangle,
            "Cov feature width self-check failed")
    require(CONTROL_WIDTHS[MBDG_RS_FULL_ABLATION_ID] == old.expected_feature_width(BAND_IDS, True),
            "full representation width self-check failed")


def describe_protocol(old) -> dict:
    previous = old.protocol_description()
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "purpose": "one unified source for the complete Table-2 analysis",
        "previous_frequency_ablation": previous,
        "new_representation_controls": {
            "BP-LDA": {
                "command_model_id": "bp_lda",
                "table_label": CONTROL_LABELS["bp_lda"],
                "features": (
                    "direct-rFFT log-average power for 5 fixed bands x 20 channels"
                ),
                "feature_width": CONTROL_WIDTHS["bp_lda"],
            },
            "Cov-LDA": {
                "command_model_id": "cov_lda",
                "table_label": CONTROL_LABELS["cov_lda"],
                "features": (
                    "five trace-normalized, identity-regularized log-covariance svecs"
                ),
                "relative_spectrum": False,
                "feature_width": CONTROL_WIDTHS["cov_lda"],
            },
        },
        "all_fixed_table2_controls": [
            {
                "id": method_id,
                "table_label": CONTROL_LABELS[method_id],
                "feature_width": CONTROL_WIDTHS[method_id],
            }
            for method_id in CONTROL_IDS
        ],
        "classifier_for_every_row": (
            "source-only StandardScaler + LinearDiscriminantAnalysis("
            "solver='lsqr', shrinkage='auto')"
        ),
        "random_seeds": "none; all methods in this file are deterministic",
        "target_use": "outer evaluation only; no fit or selection statistic",
        "command_model_choices": list(MODEL_CHOICES),
    }


def extract_unified_features(record, project, old) -> dict:
    """Extract only deterministic trial-local features."""
    values = old.extract_trial_local_features(record, project.mbdg_rs)
    data = np.asarray(
        record.data[:, :CHANNELS, :WINDOW_SAMPLES], dtype=np.float64
    )
    values["bandpower"] = np.asarray(
        project.mbdg_rs.bandpower(data, float(record.fs)), dtype=np.float64
    )
    broad_covariance = project.mbdg_rs.regularized_covariances(data)
    broad_correlation = project.mbdg_rs.correlation_matrices(data)
    values["broad_cov"] = project.mbdg_rs.matrix_log_vector(broad_covariance)
    values["broad_corr"] = project.mbdg_rs.matrix_log_vector(broad_correlation)
    return values


def unified_feature_keys():
    keys = []
    for band_id in BAND_IDS:
        keys.extend(("cov_b%d" % band_id, "corr_b%d" % band_id))
    keys.extend(
        (
            "bandpower",
            "broad_cov",
            "broad_corr",
            "relative_spectrum",
            "labels",
            "blocks",
            "cache_manifest",
        )
    )
    return tuple(keys)


def validate_unified_features(values) -> None:
    triangle = CHANNELS * (CHANNELS + 1) // 2
    for band_id in BAND_IDS:
        for prefix in ("cov", "corr"):
            key = "%s_b%d" % (prefix, band_id)
            require(tuple(values[key].shape) == (TRIALS_PER_SUBJECT, triangle),
                    "invalid cached shape for %s" % key)
            require(np.isfinite(values[key]).all(),
                    "non-finite cached values for %s" % key)
    expected_shapes = {
        "bandpower": (TRIALS_PER_SUBJECT, 100),
        "broad_cov": (TRIALS_PER_SUBJECT, triangle),
        "broad_corr": (TRIALS_PER_SUBJECT, triangle),
        "relative_spectrum": (TRIALS_PER_SUBJECT, 200),
        "labels": (TRIALS_PER_SUBJECT,),
        "blocks": (TRIALS_PER_SUBJECT,),
    }
    for key, shape in expected_shapes.items():
        require(tuple(values[key].shape) == shape,
                "invalid cached shape for %s" % key)
        if key not in ("labels", "blocks"):
            require(np.isfinite(values[key]).all(),
                    "non-finite cached values for %s" % key)


def cache_manifest(project, old, record, data_hash) -> dict:
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "subject": record.subject,
        "data_path": str(Path(record.path).resolve()),
        "data_sha256": data_hash,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "previous_frequency_runner_sha256": sha256_file(Path(old.__file__).resolve()),
        "frozen_module_sha256": project.module_sha256,
        "window_samples": WINDOW_SAMPLES,
        "sample_rate_hz": 1000.0,
        "channels": CHANNELS,
        "bands_hz": [list(band) for band in BANDS_HZ],
        "regularization": REGULARIZATION,
        "cached_operations": (
            "within-trial filtering, log-bandpower, covariance/correlation "
            "regularization and matrix-log vectorization, relative spectrum"
        ),
        "learned_operations_cached": [],
    }


def load_or_build_subject(project, old, subject, trial_root, cache_dir):
    record = project.load_trials(subject, root=trial_root)
    old.validate_record(record)
    data_hash = sha256_file(Path(record.path))
    manifest = cache_manifest(project, old, record, data_hash)
    fingerprint = hashlib.sha256(
        canonical_json(manifest).encode("utf-8")
    ).hexdigest()[:16]
    cache_path = Path(cache_dir) / ("%s_%s.npz" % (subject, fingerprint))

    try:
        with np.load(str(cache_path), allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
        observed_manifest = json.loads(str(values["cache_manifest"].item()))
        require(observed_manifest == manifest, "cache manifest mismatch")
        require(all(key in values for key in unified_feature_keys()),
                "cache arrays are incomplete")
        validate_unified_features(values)
        values.pop("cache_manifest", None)
        return values, data_hash, cache_path, True
    except FileNotFoundError:
        pass
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("invalid content-addressed cache %s: %s" % (
            cache_path, error,
        )) from error

    values = extract_unified_features(record, project, old)
    validate_unified_features(values)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(
        cache_path.name + ".tmp.%d.npz" % os.getpid()
    )
    payload = dict(values)
    payload["cache_manifest"] = np.asarray(canonical_json(manifest))
    np.savez_compressed(str(temporary), **payload)
    if cache_path.exists():
        temporary.unlink()
    else:
        os.replace(str(temporary), str(cache_path))
    return values, data_hash, cache_path, False


def load_feature_store(project, old, trial_root, cache_dir):
    store = {}
    data_hashes = {}
    cache_records = {}
    for subject in SUBJECTS:
        values, data_hash, cache_path, hit = load_or_build_subject(
            project, old, subject, trial_root, cache_dir
        )
        store[subject] = values
        data_hashes[subject] = data_hash
        cache_records[subject] = {"path": str(cache_path), "hit": bool(hit)}
        print(
            "%s: %s %s" % (
                subject, "loaded" if hit else "created", cache_path
            ),
            flush=True,
        )
    return store, data_hashes, cache_records


def control_features(item, method_id):
    if method_id == "bp_lda":
        result = item["bandpower"]
        require(result.shape[1] == CONTROL_WIDTHS[method_id],
                "%s width mismatch" % method_id)
        return result

    if method_id == "broadband_dual_geometry_relative_spectrum":
        result = np.concatenate(
            [item["broad_cov"], item["broad_corr"], item["relative_spectrum"]],
            axis=1,
        )
        require(result.shape[1] == CONTROL_WIDTHS[method_id],
                "%s width mismatch" % method_id)
        return result

    cov = np.concatenate(
        [item["cov_b%d" % band_id] for band_id in BAND_IDS], axis=1
    )
    if method_id == "cov_lda":
        result = cov
    elif method_id == "five_band_covariance_relative_spectrum":
        result = np.concatenate([cov, item["relative_spectrum"]], axis=1)
    else:
        corr = np.concatenate(
            [item["corr_b%d" % band_id] for band_id in BAND_IDS], axis=1
        )
        if method_id == "five_band_correlation_relative_spectrum":
            result = np.concatenate([corr, item["relative_spectrum"]], axis=1)
        elif method_id == "five_band_dual_geometry_no_spectrum":
            result = np.concatenate([cov, corr], axis=1)
        elif method_id == MBDG_RS_FULL_ABLATION_ID:
            result = np.concatenate(
                [cov, corr, item["relative_spectrum"]], axis=1
            )
        else:
            raise KeyError("unknown Table-2 control: %s" % method_id)
    require(result.shape[1] == CONTROL_WIDTHS[method_id],
            "%s width %d != %d" % (
                method_id, result.shape[1], CONTROL_WIDTHS[method_id]
            ))
    return result


def fit_control_outer(old, store, sources, target, method_id):
    train_x = np.concatenate(
        [control_features(store[subject], method_id) for subject in sources],
        axis=0,
    )
    train_y = np.concatenate(
        [store[subject]["labels"] for subject in sources], axis=0
    )
    test_x = control_features(store[target], method_id)
    prediction = old.fit_predict(train_x, train_y, test_x)
    row = old.metric_row(
        store[target]["labels"], prediction,
        include_trials=True, item=store[target],
    )
    row.update({
        "method_id": method_id,
        "table_label": CONTROL_LABELS[method_id],
        "n_fit_trials": int(len(train_y)),
        "feature_width": int(train_x.shape[1]),
        "bands_hz": [list(band) for band in BANDS_HZ],
        "classifier": "source-only StandardScaler + shrinkage LDA",
    })
    return row


def run_single_band(old, store, sources, target, model_id):
    band_id = int(model_id.rsplit("b", 1)[1])
    row = old.fit_outer(store, sources, target, (band_id,), False)
    return {
        "model_id": model_id,
        "kind": "fixed_single_band_dual_geometry",
        "table_label": "Fixed single band b%d, dual geometry, no spectrum"
        % band_id,
        "band_indices_one_based": [band_id],
        "bands_hz": [list(BANDS_HZ[band_id - 1])],
        "outer_test": row,
    }


def run_best_one(old, store, sources, target, model_id):
    k = int(model_id.split("_", 1)[1])
    candidates = []
    combinations = old.combinations_for_k(k)
    for position, combo in enumerate(combinations, start=1):
        print(
            "Best-%d inner candidate %d/%d: %s" % (
                k, position, len(combinations), old.combo_id(combo)
            ),
            flush=True,
        )
        candidates.append(old.evaluate_candidate_inner(store, sources, combo))
    selected, tied_ids = old.select_candidate(candidates)
    selected_combo = tuple(selected["band_indices_one_based"])
    outer = old.fit_outer(store, sources, target, selected_combo, False)
    print(
        "Best-%d selected %s: inner=%.6f outer=%.6f" % (
            k, selected["id"], selected["mean_inner_accuracy"], outer["accuracy"]
        ),
        flush=True,
    )
    return {
        "model_id": model_id,
        "kind": "source_selected_best_k_dual_geometry",
        "table_label": "Best-%d band%s + no spectrum" % (
            k, "" if k == 1 else "s"
        ),
        "k": k,
        "selection_protocol": (
            "11-fold inner LOSO over the eleven complete source subjects; "
            "each candidate uses source-only scaler and shrinkage LDA"
        ),
        "selection_metric": "mean_inner_accuracy",
        "tie_break": "one-based band-index lexicographic order",
        "tie_absolute_tolerance": TIE_ATOL,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected": {
            **old.combo_record(selected_combo),
            "mean_inner_accuracy": selected["mean_inner_accuracy"],
            "sd_inner_accuracy": selected["sd_inner_accuracy"],
            "mean_inner_macro_f1": selected["mean_inner_macro_f1"],
            "tied_candidate_ids": tied_ids,
        },
        "outer_test": outer,
    }


def run_fixed_control(old, store, sources, target, model_id):
    row = fit_control_outer(old, store, sources, target, model_id)
    print("fixed %s: %.6f" % (model_id, row["accuracy"]), flush=True)
    return {
        "model_id": model_id,
        "kind": "fixed_table2_representation_or_component",
        "table_label": CONTROL_LABELS[model_id],
        "feature_width": CONTROL_WIDTHS[model_id],
        "outer_test": row,
    }


def run_requested_model(old, store, sources, target, model_id):
    if model_id in SINGLE_MODEL_IDS:
        return run_single_band(old, store, sources, target, model_id)
    if model_id in BEST_MODEL_IDS:
        return run_best_one(old, store, sources, target, model_id)
    if model_id in CONTROL_IDS:
        return run_fixed_control(old, store, sources, target, model_id)
    raise ValueError("unknown model id: %s" % model_id)


def run_outer_fold(args, old):
    import scipy
    import sklearn

    started = time.time()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite existing output: %s" % output)

    project = old.import_frozen_project(args.project_root)
    trial_root = (
        Path(args.trial_root).expanduser().resolve()
        if args.trial_root is not None
        else Path(project.cfg.TRIAL_ROOT).expanduser().resolve()
    )
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir is not None
        else output.parent / "feature_cache"
    )
    store, data_hashes, cache_records = load_feature_store(
        project, old, trial_root, cache_dir
    )

    fold_index = int(args.fold)
    target = SUBJECTS[fold_index]
    sources = tuple(subject for subject in SUBJECTS if subject != target)
    requested_ids = ALL_MODEL_IDS if args.model == "all" else (args.model,)
    results = {}
    for position, model_id in enumerate(requested_ids, start=1):
        print(
            "model %d/%d: %s" % (position, len(requested_ids), model_id),
            flush=True,
        )
        results[model_id] = run_requested_model(
            old, store, sources, target, model_id
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "requested_model": args.model,
        "completed_model_ids": list(requested_ids),
        "available_model_ids": list(ALL_MODEL_IDS),
        "protocol": {
            "outer": "12-fold leave-one-subject-out",
            "outer_fold_index": fold_index,
            "outer_target": target,
            "source_subjects": list(sources),
            "source_blocks": list(ALL_BLOCKS),
            "target_blocks": list(ALL_BLOCKS),
            "inner": "11-fold leave-one-source-subject-out for Best-k models only",
            "inner_fit_subjects": 10,
            "inner_validation_subjects": 1,
            "target_samples_used_for_fit_or_selection": 0,
            "target_derived_cross_trial_statistics": [],
            "channels": CHANNELS,
            "window_samples": WINDOW_SAMPLES,
            "window_seconds": 1.5,
            "sample_rate_hz": 1000.0,
            "classes": list(LABEL_IDS),
            "bands_hz": [list(band) for band in BANDS_HZ],
            "classifier": (
                "StandardScaler fitted on current source fit only, followed "
                "by LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')"
            ),
            "best_k_selection_metric": (
                "accuracy averaged equally over 11 inner validation subjects"
            ),
            "best_k_tie_break": "one-based band-index lexicographic order",
            "tie_absolute_tolerance": TIE_ATOL,
            "fixed_models_are_pre_specified": True,
            "random_seed": None,
            "deterministic_methods": True,
            "feature_cache_scope": (
                "deterministic within-trial transforms only; no StandardScaler "
                "or LDA state"
            ),
        },
        "counts": {
            "n_source_subjects": len(sources),
            "n_target_subjects": 1,
            "source_trials": len(sources) * TRIALS_PER_SUBJECT,
            "target_trials": TRIALS_PER_SUBJECT,
            "inner_fit_trials": 10 * TRIALS_PER_SUBJECT,
            "inner_validation_trials": TRIALS_PER_SUBJECT,
        },
        "results": results,
        "input_sha256": data_hashes,
        "feature_cache": cache_records,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "previous_frequency_runner_sha256": sha256_file(
            Path(old.__file__).resolve()
        ),
        "frozen_module_sha256": project.module_sha256,
        "frozen_project_root": str(project.root),
        "trial_root": str(trial_root),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "runtime_seconds": float(time.time() - started),
    }
    atomic_json_new(output, payload)
    return payload


def recompute_metrics(truth, prediction):
    truth = np.asarray(truth, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    accuracy = float(np.mean(truth == prediction))
    f1_values = []
    for label in LABEL_IDS:
        tp = int(np.sum((truth == label) & (prediction == label)))
        fp = int(np.sum((truth != label) & (prediction == label)))
        fn = int(np.sum((truth == label) & (prediction != label)))
        denominator = 2 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
    return accuracy, float(np.mean(f1_values))


def validate_score_row(row, context, expected_width=None):
    require(int(row.get("n_trials", -1)) == TRIALS_PER_SUBJECT,
            "%s trial count is not 330" % context)
    truth = np.asarray(row.get("truth", []), dtype=int)
    prediction = np.asarray(row.get("prediction", []), dtype=int)
    blocks = np.asarray(row.get("blocks", []), dtype=int)
    indices = np.asarray(row.get("trial_indices_zero_based", []), dtype=int)
    require(len(truth) == TRIALS_PER_SUBJECT,
            "%s truth length is not 330" % context)
    require(len(prediction) == TRIALS_PER_SUBJECT,
            "%s prediction length is not 330" % context)
    require(sorted(np.unique(truth).tolist()) == list(LABEL_IDS),
            "%s truth labels are not 1..11" % context)
    require(sorted(np.unique(blocks).tolist()) == list(ALL_BLOCKS),
            "%s target blocks are not 1..30" % context)
    require(indices.tolist() == list(range(TRIALS_PER_SUBJECT)),
            "%s trial order mismatch" % context)
    observed_accuracy, observed_f1 = recompute_metrics(truth, prediction)
    require(abs(float(row["accuracy"]) - observed_accuracy) <= 1e-12,
            "%s stored accuracy mismatch" % context)
    require(abs(float(row["macro_f1"]) - observed_f1) <= 1e-12,
            "%s stored macro-F1 mismatch" % context)
    if expected_width is not None:
        require(int(row.get("feature_width", -1)) == int(expected_width),
                "%s feature width mismatch" % context)
    return truth, prediction


def validate_best_result(old, result, sources, context):
    k = int(result["k"])
    candidates = result["candidates"]
    expected_combos = old.combinations_for_k(k)
    require(len(candidates) == len(expected_combos),
            "%s candidate count mismatch" % context)
    observed_combos = [
        tuple(row["band_indices_one_based"]) for row in candidates
    ]
    require(observed_combos == list(expected_combos),
            "%s candidate order mismatch" % context)
    for candidate in candidates:
        inner = candidate["per_validation_subject"]
        require(len(inner) == 11, "%s inner folds incomplete" % context)
        require([row["validation_subject"] for row in inner] == list(sources),
                "%s validation order mismatch" % context)
        for row in inner:
            validation_subject = row["validation_subject"]
            expected_fit = [s for s in sources if s != validation_subject]
            require(row["fit_subjects"] == expected_fit,
                    "%s inner source boundary mismatch" % context)
            require(int(row["n_fit_trials"]) == 3300,
                    "%s inner fit count mismatch" % context)
        accuracies = np.asarray([row["accuracy"] for row in inner], dtype=float)
        require(abs(float(candidate["mean_inner_accuracy"]) -
                    float(accuracies.mean())) <= 1e-12,
                "%s mean inner accuracy mismatch" % context)
    selected, _ = old.select_candidate(candidates)
    require(result["selected"]["id"] == selected["id"],
            "%s selected combination mismatch" % context)
    outer = result["outer_test"]
    validate_score_row(
        outer, context + " outer",
        expected_width=old.expected_feature_width(
            tuple(outer["band_indices_one_based"])
        ),
    )


def validate_model_result(old, result, model_id, sources, context):
    require(result.get("model_id") == model_id,
            "%s model id mismatch" % context)
    if model_id in SINGLE_MODEL_IDS:
        band_id = int(model_id.rsplit("b", 1)[1])
        require(result.get("kind") == "fixed_single_band_dual_geometry",
                "%s kind mismatch" % context)
        require(result.get("band_indices_one_based") == [band_id],
                "%s band mismatch" % context)
        validate_score_row(
            result["outer_test"], context + " outer",
            expected_width=old.expected_feature_width((band_id,)),
        )
    elif model_id in BEST_MODEL_IDS:
        require(result.get("kind") == "source_selected_best_k_dual_geometry",
                "%s kind mismatch" % context)
        require(int(result.get("k", -1)) == int(model_id.split("_", 1)[1]),
                "%s k mismatch" % context)
        validate_best_result(old, result, sources, context)
    elif model_id in CONTROL_IDS:
        require(result.get("kind") == "fixed_table2_representation_or_component",
                "%s kind mismatch" % context)
        require(result.get("table_label") == CONTROL_LABELS[model_id],
                "%s table label mismatch" % context)
        validate_score_row(
            result["outer_test"], context + " outer",
            expected_width=CONTROL_WIDTHS[model_id],
        )
    else:
        raise ValueError("%s unknown model id" % context)


def validate_fold(path, old):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    context = Path(path).name
    require(payload.get("schema_version") == SCHEMA_VERSION,
            "%s schema mismatch" % context)
    require(payload.get("experiment") == EXPERIMENT,
            "%s experiment mismatch" % context)
    protocol = payload["protocol"]
    fold_index = int(protocol["outer_fold_index"])
    require(Path(path).stem == "fold_%02d" % fold_index,
            "%s filename/fold mismatch" % context)
    target = protocol["outer_target"]
    sources = tuple(protocol["source_subjects"])
    require(target == SUBJECTS[fold_index], "%s target mismatch" % context)
    require(sources == tuple(s for s in SUBJECTS if s != target),
            "%s source set mismatch" % context)
    require(int(protocol["target_samples_used_for_fit_or_selection"]) == 0,
            "%s target leakage flag" % context)
    require(protocol["target_derived_cross_trial_statistics"] == [],
            "%s target-derived statistics are not empty" % context)
    require(protocol["source_blocks"] == list(ALL_BLOCKS),
            "%s source blocks mismatch" % context)
    require(protocol["target_blocks"] == list(ALL_BLOCKS),
            "%s target blocks mismatch" % context)
    require(protocol.get("deterministic_methods") is True,
            "%s deterministic declaration missing" % context)
    requested = payload.get("requested_model")
    require(requested in MODEL_CHOICES, "%s requested model invalid" % context)
    expected_ids = ALL_MODEL_IDS if requested == "all" else (requested,)
    require(tuple(payload.get("completed_model_ids", [])) == expected_ids,
            "%s completed model list mismatch" % context)
    results = payload.get("results", {})
    # ``atomic_json_new`` serializes with sort_keys=True, so JSON object keys
    # are alphabetized on disk. The explicit completed_model_ids list is the
    # authoritative order; the results object only needs the same model set.
    require(set(results) == set(expected_ids),
            "%s result model set mismatch" % context)
    reference_truth = None
    for model_id in expected_ids:
        result = results[model_id]
        validate_model_result(
            old, result, model_id, sources, "%s %s" % (context, model_id)
        )
        truth = np.asarray(result["outer_test"]["truth"], dtype=int)
        if reference_truth is None:
            reference_truth = truth
        else:
            require(np.array_equal(reference_truth, truth),
                    "%s target order differs across models" % context)
    return payload


def summarize_rows(rows, targets):
    accuracy = np.asarray([float(row["accuracy"]) for row in rows], dtype=float)
    macro_f1 = np.asarray([float(row["macro_f1"]) for row in rows], dtype=float)
    return {
        "participant_macro_accuracy": float(accuracy.mean()),
        "participant_sd_accuracy": float(accuracy.std(ddof=1)),
        "participant_macro_f1": float(macro_f1.mean()),
        "participant_sd_macro_f1": float(macro_f1.std(ddof=1)),
        "per_subject": [
            {
                "subject": subject,
                "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]),
            }
            for subject, row in zip(targets, rows)
        ],
    }


def summarize_delta(left_rows, right_rows, targets, definition):
    values = np.asarray(
        [
            (float(left["accuracy"]) - float(right["accuracy"])) * 100.0
            for left, right in zip(left_rows, right_rows)
        ],
        dtype=float,
    )
    return {
        "definition": definition,
        "mean_delta_pp": float(values.mean()),
        "sd_delta_pp": float(values.std(ddof=1)),
        "per_subject": [
            {"subject": subject, "delta_pp": float(value)}
            for subject, value in zip(targets, values)
        ],
    }


def aggregate_folds(folds, paths, old):
    targets = [fold["protocol"]["outer_target"] for fold in folds]
    require(targets == list(SUBJECTS),
            "fold files must map in order to sub-01 ... sub-12")
    result_ids = tuple(folds[0]["completed_model_ids"])
    model_summaries = {}
    result_rows = {}
    for model_id in result_ids:
        result_objects = [fold["results"][model_id] for fold in folds]
        outer_rows = [result["outer_test"] for result in result_objects]
        result_rows[model_id] = outer_rows
        summary = {
            "kind": result_objects[0]["kind"],
            "table_label": result_objects[0]["table_label"],
            **summarize_rows(outer_rows, targets),
        }
        if model_id in CONTROL_IDS:
            summary["feature_width"] = CONTROL_WIDTHS[model_id]
        if model_id in SINGLE_MODEL_IDS:
            band_id = int(model_id.rsplit("b", 1)[1])
            summary["band_indices_one_based"] = [band_id]
            summary["bands_hz"] = [list(BANDS_HZ[band_id - 1])]
        if model_id in BEST_MODEL_IDS:
            k = int(model_id.split("_", 1)[1])
            counts = Counter(result["selected"]["id"] for result in result_objects)
            frequencies = []
            for combo in old.combinations_for_k(k):
                candidate_id = old.combo_id(combo)
                frequencies.append({
                    "id": candidate_id,
                    "band_indices_one_based": list(combo),
                    "bands_hz": [list(BANDS_HZ[i - 1]) for i in combo],
                    "count": int(counts[candidate_id]),
                    "proportion": float(counts[candidate_id] / len(folds)),
                })
            maximum = max(row["count"] for row in frequencies)
            summary["selection_frequency"] = frequencies
            summary["most_frequently_selected"] = next(
                row for row in frequencies if row["count"] == maximum
            )
            summary["selected_combination_by_outer_target"] = [
                {
                    "target": target,
                    "selected_id": result["selected"]["id"],
                    "band_indices_one_based": result["selected"]
                    ["band_indices_one_based"],
                    "mean_inner_accuracy": float(
                        result["selected"]["mean_inner_accuracy"]
                    ),
                    "outer_accuracy": float(result["outer_test"]["accuracy"]),
                }
                for target, result in zip(targets, result_objects)
            ]
        model_summaries[model_id] = summary

    recommended_order = BEST_MODEL_IDS[:4] + CONTROL_IDS
    table2_rows = []
    for model_id in recommended_order:
        if model_id not in model_summaries:
            continue
        summary = model_summaries[model_id]
        row = {
            "id": model_id,
            "table_label": summary["table_label"],
            "accuracy_percent": 100.0 * summary["participant_macro_accuracy"],
            "sample_sd_percent": 100.0 * summary["participant_sd_accuracy"],
        }
        if model_id in CONTROL_IDS:
            row["feature_width"] = CONTROL_WIDTHS[model_id]
        if model_id in BEST_MODEL_IDS:
            row["most_frequently_selected"] = summary[
                "most_frequently_selected"
            ]
        table2_rows.append(row)

    comparisons = {}
    comparison_specs = (
        (
            "covariance_minus_log_power", "cov_lda", "bp_lda",
            "five-band covariance without spectrum minus five-band channel log-power",
        ),
        (
            "covariance_spectrum_minus_covariance_no_spectrum",
            "five_band_covariance_relative_spectrum", "cov_lda",
            "five-band covariance plus relative spectrum minus covariance without spectrum",
        ),
        (
            "full_minus_covariance_spectrum", MBDG_RS_FULL_ABLATION_ID,
            "five_band_covariance_relative_spectrum",
            "full MBDG-RS minus five-band covariance plus relative spectrum",
        ),
        (
            "full_minus_dual_no_spectrum", MBDG_RS_FULL_ABLATION_ID,
            "five_band_dual_geometry_no_spectrum",
            "full MBDG-RS minus five-band dual geometry without spectrum",
        ),
    )
    for name, left, right, definition in comparison_specs:
        if left in result_rows and right in result_rows:
            comparisons[name] = summarize_delta(
                result_rows[left], result_rows[right], targets, definition
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": AGGREGATE_EXPERIMENT,
        "requested_model": folds[0]["requested_model"],
        "completed_model_ids": list(result_ids),
        "n_outer_folds": len(folds),
        "outer_targets": targets,
        "protocol": {
            "inference_unit": "held-out participant",
            "participant_weighting": "equal",
            "accuracy_sd": "sample SD across 12 participant accuracies",
            "classifier_fixed_for_all_rows": (
                "source-only StandardScaler + LSQR LDA with automatic shrinkage"
            ),
            "deterministic_methods": True,
            "target_samples_used_for_fit_or_selection": 0,
            "best_k_is_fold_specific_source_only_selection": True,
            "bands_hz": [list(band) for band in BANDS_HZ],
        },
        "model_summaries": model_summaries,
        "table2_rows_in_recommended_order": table2_rows,
        "paired_accuracy_deltas_pp": comparisons,
        "runner_sha256": folds[0]["runner_sha256"],
        "previous_frequency_runner_sha256": folds[0]
        ["previous_frequency_runner_sha256"],
        "frozen_module_sha256": folds[0]["frozen_module_sha256"],
        "source_fold_files": [str(path) for path in paths],
        "source_fold_sha256": {
            Path(path).name: sha256_file(path) for path in paths
        },
    }


def aggregate_mode(args, old):
    input_dir = Path(args.input_dir).expanduser().resolve()
    paths = sorted(input_dir.glob("fold_[0-9][0-9].json"))
    require(len(paths) == len(SUBJECTS),
            "expected exactly 12 fold JSONs, found %d" % len(paths))
    folds = [validate_fold(path, old) for path in paths]
    require(len({fold["requested_model"] for fold in folds}) == 1,
            "folds used different --model selections")
    if args.model != "all":
        require(folds[0]["requested_model"] == args.model,
                "aggregate --model does not match fold outputs")
    else:
        require(folds[0]["requested_model"] == "all",
                "aggregate --model all requires all-model fold outputs")
    require(len({fold["runner_sha256"] for fold in folds}) == 1,
            "folds were generated by different unified runner versions")
    require(len({fold["previous_frequency_runner_sha256"] for fold in folds}) == 1,
            "folds used different previous frequency runner versions")
    require(len({canonical_json(fold["frozen_module_sha256"]) for fold in folds}) == 1,
            "folds used different frozen modules")
    require(len({canonical_json(fold["input_sha256"]) for fold in folds}) == 1,
            "folds used different input data")
    payload = aggregate_folds(folds, paths, old)
    atomic_json_new(args.output, payload)
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Unified FaceEMG-11 Table-2 ablation runner/aggregator"
    )
    parser.add_argument(
        "--mode", choices=("describe", "run", "aggregate"), required=True
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent,
        help="project containing the frozen frequency ablation runner",
    )
    parser.add_argument(
        "--model", choices=MODEL_CHOICES, default="all",
        help="run/aggregate one model id, or all models (default: all)",
    )
    parser.add_argument("--trial-root", type=Path)
    parser.add_argument("--fold", type=int, choices=range(len(SUBJECTS)))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "run" and (args.fold is None or args.output is None):
        parser.error("run mode requires --fold and --output")
    if args.mode == "aggregate" and (
        args.input_dir is None or args.output is None
    ):
        parser.error("aggregate mode requires --input-dir and --output")
    return args


def main(argv=None):
    args = parse_args(argv)
    old = import_frequency_runner(args.project_root)
    self_check(old)
    if args.mode == "describe":
        print(json.dumps(describe_protocol(old), indent=2, sort_keys=True))
        return 0
    if args.mode == "run":
        payload = run_outer_fold(args, old)
        print(json.dumps({
            "output": str(Path(args.output).expanduser().resolve()),
            "outer_target": payload["protocol"]["outer_target"],
            "requested_model": payload["requested_model"],
            "completed": {
                model_id: {
                    "accuracy": result["outer_test"]["accuracy"],
                    "selected_combination": (
                        result["selected"]["id"]
                        if model_id in BEST_MODEL_IDS else None
                    ),
                }
                for model_id, result in payload["results"].items()
            },
        }, indent=2, sort_keys=True), flush=True)
        return 0

    payload = aggregate_mode(args, old)
    print(json.dumps({
        "output": str(Path(args.output).expanduser().resolve()),
        "n_outer_folds": payload["n_outer_folds"],
        "requested_model": payload["requested_model"],
        "table2_rows": [
            {
                "id": row["id"],
                "accuracy_percent": row["accuracy_percent"],
                "sample_sd_percent": row["sample_sd_percent"],
            }
            for row in payload["table2_rows_in_recommended_order"]
        ],
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

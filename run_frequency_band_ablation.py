#!/usr/bin/env python3
"""Nested-LOSO frequency-band contribution experiment for FaceEMG-11.

This is an additive runner: it imports the frozen preprocessing and geometry
helpers from ``run_mbdg_rs_ablations.py`` but does not modify any project file.
For each outer target, the script reports:

* all five pre-specified single-band dual-geometry models;
* the fixed five-band dual-geometry anchor without relative spectrum;
* the existing complete five-band MBDG-RS representation with relative spectrum;
* source-selected Best-k dual-geometry procedures for k=1,...,5.

Every Best-k candidate is evaluated by inner LOSO over the eleven complete
source subjects.  StandardScaler and shrinkage LDA are re-fitted in every
inner split.  The isolated outer target is used only for the final prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SUBJECTS = tuple("sub-%02d" % index for index in range(1, 13))
LABEL_IDS = tuple(range(1, 12))
ALL_BLOCKS = tuple(range(1, 31))
BANDS_HZ = ((2, 20), (20, 60), (60, 120), (120, 250), (250, 450))
BAND_IDS = tuple(range(1, len(BANDS_HZ) + 1))
WINDOW_SAMPLES = 1500
CHANNELS = 20
TRIALS_PER_SUBJECT = 330
REGULARIZATION = 0.05
TIE_ATOL = 1e-12
SCHEMA_VERSION = 1
EXPERIMENT = "faceemg11_frequency_band_nested_loso"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def combo_id(combo) -> str:
    return "-".join("b%d" % int(index) for index in combo)


def combo_record(combo) -> dict:
    combo = tuple(int(index) for index in combo)
    return {
        "id": combo_id(combo),
        "band_indices_one_based": list(combo),
        "bands_hz": [list(BANDS_HZ[index - 1]) for index in combo],
    }


def combinations_for_k(k: int):
    return tuple(itertools.combinations(BAND_IDS, int(k)))


def all_combinations():
    return {k: combinations_for_k(k) for k in BAND_IDS}


def expected_feature_width(combo, include_relative_spectrum=False) -> int:
    triangle = CHANNELS * (CHANNELS + 1) // 2
    width = 2 * triangle * len(tuple(combo))
    if include_relative_spectrum:
        width += 2 * CHANNELS * len(BANDS_HZ)
    return width


def select_candidate(candidates: list[dict]) -> tuple[dict, list[str]]:
    """Maximize mean accuracy; numerical ties use fixed band-index order."""
    best_score = max(float(row["mean_inner_accuracy"]) for row in candidates)
    tied = [
        row for row in candidates
        if abs(float(row["mean_inner_accuracy"]) - best_score) <= TIE_ATOL
    ]
    selected = min(
        tied,
        key=lambda row: tuple(int(x) for x in row["band_indices_one_based"]),
    )
    return selected, [row["id"] for row in tied]


def protocol_description() -> dict:
    combos = all_combinations()
    return {
        "experiment": EXPERIMENT,
        "outer_folds": len(SUBJECTS),
        "source_subjects_per_fold": len(SUBJECTS) - 1,
        "inner_folds_per_candidate": len(SUBJECTS) - 1,
        "blocks_per_subject": len(ALL_BLOCKS),
        "trials_per_subject": TRIALS_PER_SUBJECT,
        "channels": CHANNELS,
        "window_seconds": WINDOW_SAMPLES / 1000.0,
        "bands": [combo_record((index,)) for index in BAND_IDS],
        "candidate_counts_by_k": {str(k): len(combos[k]) for k in BAND_IDS},
        "total_nonempty_subsets": sum(len(rows) for rows in combos.values()),
        "inner_model_fits_per_outer_fold": sum(len(rows) for rows in combos.values()) * 11,
        "fixed_outer_models_per_fold": 7,
        "best_k_outer_refits_per_fold": 5,
        "single_band_feature_width": expected_feature_width((1,)),
        "five_band_dual_geometry_width": expected_feature_width(BAND_IDS),
        "full_mbdg_rs_width": expected_feature_width(BAND_IDS, include_relative_spectrum=True),
        "selection_metric": "mean of 11 validation-subject accuracies (equal subject weight)",
        "tie_break": "one-based band-index lexicographic order",
        "tie_absolute_tolerance": TIE_ATOL,
        "relative_spectrum_in_selection": False,
    }


def self_check() -> None:
    expected_counts = {1: 5, 2: 10, 3: 10, 4: 5, 5: 1}
    observed = {k: len(combinations_for_k(k)) for k in BAND_IDS}
    if observed != expected_counts:
        raise AssertionError("unexpected subset counts: %r" % observed)
    if combinations_for_k(2)[0] != (1, 2) or combinations_for_k(2)[-1] != (4, 5):
        raise AssertionError("candidate order is not lexicographic")
    synthetic = [
        {"id": "b2", "band_indices_one_based": [2], "mean_inner_accuracy": 0.75},
        {"id": "b1", "band_indices_one_based": [1], "mean_inner_accuracy": 0.75},
    ]
    selected, tied = select_candidate(synthetic)
    if selected["id"] != "b1" or tied != ["b2", "b1"]:
        raise AssertionError("tie-break self-check failed")
    if expected_feature_width((1,)) != 420:
        raise AssertionError("unexpected single-band feature width")
    if expected_feature_width(BAND_IDS) != 2100:
        raise AssertionError("unexpected five-band geometry width")
    if expected_feature_width(BAND_IDS, True) != 2300:
        raise AssertionError("unexpected complete MBDG-RS feature width")


def import_frozen_project(project_root: Path):
    """Import only the frozen project interfaces needed by this new runner."""
    project_root = Path(project_root).expanduser().resolve()
    required = (
        project_root / "config.py",
        project_root / "dataset.py",
        project_root / "run_mbdg_rs_ablations.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen project modules: %s" % missing)

    sys.path.insert(0, str(project_root))
    importlib.invalidate_caches()
    cfg = importlib.import_module("config")
    dataset = importlib.import_module("dataset")
    mbdg_rs = importlib.import_module("run_mbdg_rs_ablations")

    loaded = {
        "config": Path(cfg.__file__).resolve(),
        "dataset": Path(dataset.__file__).resolve(),
        "run_mbdg_rs_ablations": Path(mbdg_rs.__file__).resolve(),
    }
    for name, path in loaded.items():
        if path.parent != project_root:
            raise RuntimeError("%s imported from unexpected path %s" % (name, path))

    if tuple(cfg.SUBJECTS) != SUBJECTS:
        raise ValueError("frozen project subject list does not match this protocol")
    if tuple(cfg.LABEL_IDS) != LABEL_IDS:
        raise ValueError("frozen project label list does not match this protocol")
    if int(cfg.CHANNELS) != CHANNELS or int(cfg.TRIAL_SAMPLES) != WINDOW_SAMPLES:
        raise ValueError("frozen project channel/window constants do not match")
    if tuple(tuple(x) for x in mbdg_rs.BANDS) != BANDS_HZ:
        raise ValueError("frozen MBDG-RS band definitions do not match")
    if float(mbdg_rs.REGULARIZATION) != REGULARIZATION:
        raise ValueError("frozen MBDG-RS regularization does not match")

    return SimpleNamespace(
        root=project_root,
        cfg=cfg,
        load_trials=dataset.load_trials,
        mbdg_rs=mbdg_rs,
        module_paths=loaded,
        module_sha256={name: sha256_file(path) for name, path in loaded.items()},
    )


def validate_record(record) -> None:
    expected_shape = (TRIALS_PER_SUBJECT, CHANNELS, WINDOW_SAMPLES)
    if tuple(record.data.shape) != expected_shape:
        raise AssertionError("%s data shape %r != %r" % (
            record.subject, tuple(record.data.shape), expected_shape,
        ))
    if float(record.fs) != 1000.0:
        raise AssertionError("%s sample rate is not 1000 Hz" % record.subject)
    if not np.isfinite(record.data).all():
        raise AssertionError("%s contains non-finite samples" % record.subject)
    if sorted(np.unique(record.labels).astype(int).tolist()) != list(LABEL_IDS):
        raise AssertionError("%s labels are not exactly 1..11" % record.subject)
    if sorted(np.unique(record.groups).astype(int).tolist()) != list(ALL_BLOCKS):
        raise AssertionError("%s blocks are not exactly 1..30" % record.subject)
    for block in ALL_BLOCKS:
        for label in LABEL_IDS:
            count = int(np.sum((record.groups == block) & (record.labels == label)))
            if count != 1:
                raise AssertionError(
                    "%s block %d label %d count=%d" % (
                        record.subject, block, label, count,
                    )
                )


def extract_trial_local_features(record, mbdg_rs) -> dict[str, np.ndarray]:
    """Compute deterministic within-trial features, with no cross-trial fit."""
    data = np.asarray(record.data[:, :CHANNELS, :WINDOW_SAMPLES], dtype=np.float64)
    result = {}
    filtered_bands = mbdg_rs.filter_bank(data, float(record.fs))
    if len(filtered_bands) != len(BANDS_HZ):
        raise AssertionError("filter bank returned an unexpected number of bands")

    for band_id, filtered in zip(BAND_IDS, filtered_bands):
        covariance = mbdg_rs.regularized_covariances(filtered)
        correlation = mbdg_rs.correlation_matrices(filtered)
        result["cov_b%d" % band_id] = mbdg_rs.matrix_log_vector(covariance)
        result["corr_b%d" % band_id] = mbdg_rs.matrix_log_vector(correlation)

    spectral = mbdg_rs.bandpower(data, float(record.fs))
    channel_relative = mbdg_rs.relative_groups(spectral, CHANNELS)
    cube = spectral.reshape(
        len(spectral), len(BANDS_HZ), CHANNELS,
    ).transpose(0, 2, 1)
    band_relative = mbdg_rs.relative_groups(
        cube.reshape(len(cube), -1), len(BANDS_HZ),
    )
    band_relative = band_relative.reshape(
        len(cube), CHANNELS, len(BANDS_HZ),
    ).transpose(0, 2, 1).reshape(len(cube), -1)
    result["relative_spectrum"] = np.concatenate(
        [channel_relative, band_relative], axis=1,
    )
    result["labels"] = np.asarray(record.labels, dtype=np.int64)
    result["blocks"] = np.asarray(record.groups, dtype=np.int64)
    return result


def feature_keys():
    keys = []
    for band_id in BAND_IDS:
        keys.extend(("cov_b%d" % band_id, "corr_b%d" % band_id))
    keys.extend(("relative_spectrum", "labels", "blocks", "cache_manifest"))
    return tuple(keys)


def validate_cached_features(values: dict[str, np.ndarray]) -> None:
    triangle = CHANNELS * (CHANNELS + 1) // 2
    for band_id in BAND_IDS:
        for prefix in ("cov", "corr"):
            key = "%s_b%d" % (prefix, band_id)
            if tuple(values[key].shape) != (TRIALS_PER_SUBJECT, triangle):
                raise ValueError("invalid cached shape for %s" % key)
            if not np.isfinite(values[key]).all():
                raise ValueError("non-finite cached values for %s" % key)
    if tuple(values["relative_spectrum"].shape) != (
        TRIALS_PER_SUBJECT, 2 * CHANNELS * len(BANDS_HZ),
    ):
        raise ValueError("invalid cached relative-spectrum shape")
    if tuple(values["labels"].shape) != (TRIALS_PER_SUBJECT,):
        raise ValueError("invalid cached label shape")
    if tuple(values["blocks"].shape) != (TRIALS_PER_SUBJECT,):
        raise ValueError("invalid cached block shape")


def cache_manifest(project, record, data_sha256: str) -> dict:
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "subject": record.subject,
        "data_path": str(Path(record.path).resolve()),
        "data_sha256": data_sha256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_module_sha256": project.module_sha256,
        "window_samples": WINDOW_SAMPLES,
        "sample_rate_hz": 1000.0,
        "channels": CHANNELS,
        "bands_hz": [list(band) for band in BANDS_HZ],
        "regularization": REGULARIZATION,
        "feature_order": (
            ["cov_b%d" % i for i in BAND_IDS]
            + ["corr_b%d" % i for i in BAND_IDS]
            + ["relative_spectrum"]
        ),
        "normalization": (
            "geometry is trial-local trace/correlation normalization; relative "
            "spectrum is trial-local; no StandardScaler is cached"
        ),
    }


def load_or_build_subject(project, subject: str, trial_root: Path, cache_dir: Path):
    record = project.load_trials(subject, root=trial_root)
    validate_record(record)
    data_hash = sha256_file(Path(record.path))
    manifest = cache_manifest(project, record, data_hash)
    fingerprint = hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()[:16]
    cache_path = Path(cache_dir) / ("%s_%s.npz" % (subject, fingerprint))

    try:
        with np.load(str(cache_path), allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
        cached_manifest = json.loads(str(values["cache_manifest"].item()))
        if cached_manifest != manifest:
            raise ValueError("cache manifest mismatch")
        if not all(key in values for key in feature_keys()):
            raise ValueError("cache is missing required arrays")
        validate_cached_features(values)
        values.pop("cache_manifest", None)
        return values, data_hash, cache_path, True
    except FileNotFoundError:
        pass
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("invalid content-addressed cache %s: %s" % (
            cache_path, error,
        )) from error

    values = extract_trial_local_features(record, project.mbdg_rs)
    validate_cached_features({**values, "cache_manifest": np.asarray("")})
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + ".tmp.%d.npz" % os.getpid())
    payload = dict(values)
    payload["cache_manifest"] = np.asarray(canonical_json(manifest))
    np.savez_compressed(str(temporary), **payload)
    if cache_path.exists():
        temporary.unlink()
    else:
        os.replace(str(temporary), str(cache_path))
    return values, data_hash, cache_path, False


def load_feature_store(project, trial_root: Path, cache_dir: Path):
    store = {}
    hashes = {}
    cache_records = {}
    for subject in SUBJECTS:
        values, data_hash, cache_path, cache_hit = load_or_build_subject(
            project, subject, trial_root, cache_dir,
        )
        store[subject] = values
        hashes[subject] = data_hash
        cache_records[subject] = {
            "path": str(cache_path),
            "hit": bool(cache_hit),
        }
        print(
            "%s: %s %s" % (
                subject, "loaded" if cache_hit else "created", cache_path,
            ),
            flush=True,
        )
    return store, hashes, cache_records


def geometry_features(item: dict[str, np.ndarray], combo) -> np.ndarray:
    combo = tuple(int(index) for index in combo)
    if not combo or tuple(sorted(combo)) != combo or not set(combo) <= set(BAND_IDS):
        raise ValueError("invalid band combination %r" % (combo,))
    parts = [item["cov_b%d" % index] for index in combo]
    parts.extend(item["corr_b%d" % index] for index in combo)
    result = np.concatenate(parts, axis=1)
    expected = expected_feature_width(combo)
    if result.shape[1] != expected:
        raise AssertionError("geometry feature width %d != %d" % (
            result.shape[1], expected,
        ))
    return result


def representation(item, combo, include_relative_spectrum=False) -> np.ndarray:
    geometry = geometry_features(item, combo)
    if not include_relative_spectrum:
        return geometry
    result = np.concatenate([geometry, item["relative_spectrum"]], axis=1)
    expected = expected_feature_width(combo, include_relative_spectrum=True)
    if result.shape[1] != expected:
        raise AssertionError("complete feature width %d != %d" % (
            result.shape[1], expected,
        ))
    return result


def estimator():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )


def stack_subjects(store, subjects, combo, include_relative_spectrum=False):
    x = np.concatenate([
        representation(store[subject], combo, include_relative_spectrum)
        for subject in subjects
    ])
    y = np.concatenate([store[subject]["labels"] for subject in subjects])
    return x, y


def fit_predict(train_x, train_y, test_x):
    model = estimator()
    model.fit(train_x, train_y)
    return model.predict(test_x)


def metric_row(truth, prediction, include_trials=False, item=None) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    truth = np.asarray(truth, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    row = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(
            truth, prediction, labels=LABEL_IDS, average="macro", zero_division=0,
        )),
        "n_trials": int(len(truth)),
    }
    if include_trials:
        row["truth"] = truth.tolist()
        row["prediction"] = prediction.tolist()
        row["blocks"] = np.asarray(item["blocks"], dtype=int).tolist()
        row["trial_indices_zero_based"] = list(range(len(truth)))
    return row


def fit_outer(store, sources, target, combo, include_relative_spectrum=False):
    train_x, train_y = stack_subjects(
        store, sources, combo, include_relative_spectrum,
    )
    test_x = representation(store[target], combo, include_relative_spectrum)
    prediction = fit_predict(train_x, train_y, test_x)
    row = metric_row(
        store[target]["labels"], prediction, include_trials=True, item=store[target],
    )
    row.update({
        "n_fit_trials": int(len(train_y)),
        "feature_width": int(train_x.shape[1]),
        "include_relative_spectrum": bool(include_relative_spectrum),
    })
    row.update(combo_record(combo))
    return row


def evaluate_candidate_inner(store, sources, combo) -> dict:
    per_subject_features = {
        subject: representation(store[subject], combo, False)
        for subject in sources
    }
    inner_rows = []
    for validation_subject in sources:
        fit_subjects = tuple(
            subject for subject in sources if subject != validation_subject
        )
        train_x = np.concatenate([
            per_subject_features[subject] for subject in fit_subjects
        ])
        train_y = np.concatenate([
            store[subject]["labels"] for subject in fit_subjects
        ])
        validation_x = per_subject_features[validation_subject]
        validation_truth = store[validation_subject]["labels"]
        prediction = fit_predict(train_x, train_y, validation_x)
        score = metric_row(validation_truth, prediction, include_trials=False)
        score.update({
            "validation_subject": validation_subject,
            "fit_subjects": list(fit_subjects),
            "n_fit_subjects": len(fit_subjects),
            "n_fit_trials": int(len(train_y)),
            "validation_blocks": list(ALL_BLOCKS),
            "fit_blocks": list(ALL_BLOCKS),
        })
        inner_rows.append(score)

    accuracy = np.asarray([row["accuracy"] for row in inner_rows], dtype=float)
    macro_f1 = np.asarray([row["macro_f1"] for row in inner_rows], dtype=float)
    row = combo_record(combo)
    row.update({
        "feature_width": expected_feature_width(combo),
        "mean_inner_accuracy": float(accuracy.mean()),
        "sd_inner_accuracy": float(accuracy.std(ddof=1)),
        "mean_inner_macro_f1": float(macro_f1.mean()),
        "sd_inner_macro_f1": float(macro_f1.std(ddof=1)),
        "subject_weighting": "equal",
        "per_validation_subject": inner_rows,
    })
    return row


def run_best_k(store, sources, target) -> dict:
    output = {}
    for k in BAND_IDS:
        candidates = []
        combinations = combinations_for_k(k)
        for position, combo in enumerate(combinations, start=1):
            print(
                "Best-%d inner candidate %d/%d: %s" % (
                    k, position, len(combinations), combo_id(combo),
                ),
                flush=True,
            )
            candidates.append(evaluate_candidate_inner(store, sources, combo))
        selected, tied_ids = select_candidate(candidates)
        selected_combo = tuple(selected["band_indices_one_based"])
        outer = fit_outer(store, sources, target, selected_combo, False)
        output[str(k)] = {
            "k": int(k),
            "selection_protocol": (
                "11-fold inner LOSO over complete source subjects; each fit uses "
                "10 subjects and validates on one; accuracy averaged with equal "
                "subject weight"
            ),
            "selection_metric": "mean_inner_accuracy",
            "tie_break": "one-based band-index lexicographic order",
            "tie_absolute_tolerance": TIE_ATOL,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "selected": {
                **combo_record(selected_combo),
                "mean_inner_accuracy": selected["mean_inner_accuracy"],
                "sd_inner_accuracy": selected["sd_inner_accuracy"],
                "mean_inner_macro_f1": selected["mean_inner_macro_f1"],
                "tied_candidate_ids": tied_ids,
            },
            "outer_test": outer,
        }
        print(
            "Best-%d selected %s: inner=%.6f outer=%.6f" % (
                k, selected["id"], selected["mean_inner_accuracy"], outer["accuracy"],
            ),
            flush=True,
        )
    return output


def run_fixed_models(store, sources, target) -> dict:
    single_bands = {}
    for band_id in BAND_IDS:
        combo = (band_id,)
        row = fit_outer(store, sources, target, combo, False)
        single_bands[row["id"]] = row
        print("fixed %s: %.6f" % (row["id"], row["accuracy"]), flush=True)

    geometry_anchor = fit_outer(store, sources, target, BAND_IDS, False)
    full_mbdg_rs = fit_outer(store, sources, target, BAND_IDS, True)
    return {
        "single_band_dual_geometry": single_bands,
        "five_band_dual_geometry_anchor": geometry_anchor,
        "five_band_dual_geometry_relative_spectrum": full_mbdg_rs,
    }


def atomic_json_new(path: Path, payload) -> None:
    """Write a new result atomically and refuse to replace an existing path."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite existing output: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.%d" % os.getpid())
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8",
    )
    if path.exists():
        temporary.unlink()
        raise FileExistsError("output appeared during run: %s" % path)
    os.replace(str(temporary), str(path))


def run_fold(args) -> dict:
    import scipy
    import sklearn

    started = time.time()
    project = import_frozen_project(args.project_root)
    trial_root = (
        Path(args.trial_root).expanduser().resolve()
        if args.trial_root is not None
        else Path(project.cfg.TRIAL_ROOT).expanduser().resolve()
    )
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite existing output: %s" % output)
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir is not None
        else output.parent / "feature_cache"
    )

    store, data_hashes, cache_records = load_feature_store(
        project, trial_root, cache_dir,
    )
    fold_index = int(args.fold)
    target = SUBJECTS[fold_index]
    sources = tuple(subject for subject in SUBJECTS if subject != target)

    fixed = run_fixed_models(store, sources, target)
    best_k = run_best_k(store, sources, target)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "protocol": {
            "outer": "12-fold leave-one-subject-out",
            "outer_fold_index": fold_index,
            "outer_target": target,
            "source_subjects": list(sources),
            "source_blocks": list(ALL_BLOCKS),
            "target_blocks": list(ALL_BLOCKS),
            "inner": "11-fold leave-one-source-subject-out",
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
            "candidate_counts_by_k": {
                str(k): len(combinations_for_k(k)) for k in BAND_IDS
            },
            "classifier": (
                "StandardScaler fitted on current fit subjects only, followed "
                "by LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')"
            ),
            "selection_metric": (
                "accuracy averaged equally over 11 inner validation subjects"
            ),
            "tie_break": "one-based band-index lexicographic order",
            "tie_absolute_tolerance": TIE_ATOL,
            "feature_cache_scope": (
                "deterministic within-trial filtering/geometry/spectrum only; "
                "no StandardScaler or LDA state"
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
        "fixed_models": fixed,
        "best_k": best_k,
        "input_sha256": data_hashes,
        "feature_cache": cache_records,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="FaceEMG-11 single-band and source-selected Best-k experiment",
    )
    parser.add_argument(
        "--describe", action="store_true",
        help="print protocol/candidate self-checks without importing the project or data",
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent,
        help="directory containing config.py, dataset.py, and run_mbdg_rs_ablations.py",
    )
    parser.add_argument(
        "--trial-root", type=Path,
        help="directory containing sub-01_trials.npz ... sub-12_trials.npz",
    )
    parser.add_argument("--fold", type=int, choices=range(len(SUBJECTS)))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--cache-dir", type=Path,
        help="new deterministic feature-cache directory; defaults beside --output",
    )
    args = parser.parse_args(argv)
    if not args.describe and (args.fold is None or args.output is None):
        parser.error("--fold and --output are required unless --describe is used")
    return args


def main(argv=None):
    args = parse_args(argv)
    self_check()
    if args.describe:
        print(json.dumps(protocol_description(), indent=2, sort_keys=True))
        return 0
    payload = run_fold(args)
    print(json.dumps({
        "output": str(Path(args.output).expanduser().resolve()),
        "outer_target": payload["protocol"]["outer_target"],
        "fixed_accuracy": {
            key: value["accuracy"]
            for key, value in payload["fixed_models"]["single_band_dual_geometry"].items()
        },
        "selected_best_k": {
            key: value["selected"]["id"] for key, value in payload["best_k"].items()
        },
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

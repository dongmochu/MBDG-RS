"""Feature construction for the classical and geometry baseline benchmark."""

from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

import config as cfg
from feature_utils import relative_group_zscore


BANDS = ((2, 20), (20, 60), (60, 120), (120, 250), (250, 450))


def _td_features(data, threshold=10e-6):
    """Hudgins time-domain features shared by the classical baselines."""
    data = np.asarray(data)
    mav = np.mean(np.abs(data), axis=2)
    wl = np.sum(np.abs(np.diff(data, axis=2)), axis=2)
    sign_change = np.diff(np.signbit(data), axis=2)
    zc_amplitude = np.abs(np.diff(data, axis=2)) > threshold
    zc = np.sum(sign_change & zc_amplitude, axis=2)
    slopes = np.diff(data, axis=2)
    slope_change = np.diff(np.signbit(slopes), axis=2)
    ssc_amplitude = (
        (np.abs(slopes[:, :, :-1]) > threshold)
        & (np.abs(slopes[:, :, 1:]) > threshold)
    )
    ssc = np.sum(slope_change & ssc_amplitude, axis=2)
    return np.concatenate([mav, wl, zc, ssc], axis=1)


def _matrix_log_vector(matrices, regularization):
    """Map trace-normalized covariance matrices to symmetric log-vectors."""
    channels = matrices.shape[1]
    identity = np.eye(channels)
    upper = np.triu_indices(channels)
    off_diagonal = upper[0] != upper[1]
    result = np.empty((len(matrices), len(upper[0])), dtype=np.float64)
    for index, matrix in enumerate(matrices):
        matrix = (1.0 - regularization) * matrix + regularization * identity
        values, vectors = np.linalg.eigh(matrix)
        logged = (vectors * np.log(np.maximum(values, 1e-8))) @ vectors.T
        vector = logged[upper].copy()
        vector[off_diagonal] *= np.sqrt(2.0)
        result[index] = vector
    return result


def _bandpower(data, fs, bands):
    """Return trial-local channel log-power for the requested bands."""
    spectrum = np.fft.rfft(data, axis=2)
    power = np.square(np.abs(spectrum)) / data.shape[2]
    frequencies = np.fft.rfftfreq(data.shape[2], 1.0 / fs)
    features = []
    for low, high in bands:
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            raise ValueError("frequency band %s-%s Hz has no FFT bins" % (low, high))
        features.append(
            np.log(np.maximum(power[:, :, mask].mean(axis=2), 1e-20))
        )
    return np.concatenate(features, axis=1)


def tangent_from_filtered(data, regularization=0.05, corr=False):
    centered = data - data.mean(axis=2, keepdims=True)
    covariance = centered @ np.swapaxes(centered, 1, 2) / (data.shape[2] - 1)
    if corr:
        scale = np.sqrt(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 1e-12))
        covariance = covariance / np.maximum(scale[:, :, None] * scale[:, None, :], 1e-12)
    else:
        trace = np.trace(covariance, axis1=1, axis2=2)
        covariance = covariance / np.maximum(trace[:, None, None], 1e-12) * data.shape[1]
    return _matrix_log_vector(covariance, regularization)


def filter_bank_tangent(data, fs=1000.0, corr=False):
    rows = []
    for low, high in BANDS:
        sos = butter(4, (low, high), btype="bandpass", fs=fs, output="sos")
        rows.append(tangent_from_filtered(sosfiltfilt(sos, data, axis=2), corr=corr))
    return np.concatenate(rows, axis=1)


def build_subject_features(record, cache_path):
    """Return cached trial features for one participant, creating them if needed."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        with np.load(str(cache_path), allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    data = np.asarray(record.data, dtype=np.float64)
    td_relative = relative_group_zscore(np.log1p(np.maximum(_td_features(data), 0)), cfg.CHANNELS)
    spectral = _bandpower(data, cfg.SAMPLE_RATE_HZ, BANDS)
    spectral_topography = relative_group_zscore(spectral, cfg.CHANNELS)
    spectral_cube = spectral.reshape(len(spectral), len(BANDS), cfg.CHANNELS).transpose(0, 2, 1)
    spectral_shape = relative_group_zscore(spectral_cube.reshape(len(spectral), -1), len(BANDS))
    spectral_shape = spectral_shape.reshape(len(spectral), cfg.CHANNELS, len(BANDS)).transpose(0, 2, 1).reshape(len(spectral), -1)
    result = {
        "td_relative": td_relative,
        "spectral_topography": spectral_topography,
        "spectral_shape": spectral_shape,
        "broadband_covariance": tangent_from_filtered(data, corr=False),
        "broadband_correlation": tangent_from_filtered(data, corr=True),
        "filter_bank_covariance": filter_bank_tangent(data, corr=False),
        "filter_bank_correlation": filter_bank_tangent(data, corr=True),
        "labels": record.labels,
        "blocks": record.groups,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), **result)
    return result


def views(item):
    """Named feature representations used by the benchmark runner."""
    spectral = np.concatenate([item["spectral_topography"], item["spectral_shape"]], axis=1)
    return {
        "broadband_spectrum": np.concatenate([item["broadband_covariance"], item["spectral_topography"]], axis=1),
        "correlation_spectrum": np.concatenate([item["broadband_correlation"], item["spectral_topography"]], axis=1),
        "filter_bank_covariance": item["filter_bank_covariance"],
        "filter_bank_correlation": item["filter_bank_correlation"],
        "filter_bank_covariance_spectrum": np.concatenate([item["filter_bank_covariance"], spectral], axis=1),
        "filter_bank_correlation_spectrum": np.concatenate([item["filter_bank_correlation"], spectral], axis=1),
        "filter_bank_dual_geometry_spectrum": np.concatenate([item["filter_bank_covariance"], item["filter_bank_correlation"], spectral], axis=1),
        "time_domain_filter_bank_correlation_spectrum": np.concatenate([item["td_relative"], item["filter_bank_correlation"], spectral], axis=1),
    }

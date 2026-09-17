"""Paths and frozen constants for the final FaceEMG-11 pure-LOSO protocol."""

from pathlib import Path
import os


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("FACEEMG_DATA_ROOT", str(ROOT / "data")))
TRIAL_ROOT = DATA_ROOT / "trials"
RESULT_ROOT = Path(os.environ.get("FACEEMG_RESULT_ROOT", str(ROOT / "results")))

SUBJECTS = tuple("sub-%02d" % index for index in range(1, 13))
LABEL_IDS = tuple(range(1, 12))
SAMPLE_RATE_HZ = 1000.0
CHANNELS = 20
TRIAL_SAMPLES = 1500
RECORDED_BLOCKS = 30

DEFAULT_SEED = 42
FREQUENCY_BANDS_HZ = ((2, 20), (20, 60), (60, 120), (120, 250), (250, 450))

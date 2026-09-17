# FaceEMG-11 data location

Physiological recordings are not included in this Git repository. Download
the dataset from Zenodo:

- https://zenodo.org/records/22137268
- DOI: https://doi.org/10.5281/zenodo.22137268
- License: CC BY 4.0

After extracting `FaceEMG-11.zip`, set `FACEEMG_DATA_ROOT` to the directory
that contains `trials/`, or place the participant files here:

```text
data/trials/sub-01_trials.npz
...
data/trials/sub-12_trials.npz
```

The final paper protocol uses all 30 blocks (330 trials) from each participant.
No target-participant sample or statistic may be used for fitting or selection.


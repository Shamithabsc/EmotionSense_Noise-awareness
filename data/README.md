# Dataset location

Download the RAVDESS **speech audio** archive from the official RAVDESS project page:
https://zenodo.org/records/1188976

Follow the dataset's stated license and terms. Extract the speech audio files under this directory so that `data/ravdess/` contains the actor folders and/or audio files recursively.

The project does not include the audio archive. After extraction, run:

```powershell
python scripts/inspect_dataset.py
```

The filename parser expects the standard seven-part RAVDESS naming convention.

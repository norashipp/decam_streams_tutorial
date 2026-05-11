# Mapping the Milky Way Stellar Halo: Discovering the ATLAS Stream

Tutorial notebook for the LSST DA Regional Workshop — Near-Field Cosmology session.

## Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate decam_streams

# OR install dependencies manually
pip install numpy pandas matplotlib healpy skyproj ipywidgets scipy jupyter
```

## Get the data

The catalog file is too large for git. Download it before running the notebook:

**If you have cluster access** (e.g. UW `epyc`), run the download script:
```bash
python scripts/download_atlas_cutout.py
# writes data/atlas_cutout.parquet (~XXX MB)
```

**Otherwise**, download the pre-made cutout from [link TBD] and place it at `data/atlas_cutout.parquet`.

The other data files (stream tracks, isochrones) are included in the repo.

## Run the tutorial

```bash
jupyter notebook notebooks/tutorial_aau_stream.ipynb
```

## Repository layout

```
notebooks/
    tutorial_aau_stream.ipynb     # main tutorial
    stream_utils.py               # helper functions (coordinate transforms, Hess diagrams, etc.)
data/
    atlas_cutout.parquet          # NOT in git — download separately (see above)
    atlas_track.npy               # ATLAS stream track (RA, Dec)
    aliqa_uma_track.npy           # Aliqa Uma stream track (RA, Dec)
    isochrones/                   # PARSEC Bressan2012 isochrone files (DES/DECam)
scripts/
    download_atlas_cutout.py      # download ATLAS+Aliqa Uma cutout from DELVE DR3 via LSDB
```

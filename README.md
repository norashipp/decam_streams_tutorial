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

Download the data folder from Google Drive and place its contents at `data/` in the repo root:

**[Download data from Google Drive](https://drive.google.com/drive/folders/1wd1SonTZKj-H6hBuchB4ko4vCiiaJP5Y?usp=sharing)**

After downloading, your directory should look like:

```
data/
    atlas_cutout.parquet
    atlas_track.npy
    aliqa_uma_track.npy
    isochrones/
        iso_a10.0_z0.00010.dat
        ...
```

**If you have cluster access** (e.g. UW `epyc`), you can regenerate `atlas_cutout.parquet` yourself instead:
```bash
python scripts/download_atlas_cutout.py
```

## Run the tutorial

```bash
jupyter notebook notebooks/tutorial_aau_stream.ipynb
```

## Repository layout

```
notebooks/
    tutorial_aau_stream.ipynb     # main tutorial
    stream_utils.py               # helper functions (coordinate transforms, Hess diagrams, etc.)
data/                             # NOT in git — download from Google Drive (see above)
    atlas_cutout.parquet
    atlas_track.npy
    aliqa_uma_track.npy
    isochrones/
scripts/
    download_atlas_cutout.py      # download ATLAS+Aliqa Uma cutout from DELVE DR3 via LSDB
```

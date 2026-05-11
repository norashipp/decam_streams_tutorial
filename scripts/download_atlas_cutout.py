#!/usr/bin/env python
"""Download a DELVE DR3 cutout covering the ATLAS and Aliqa Uma streams.

The region covers both arms of the AAU system plus surrounding area for
off-stream background. Output is a single parquet file suitable for the
tutorial notebook.

Usage
-----
python scripts/download_atlas_cutout.py
python scripts/download_atlas_cutout.py --output data/atlas_cutout.parquet

Requires cluster access to the DELVE HATS catalog.
"""

import argparse

import lsdb
import pandas as pd


# ATLAS:     RA  8–31,  Dec -33 to -20
# Aliqa Uma: RA 31–41,  Dec -38 to -33
# Combined box with generous padding for off-stream background
RA_MIN, RA_MAX   = 5, 42
DEC_MIN, DEC_MAX = -42, -14

COLUMNS = [
    'RA', 'DEC',
    'PSF_MAG_APER_8_G_CORRECTED',
    'PSF_MAG_ERR_APER_8_G',
    'PSF_MAG_APER_8_R_CORRECTED',
    'PSF_MAG_ERR_APER_8_R',
    'EXT_XGB',
]

DELVE_PATH = '/epyc/data/delve/dr3/delve_dr3_gold/delve_dr3_gold/'


def main():
    parser = argparse.ArgumentParser(
        description='Download DELVE cutout around ATLAS and Aliqa Uma streams.')
    parser.add_argument(
        '--output', default='data/atlas_cutout.parquet',
        help='Output parquet file path (default: data/atlas_cutout.parquet)')
    args = parser.parse_args()

    print(f'Loading RA ({RA_MIN}–{RA_MAX}), Dec ({DEC_MIN}–{DEC_MAX}) '
          f'from DELVE DR3...')
    df = lsdb.read_hats(
        DELVE_PATH,
        search_filter=lsdb.BoxSearch(
            ra=(RA_MIN, RA_MAX), dec=(DEC_MIN, DEC_MAX)),
        columns=COLUMNS,
    ).compute()
    print(f'  {len(df):,} objects before cuts')

    # Star/galaxy separation and magnitude cuts
    df = df[df['EXT_XGB'] == 0]
    df = df[
        (df['PSF_MAG_APER_8_G_CORRECTED'] > 16) &
        (df['PSF_MAG_APER_8_G_CORRECTED'] < 24)
    ].reset_index(drop=True)
    print(f'  {len(df):,} stars after cuts')

    df.to_parquet(args.output)
    size_mb = __import__('os').path.getsize(args.output) / 1e6
    print(f'Saved to {args.output} ({size_mb:.0f} MB)')


if __name__ == '__main__':
    main()

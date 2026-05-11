#!/usr/bin/env python3
"""Download PARSEC isochrones from the CMD web service for the tutorial.

Fetches all (age, Z) combinations on the tutorial grid and saves them as
iso_a{age}_z{Z:.5f}.dat files in the output directory.  Skips files that
already exist.

USAGE
-----
    python scripts/download_isochrones.py
    python scripts/download_isochrones.py --out-dir data/isochrones

REQUIRES
--------
    pip install requests numpy

NOTE
----
Targets CMD 3.7 (https://stev.oapd.inaf.it/cgi-bin/cmd_3.7) with PARSEC v1.2S
and the DECam photometric system.  Column layout of the output files is
remapped to match the existing repo isochrones (CMD 2.7 format):
  col 9 = g (ABmag), col 10 = r (ABmag), last column = evolutionary stage.
"""

import argparse
import io
import os
import re
import sys
import time

import numpy as np
import requests

CMD_URL  = "https://stev.oapd.inaf.it/cgi-bin/cmd_3.7"
BASE_URL = "https://stev.oapd.inaf.it"

# Tutorial age/metallicity grid
AGES_GYR = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 13.5]
Z_VALUES  = [0.00010, 0.00020, 0.00030, 0.00050, 0.00100]

# CMD 3.7 DECam column layout (0-indexed, from the header line):
#   0  Zini   1  MH     2  logAge  3  Mini  4  int_IMF  5  Mass
#   6  logL   7  logTe  8  logg    9  label (evol stage)
#  10  McoreTP  11  C_O  12-16 period0-4  17  pmode  18  Mloss
#  19  tau1m  20  X  21  Y  22  Xc  23  Xn  24  Xo  25  Cexcess
#  26  Z_act  27  mbolmag  28  umag  29  gmag  30  rmag
#  31  imag   32  zmag  33  Ymag
#
# Target (CMD 2.7-compatible) layout:
#  0  Z  1  logAge  2  Mini  3  Mass  4  logL  5  logTe  6  logg
#  7  mbol  8  u  9  g  10  r  11  i  12  z  13  Y  14  int_IMF  15  stage
_REMAP = [0, 2, 3, 5, 6, 7, 8, 27, 28, 29, 30, 31, 32, 33, 4, 9]


def _form_params(age_gyr, Z):
    log_age = np.log10(age_gyr * 1e9)
    return {
        "cmd_version"      : "3.7",
        "track_parsec"     : "parsec_CAF09_v1.2S",
        "track_colibri"    : "parsec_CAF09_v1.2S_S_LMC_08_web",
        "track_postagb"    : "no",
        "n_inTPC"          : 10,
        "eta_reimers"      : 0.2,
        "kind_interp"      : 1,
        "kind_postagb"     : -1,
        "imf_file"         : "tab_imf/imf_chabrier_lognormal.dat",
        "photsys_file"     : "YBC_tab_mag_odfnew/tab_mag_decam.dat",
        "photsys_version"  : "YBCnewVega",
        "dust_sourceM"     : "nodustM",
        "dust_sourceC"     : "nodustC",
        "kind_mag"         : 2,
        "kind_dust"        : 0,
        "extinction_av"    : 0.0,
        "extinction_coeff" : "constant",
        "extinction_curve" : "cardelli",
        "kind_LPV"         : 3,
        "isoc_isagelog"    : 1,
        "isoc_lagelow"     : f"{log_age:.4f}",
        "isoc_lageupp"     : f"{log_age:.4f}",
        "isoc_dlage"       : 0.0,
        "isoc_ismetlog"    : 0,
        "isoc_zlow"        : Z,
        "isoc_zupp"        : Z,
        "isoc_dz"          : 0.0,
        "output_kind"      : 0,
        "output_evstage"   : 1,
        ".cgifields"       : ["dust_sourceC", "dust_sourceM", "extinction_coeff",
                              "extinction_curve", "isoc_isagelog", "isoc_ismetlog",
                              "kind_LPV", "output_gzip", "output_kind", "photsys_version",
                              "track_colibri", "track_omegai", "track_parsec", "track_postagb"],
        "submit_form"      : "Submit",
    }


def _extract_file_url(html):
    for pat in [
        r"(https?://stev\.oapd\.inaf\.it/tmp/output\w+\.dat)",
        r'href=[""]?(\.\./tmp/output\w+\.dat)["">\s]',
        r'href=[""]?(tmp/output\w+\.dat)["">\s]',
        r'(\.\.?/tmp/output\w+\.dat)',
    ]:
        m = re.search(pat, html)
        if m:
            url = m.group(1)
            if url.startswith("http"):
                return url
            return BASE_URL + "/" + url.lstrip("./")
    return None


def _reformat(raw_text, age_gyr, Z):
    """Remap CMD 3.7 columns to the CMD 2.7-compatible layout."""
    header_lines = []
    data_rows = []
    for line in raw_text.splitlines():
        if line.startswith("#"):
            header_lines.append(line)
        elif line.strip():
            data_rows.append(line.split())

    if not data_rows:
        return None

    arr = np.array(data_rows, dtype=float)
    remapped = arr[:, _REMAP]

    buf = io.StringIO()
    # Write a compact compatible header
    buf.write(f"# Downloaded from CMD 3.7 (https://stev.oapd.inaf.it/cmd), "
              f"PARSEC v1.2S, DECam ABmags (YBCnewVega)\n")
    buf.write(f"#\tIsochrone  Z = {Z:.5f}\tAge = {age_gyr*1e9:.4e} yr\n")
    buf.write(
        "#      Z\tlog(age/yr)\tM_ini\tM_act\tlogL/Lo\tlogTe\tlogG\t"
        "mbol\tu\tg\tr\ti\tz\tY\tint_IMF\tstage\n"
    )
    np.savetxt(buf, remapped, fmt="%.6g")
    return buf.getvalue()


def download_one(age_gyr, Z, out_dir, retries=3):
    fname   = f"iso_a{age_gyr}_z{Z:.5f}.dat"
    outpath = os.path.join(out_dir, fname)

    if os.path.exists(outpath):
        print(f"  {fname}  already exists, skipping")
        return True

    params = _form_params(age_gyr, Z)
    for attempt in range(retries):
        try:
            r = requests.post(CMD_URL, data=params, timeout=60)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  attempt {attempt+1}/{retries} failed: {e}")
            time.sleep(5)
            continue

        if "CMD 3.7 input form" in r.text:
            print(f"  {fname}  ERROR: server returned the input form (bad parameters?)")
            return False

        file_url = _extract_file_url(r.text)
        if not file_url:
            print(f"  {fname}  ERROR: could not find output URL in CMD response")
            return False

        try:
            dat = requests.get(file_url, timeout=60)
            dat.raise_for_status()
        except requests.RequestException as e:
            print(f"  {fname}  ERROR downloading file: {e}")
            return False

        reformatted = _reformat(dat.text, age_gyr, Z)
        if reformatted is None:
            print(f"  {fname}  ERROR: no data rows in downloaded file")
            return False

        with open(outpath, "w") as fh:
            fh.write(reformatted)
        print(f"  {fname}  OK  ({os.path.getsize(outpath):,} bytes)")
        return True

    print(f"  {fname}  FAILED after {retries} attempts")
    return False


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out-dir", default="data/isochrones",
                    help="Output directory (default: data/isochrones)")
    ap.add_argument("--ages", nargs="+", type=float, default=AGES_GYR,
                    metavar="AGE", help="Ages in Gyr")
    ap.add_argument("--z-values", nargs="+", type=float, default=Z_VALUES,
                    metavar="Z", help="Metallicity Z values")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    total = len(args.ages) * len(args.z_values)
    print(f"Downloading up to {total} isochrones → {args.out_dir}/")

    ok = fail = 0
    for age in sorted(args.ages):
        for Z in sorted(args.z_values):
            if download_one(age, Z, args.out_dir):
                ok += 1
            else:
                fail += 1
            time.sleep(1)   # be polite to the server

    print(f"\n{ok} downloaded/skipped, {fail} failed")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()

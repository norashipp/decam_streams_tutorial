"""Helper functions for the ATLAS stream tutorial.

All the machinery lives here so the notebook stays readable.
Students interact with simple parameters; this module handles
coordinate transforms, Hess diagrams, matched filters, and maps.
"""

import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import healpy as hp


# ---------------------------------------------------------------------------
# Stream coordinate frame
# ---------------------------------------------------------------------------

def _unit_vec(ra_deg, dec_deg):
    ra  = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])


def rotation_matrix_from_track(track_ra_dec):
    """Build a rotation matrix from the two endpoints of a stream track.

    Parameters
    ----------
    track_ra_dec : ndarray, shape (N, 2)
        Track coordinates in degrees, columns [RA, Dec].

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix: R @ v_icrs -> (x, y, z) in stream frame.
    """
    v1 = _unit_vec(track_ra_dec[0,  0], track_ra_dec[0,  1])
    v2 = _unit_vec(track_ra_dec[-1, 0], track_ra_dec[-1, 1])

    z = np.cross(v1, v2)
    z /= np.linalg.norm(z)

    x = v1 + v2
    x /= np.linalg.norm(x)

    y = np.cross(z, x)
    y /= np.linalg.norm(y)

    return np.array([x, y, z])


def icrs_to_stream(ra_deg, dec_deg, R):
    """Convert ICRS (RA, Dec) arrays to stream (phi1, phi2) using R.

    Parameters
    ----------
    ra_deg, dec_deg : array_like
    R : ndarray, shape (3, 3)

    Returns
    -------
    phi1, phi2 : ndarray  (degrees)
    """
    ra  = np.deg2rad(np.asarray(ra_deg,  dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    xyz = np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])                        # shape (3, N)
    rot  = R @ xyz            # shape (3, N)
    phi1 = np.rad2deg(np.arctan2(rot[1], rot[0]))
    phi2 = np.rad2deg(np.arcsin(np.clip(rot[2], -1, 1)))
    return phi1, phi2


def stream_to_icrs(phi1_deg, phi2_deg, R):
    """Convert stream (phi1, phi2) to ICRS (RA, Dec) using R.

    Parameters
    ----------
    phi1_deg, phi2_deg : array_like
    R : ndarray, shape (3, 3)  rotation matrix from rotation_matrix_from_track

    Returns
    -------
    ra, dec : ndarray  (degrees, RA in [0, 360))
    """
    phi1 = np.deg2rad(np.asarray(phi1_deg, dtype=float))
    phi2 = np.deg2rad(np.asarray(phi2_deg, dtype=float))
    xyz = np.array([
        np.cos(phi2) * np.cos(phi1),
        np.cos(phi2) * np.sin(phi1),
        np.sin(phi2),
    ])
    xyz_icrs = R.T @ xyz
    dec = np.rad2deg(np.arcsin(np.clip(xyz_icrs[2], -1, 1)))
    ra  = np.rad2deg(np.arctan2(xyz_icrs[1], xyz_icrs[0])) % 360
    return ra, dec


def add_stream_coords(cat, atlas_track):
    """Add phi1/phi2 stream coordinates to the catalog in-place.

    Parameters
    ----------
    cat : pd.DataFrame
        Must have 'RA' and 'DEC' columns.
    atlas_track : ndarray, shape (N, 2)
        Track from ``np.load('data/atlas_track.npy')``.

    Returns
    -------
    cat : pd.DataFrame  (modified in-place, also returned for chaining)
    R   : ndarray, shape (3, 3)  rotation matrix
    """
    R = rotation_matrix_from_track(atlas_track)
    cat['phi1'], cat['phi2'] = icrs_to_stream(cat['RA'].values,
                                               cat['DEC'].values, R)
    return cat, R


# ---------------------------------------------------------------------------
# Hess diagram
# ---------------------------------------------------------------------------

def compute_hess(
    cat,
    phi1_range=(-15, 15),
    phi2_center=0.0,
    phi2_on_hw=1.0,
    phi2_off_sep=0.5,
    phi2_off_hw=2.0,
    use_above=True,
    use_below=True,
    g_bins=None,
    color_bins=None,
):
    """Compute an area-normalised on-minus-off Hess diagram in stream coords.

    The on-stream strip is  phi2_center ± phi2_on_hw.
    The above off-stream strip (if use_above) is above the on-stream strip,
    separated by phi2_off_sep, with width phi2_off_hw.
    The below off-stream strip (if use_below) is the mirror below.

    Parameters
    ----------
    cat : pd.DataFrame
        Must have 'phi1', 'phi2', 'g', 'color' columns.
    phi1_range : (float, float)
        Along-stream longitude range (degrees).
    phi2_center : float
        Center of the on-stream strip in phi2 (degrees).
    phi2_on_hw : float
        Half-width of the on-stream strip (degrees).
    phi2_off_sep : float
        Gap between on-stream edge and off-stream strip (degrees).
    phi2_off_hw : float
        Width of each off-stream strip (degrees).
    use_above, use_below : bool
        Toggle the above- and below-stream off regions independently.
    g_bins, color_bins : array_like, optional
        Bin edges.  Defaults to sensible DECam values.

    Returns
    -------
    diff       : ndarray  area-normalised difference CMD
    g_bins     : ndarray
    color_bins : ndarray
    on_region  : dict  with keys phi1_range, phi2_range, area_deg2
    off_region : dict  with keys area_deg2 (total off area)
    """
    if g_bins is None:
        g_bins = np.arange(16, 24.1, 0.2)
    if color_bins is None:
        color_bins = np.arange(-0.2, 1.21, 0.04)

    phi1_lo, phi1_hi = phi1_range
    phi2_on_lo = phi2_center - phi2_on_hw
    phi2_on_hi = phi2_center + phi2_on_hw

    phi2_above_inner = phi2_on_hi + phi2_off_sep
    phi2_above_outer = phi2_above_inner + phi2_off_hw
    phi2_below_inner = phi2_on_lo - phi2_off_sep
    phi2_below_outer = phi2_below_inner - phi2_off_hw

    phi1_vals = cat['phi1'].values
    phi2_vals = cat['phi2'].values
    phi1_mask = (phi1_vals >= phi1_lo) & (phi1_vals <= phi1_hi)

    on_mask  = phi1_mask & (phi2_vals >= phi2_on_lo) & (phi2_vals <= phi2_on_hi)
    off_mask = np.zeros(len(cat), dtype=bool)
    if use_above:
        off_mask |= (phi2_vals >= phi2_above_inner) & (phi2_vals <= phi2_above_outer)
    if use_below:
        off_mask |= (phi2_vals >= phi2_below_outer) & (phi2_vals <= phi2_below_inner)
    off_mask &= phi1_mask

    on_hess,  _, _ = np.histogram2d(
        cat['g'].values[on_mask],
        cat['color'].values[on_mask],
        bins=[g_bins, color_bins],
    )
    off_hess, _, _ = np.histogram2d(
        cat['g'].values[off_mask],
        cat['color'].values[off_mask],
        bins=[g_bins, color_bins],
    )

    def _area(phi2_lo, phi2_hi):
        dra_rad  = np.deg2rad(phi1_hi - phi1_lo)
        ddec_rad = np.deg2rad(phi2_hi - phi2_lo)
        dec_mid  = np.deg2rad(0.5 * (phi2_lo + phi2_hi))
        return np.degrees(dra_rad * np.cos(dec_mid)) * np.degrees(ddec_rad)

    on_area  = _area(phi2_on_lo, phi2_on_hi)
    off_area = 0.0
    if use_above:
        off_area += _area(phi2_above_inner, phi2_above_outer)
    if use_below:
        off_area += _area(phi2_below_outer, phi2_below_inner)

    if off_area > 0:
        diff = on_hess / on_area - off_hess / off_area
    else:
        diff = on_hess / on_area

    on_region  = dict(phi1_range=phi1_range,
                      phi2_range=(phi2_on_lo, phi2_on_hi),
                      area_deg2=on_area)
    off_region = dict(phi1_range=phi1_range,
                      area_deg2=off_area)

    return diff, g_bins, color_bins, on_region, off_region


# ---------------------------------------------------------------------------
# Isochrone utilities
# ---------------------------------------------------------------------------

def load_isochrone(path, max_stage=3):
    """Load a PARSEC isochrone (Bressan 2012) .dat file.

    Returns (g_abs, color) where color = g_abs - r_abs,
    restricted to evolutionary stages <= max_stage (excludes TP-AGB).
    """
    iso = np.loadtxt(path, comments='#')
    sel   = iso[:, -1] <= max_stage
    g_abs = iso[sel, 9]
    r_abs = iso[sel, 10]
    return g_abs, g_abs - r_abs


def overplot_isochrone(ax, iso_dir, age, Z, mu, g_bins, **kwargs):
    """Plot an isochrone on a CMD axes."""
    path = os.path.join(iso_dir, f'iso_a{age}_z{Z}.dat')
    if not os.path.exists(path):
        return
    g_abs, color = load_isochrone(path)
    g_app = g_abs + mu
    mask  = (g_app > g_bins[0]) & (g_app < g_bins[-1])
    ax.plot(color[mask], g_app[mask], **kwargs)


# ---------------------------------------------------------------------------
# Matched filter
# ---------------------------------------------------------------------------

def _delve_err(g):
    """DES photometric error model σ(g)."""
    return 0.00109 + np.exp((g - 27.09) / 1.09)


def apply_matched_filter(cat, iso_dir, age, Z, mu,
                         C=0.07, E=2.0, gmin=None, gmax=None):
    """Select stars within a colour window around an isochrone.

    Parameters
    ----------
    cat : pd.DataFrame   Must have 'g' and 'color' columns.
    iso_dir : str        Directory containing isochrone .dat files.
    age, Z  : str        Isochrone parameters (must match a file in iso_dir).
    mu      : float      Distance modulus.
    C       : float      Symmetric fixed colour half-width.
    E       : float      Multiplier on the photometric error envelope.
    gmin, gmax : float   Magnitude range; stars outside are excluded.

    Returns
    -------
    mask : ndarray of bool
    """
    path = os.path.join(iso_dir, f'iso_a{age}_z{Z}.dat')
    g_abs, iso_color = load_isochrone(path)
    g_app = g_abs + mu
    sort  = np.argsort(g_app)
    color_at_g = interp1d(g_app[sort], iso_color[sort],
                          bounds_error=False, fill_value=np.nan)

    g_cat    = cat['g'].values
    expected = color_at_g(g_cat)
    err      = _delve_err(g_cat)
    mask = (
        (cat['color'].values > expected - C - E * err) &
        (cat['color'].values < expected + C + E * err) &
        np.isfinite(expected)
    )
    if gmin is not None:
        mask &= (g_cat >= gmin)
    if gmax is not None:
        mask &= (g_cat <= gmax)
    return mask


def filter_window_edges(iso_dir, age, Z, mu, g_bins,
                        C=0.07, E=2.0, gmin=None, gmax=None, n=300):
    """Return (g_grid, blue_edge, red_edge) arrays for plotting the filter."""
    path = os.path.join(iso_dir, f'iso_a{age}_z{Z}.dat')
    g_abs, iso_color = load_isochrone(path)
    g_app = g_abs + mu
    sort  = np.argsort(g_app)
    color_at_g = interp1d(g_app[sort], iso_color[sort],
                          bounds_error=False, fill_value=np.nan)
    g_lo = gmin if gmin is not None else g_bins[0]
    g_hi = gmax if gmax is not None else g_bins[-1]
    g_grid   = np.linspace(g_lo, g_hi, n)
    expected = color_at_g(g_grid)
    err      = _delve_err(g_grid)
    return g_grid, expected - C - E * err, expected + C + E * err


# ---------------------------------------------------------------------------
# HEALPix maps
# ---------------------------------------------------------------------------

def stars_to_hpxmap(df, nside, fwhm_deg, ra_col='RA', dec_col='DEC'):
    """Bin stars into a smoothed HEALPix map (NESTED ordering)."""
    ipix = hp.ang2pix(nside, df[ra_col].values, df[dec_col].values,
                      lonlat=True, nest=True)
    hpx = np.zeros(hp.nside2npix(nside))
    np.add.at(hpx, ipix, 1)
    return hp.reorder(
        hp.smoothing(hp.reorder(hpx.astype(float), n2r=True),
                     fwhm=np.radians(fwhm_deg)),
        r2n=True,
    )


def normalize_channel(arr, lo=1, hi=99.5):
    """Scale a map to [0, 1] for RGB compositing."""
    finite = arr[np.isfinite(arr) & (arr > 0)]
    vmin   = np.percentile(finite, lo)
    vmax   = np.percentile(finite, hi)
    return np.clip((arr - vmin) / (vmax - vmin), 0, 1)

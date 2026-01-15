from __future__ import annotations
"""
Make a 10x10 grid on the sky, map each grid cell to a HEALPix pixel (at a chosen nside),
compute star counts (using your existing count_stars_in_healpix_bgm + caching),
and plot a color-scaled “starcount map”.

This does NOT require full-sky healpy plotting; it’s just a rectangular lon/lat patch plot.

Assumptions:
- Galactic coordinates (l,b) in degrees.
- You already have:
    - HealpixQuery dataclass
    - BGMUWS client
    - count_stars_in_healpix_bgm(hpq, bgm, cache_dir=..., extra_params=..., mag_col=..., mag_max=...)
"""

"""
10x10 Galactic lon/lat grid mapped to HEALPix pixels,
using the Besançon UWS client with caching,
and plotted as a color-scaled starcount map.
"""

"""
10x10 Galactic lon/lat grid mapped to HEALPix pixels,
using the Besançon UWS client with caching,
and plotted as a color-scaled starcount map.
"""



import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# -------------------------------------------------
# IMPORT FROM YOUR EXISTING BESANCON CLIENT CODE
# -------------------------------------------------
from queryModelWithClient import (
    BGMUWS,
    HealpixQuery,
    count_stars_in_healpix_bgm,
)

# -------------------------------------------------
# GRID DEFINITION
# -------------------------------------------------
def make_lonlat_grid(
    lon_center: float,
    lat_center: float,
    lon_width: float,
    lat_width: float,
    nx: int = 10,
    ny: int = 10,
):
    lon_edges = np.linspace(lon_center - lon_width / 2,
                            lon_center + lon_width / 2,
                            nx + 1)
    lat_edges = np.linspace(lat_center - lat_width / 2,
                            lat_center + lat_width / 2,
                            ny + 1)

    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])

    LonC, LatC = np.meshgrid(lon_centers, lat_centers)  # (ny, nx)
    return lon_edges, lat_edges, LonC, LatC


def grid_centers_to_healpix(
    LonC: np.ndarray,
    LatC: np.ndarray,
    nside: int,
    nest: bool = False,
) -> np.ndarray:
    """
    Convert Galactic (l,b) grid centers to HEALPix ipix.
    """
    theta = np.deg2rad(90.0 - LatC)  # colatitude
    phi = np.deg2rad(LonC % 360.0)
    return hp.ang2pix(nside, theta, phi, nest=nest)


# -------------------------------------------------
# STARCOUNT GRID COMPUTATION
# -------------------------------------------------
def compute_starcount_grid(
    *,
    bgm: BGMUWS,
    nside: int,
    lon_center: float,
    lat_center: float,
    lon_width: float,
    lat_width: float,
    nx: int = 10,
    ny: int = 10,
    nest: bool = False,
    cache_dir=".bgm_ws_cache",
    extra_params: dict | None = None,
    mag_col: str | None = None,
    mag_max: float | None = None,
):
    lon_edges, lat_edges, LonC, LatC = make_lonlat_grid(
        lon_center, lat_center, lon_width, lat_width, nx, ny
    )

    ipix = grid_centers_to_healpix(LonC, LatC, nside=nside, nest=nest)
    counts = np.full(ipix.shape, np.nan, dtype=float)

    # Deduplicate HEALPix pixels to avoid duplicate jobs
    cache: dict[int, int] = {}

    for (j, i), p in np.ndenumerate(ipix):
        p = int(p)
        if p not in cache:
            hpq = HealpixQuery(
                nside=nside,
                ipix=p,
                nest=nest,
                frame="galactic",
            )
            cache[p] = count_stars_in_healpix_bgm(
                hpq,
                bgm,
                cache_dir=cache_dir,
                extra_params=extra_params,
                mag_col=mag_col,
                mag_max=mag_max,
            )
        counts[j, i] = cache[p]

    return lon_edges, lat_edges, counts, ipix


# -------------------------------------------------
# PLOTTING
# -------------------------------------------------
def plot_starcount_map(
    lon_edges,
    lat_edges,
    counts,
    title="Besançon starcount map (10×10 grid)",
):
    fig, ax = plt.subplots(figsize=(8, 6))

    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        counts,
        shading="auto",
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Star count (per HEALPix pixel)")

    ax.set_xlabel("Galactic longitude l [deg]")
    ax.set_ylabel("Galactic latitude b [deg]")
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# EXAMPLE USAGE
# -------------------------------------------------
if __name__ == "__main__":
    import os

    bgm = BGMUWS(
        user=os.environ["BGM_USER"],
        password=os.environ["BGM_PASS"],
    )

    lon_edges, lat_edges, counts, ipix = compute_starcount_grid(
        bgm=bgm,
        nside=128,
        lon_center=90.5,
        lat_center=30.5,
        lon_width=1.0,
        lat_width=1.0,
        nx=10,
        ny=10,
        cache_dir=".bgm_ws_cache",
        extra_params={
            "SOLI": "1",
            "ref_filter": "V",
            "band_max": "20.,99.,99.,99.,99.,99.,99.,99.,99.",
        },
    )

    plot_starcount_map(lon_edges, lat_edges, counts)

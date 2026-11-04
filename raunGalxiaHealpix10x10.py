#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
from fullRun_galaxia_crowding import run_galaxia_stream, crowding_limited_depth

# IMPORT these from your existing module or paste them here:
# from your_module import run_galaxia_stream, crowding_limited_depth


def lb_to_theta_phi(l_deg: np.ndarray, b_deg: np.ndarray):
    """
    Convert Galactic lon/lat (deg) to healpy angles (theta, phi) in radians.
    healpy uses theta=colatitude, phi=longitude.
    """
    l_deg = np.asarray(l_deg, dtype=float)
    b_deg = np.asarray(b_deg, dtype=float)
    theta = np.radians(90.0 - b_deg)
    phi = np.radians(l_deg % 360.0)
    return theta, phi


def make_plane_grid_10x10(
    l_center: float = 30.0,
    b_center: float = 0.0,
    l_width: float = 20.0,   # degrees
    b_width: float = 10.0,    # degrees
    n_l: int = 100,
    n_b: int = 100,
):
    """
    Create a 10x10 grid of (l,b) points over a rectangular region.
    Default: a 10°x5° box centered near the plane at l=30°, b=0°.
    """
    l_min = l_center - l_width / 2.0
    l_max = l_center + l_width / 2.0
    b_min = b_center - b_width / 2.0
    b_max = b_center + b_width / 2.0

    l_vals = np.linspace(l_min, l_max, n_l)
    b_vals = np.linspace(b_min, b_max, n_b)

    LL, BB = np.meshgrid(l_vals, b_vals, indexing="xy")
    return LL.ravel(), BB.ravel(), (l_min, l_max, b_min, b_max)


def healpix_area_deg2(nside: int) -> float:
    area_sr = hp.nside2pixarea(nside)
    return float(area_sr * (180.0 / np.pi) ** 2)


def run_crowding_for_ipix_list(
    *,
    nside: int,
    ipix_list: np.ndarray,
    outdir: str,
    k_max: float,
    k_min: float,
    mag_key: str,
    fwhm_arcsec: float,
    k: float,
    mode: str = "poisson",
    p_blend: float = 0.10,
    beams_per_source: float = 30.0,
    skip_existing: bool = True,
    galaxia_exe: str = str(Path.home() / ".local/bin/galaxia"),
):
    """
    For each HEALPix pixel index, run Galaxia at pixel center and compute crowding limit.
    Returns DataFrame with one row per ipix.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    area_deg2 = healpix_area_deg2(nside)

    rows = []
    for ipix in ipix_list:
        # Pixel center in healpy angles
        theta, phi = hp.pix2ang(nside, int(ipix))
        l_deg = float(np.degrees(phi))
        b_deg = float(90.0 - np.degrees(theta))

        workdir = outdir / f"hp_{int(ipix):06d}"
        workdir.mkdir(parents=True, exist_ok=True)

        stem = f"hp{int(ipix):06d}_n{nside}_Ks{k_max:g}"
        ebf_path = workdir / f"{stem}.ebf"

        if skip_existing and ebf_path.exists():
            rc = 0
        else:
            ebf_path, _, rc = run_galaxia_stream(
                workdir=str(workdir),
                output_file_stem=stem,
                l_deg=l_deg,
                b_deg=b_deg,
                survey_area_deg2=area_deg2,
                k_max=k_max,
                galaxia_exe=galaxia_exe,
            )

        res = crowding_limited_depth(
            ebf_file=str(ebf_path),
            mag_key=mag_key,
            survey_area_deg2=area_deg2,
            fwhm_arcsec=fwhm_arcsec,
            k=k,
            mode=mode,
            p_blend=p_blend,
            beams_per_source=beams_per_source,
            dmag=0.1,
            mag_min=k_min,
            mag_max=k_max,
        )

        m_lim = res["m_lim"]
        # Use NaN in the map if not reached
        m_lim_num = np.nan if (m_lim is None) else float(m_lim)

        dens_deg2 = res["N_total"] / area_deg2
        dens_arcsec2 = dens_deg2 / (3600.0**2)

        rows.append({
            "nside": nside,
            "ipix": int(ipix),
            "l_deg": l_deg,
            "b_deg": b_deg,
            "area_deg2": area_deg2,
            "k_max": float(k_max),
            "mag_key": mag_key,
            "N_total": int(res["N_total"]),
            "density_deg2": float(dens_deg2),
            "density_arcsec2": float(dens_arcsec2),
            "mode": mode,
            "p_blend": float(p_blend) if mode == "poisson" else np.nan,
            "beams_per_source": float(beams_per_source) if mode == "beams" else np.nan,
            "fwhm_arcsec": float(fwhm_arcsec),
            "k": float(k),
            "m_lim": m_lim_num,
            "galaxia_exit_code": int(rc),
            "ebf_path": str(ebf_path),
        })

        print(f"[ipix {int(ipix):06d}] (l,b)=({l_deg:6.2f},{b_deg:6.2f})  N={res['N_total']:,}  m_lim={m_lim_num}")

    return pd.DataFrame(rows)

def make_zoom_cartview_plot(
    *,
    healpix_map: np.ndarray,
    box: tuple[float, float, float, float],   # (l_min, l_max, b_min, b_max)
    title: str,
    out_png: str | None = None,
    unit: str = "mag",
    min_val: float | None = None,
    max_val: float | None = None,
):
    """
    Zoomed Cartesian view in Galactic coordinates for a rectangular lon/lat box.
    """
    l_min, l_max, b_min, b_max = box

    # cartview: lonra/latra in degrees; coord='G' means Galactic
    hp.cartview(
        healpix_map,
        coord="G",
        lonra=[l_min, l_max],
        latra=[b_min, b_max],
        title=title,
        unit=unit,
        min=min_val,
        max=max_val,
        badcolor="lightgray",  # UNSEEN
        notext=False,
        cbar=True,
        return_projected_map=False,
    )
    hp.graticule()

    if out_png:
        plt.savefig(out_png, dpi=250, bbox_inches="tight")
        print(f"Wrote zoom plot: {out_png}")
def make_healpix_map_and_plot(
    *,
    nside: int,
    df: pd.DataFrame,
    value_col: str = "m_lim",
    title: str = "Crowding-limited depth (K_s band, DCMC)",
    out_png: str | None = None,
    unit: str = "mag (Ks)",
    min_val: float | None = None,
    max_val: float | None = None,
):
    """
    Build a full-sky HEALPix map array with UNSEEN except for computed pixels, and plot it.
    """
    npix = hp.nside2npix(nside)
    m = np.full(npix, hp.UNSEEN, dtype=float)

    for _, row in df.iterrows():
        ipix = int(row["ipix"])
        val = float(row[value_col])
        if np.isfinite(val):
            m[ipix] = val

    # Plot (healpy chooses default colormap; we don't hard-code colors)
    hp.mollview(
        m,
        title=title,
        unit=unit,
        min=min_val,
        max=max_val,
        badcolor="lightgray",   # only affects UNSEEN
    )
    hp.graticule()

    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"Wrote plot: {out_png}")

    return m


if __name__ == "__main__":
    # -------------------------
    # USER SETTINGS
    # -------------------------
    NSIDE = 128
    OUTDIR = "/home/balog/Work/GaiaNIR/Galaxia/galaxia_runs/healpix_n128_plane_100x100_Ks_22012026"

    # Choose a 10x10 region in the Galactic plane:
    # Default: centered at l=30°, b=0°, spanning 10° in l and 5° in b.
    l_grid, b_grid, box = make_plane_grid_10x10(
        l_center=30.0,
        b_center=0.0,
        l_width=20.0,
        b_width=10.0,
        n_l=100,
        n_b=100,
    )
    print("Grid box (l_min, l_max, b_min, b_max) =", box)

    # Science settings
    K_MAX = 30.0
    MAG_KEY = "/dcmc_ks"     # Galaxia EBF key
    FWHM_ARCSEC = 0.2
    K = 1.0
    MODE = "poisson"
    P_BLEND = 0.10

    # Map grid points -> HEALPix pixels (unique ipix list)
    theta, phi = lb_to_theta_phi(l_grid, b_grid)
    ipix = hp.ang2pix(NSIDE, theta, phi, nest=False)
    ipix_unique = np.unique(ipix)

    print(f"100x100 grid points: {l_grid.size} -> unique healpix pixels: {ipix_unique.size}")

    # -------------------------
    # RUN ANALYSIS
    # -------------------------
    df = run_crowding_for_ipix_list(
        nside=NSIDE,
        ipix_list=ipix_unique,
        outdir=OUTDIR,
        k_max=K_MAX,
        k_min=-100.0,
        mag_key=MAG_KEY,
        fwhm_arcsec=FWHM_ARCSEC,
        k=K,
        mode=MODE,
        p_blend=P_BLEND,
        skip_existing=True,
    )

    # Save results table
    out_csv = str(Path(OUTDIR) / "results_plane_100x100.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote results: {out_csv}")

    # -------------------------
    # PLOT HEALPIX MAP
    # -------------------------
    out_png = str(Path(OUTDIR) / "m_lim_healpix_mollview.png")

    # Mollweide (all-sky)
    out_png_allsky = str(Path(OUTDIR) / "m_lim_healpix_mollview.png")
    m = make_healpix_map_and_plot(
        nside=NSIDE,
        df=df,
        value_col="m_lim",
        title=f"Crowding-limited depth ({MODE}, p={P_BLEND})  NSIDE={NSIDE}",
        out_png=out_png_allsky,
        unit="mag",
    )

    # Zoomed Cartesian view (your 10x10 plane box)
    out_png_zoom = str(Path(OUTDIR) / "m_lim_healpix_cartview_zoom.png")
    make_zoom_cartview_plot(
        healpix_map=m,
        box=box,  # <-- from make_plane_grid_10x10()
        title=f"Zoom: Crowding-limited depth ({MODE}, p={P_BLEND})",
        out_png=out_png_zoom,
        unit="mag",
    )

    #_m = make_healpix_map_and_plot(
    #    nside=NSIDE,
    #    df=df,
    #    value_col="m_lim",
    #    title=f"Crowding-limited depth ({MODE}, p={P_BLEND})  NSIDE={NSIDE}",
    #    out_png=out_png,
    #    unit="mag",
    #)

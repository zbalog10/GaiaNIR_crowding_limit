#!/usr/bin/env python3
"""
recalc_crowding_from_ebf.py

Recompute crowding-limited magnitude per HEALPix directory from Galaxia EBF outputs,
using distance modulus + extinction correction (Galaxia website prescription).

Outputs:
- results_recalc.csv (one row per ipix)
- per-healpix diagnostic PNG plots:
  1) abs mags (2-panel)
  2) corrected mags (2-panel)
  3) distance + reddening (2x2)

Assumptions / intent:
- The stored band magnitude column (MAG_ABS_KEY) behaves like an absolute magnitude M_band.
- Apparent/“corrected” magnitude reconstructed via:
    m = M + 5*log10(100*r_kpc) + f_band * E(B-V)
- E(B-V) is taken as:
    exbv_schlegel (preferred)  OR  0.53 * exbv_solar
- Crowding limit defined by Poisson model:
    P(>=1 neighbor) = 1 - exp(-rho(<m) * A) >= P_CRIT
"""

from __future__ import annotations

from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
import healpy as hp
import matplotlib

from raunGalxiaHealpix10x10 import make_plane_grid_10x10

matplotlib.use("Agg")  # PNG only; avoids PyCharm backend issues
import matplotlib.pyplot as plt

import ebf


# =========================
# USER CONFIGURATION
# =========================

ROOT_DIR = Path(
    "/home/balog/Work/GaiaNIR/Galaxia/galaxia_runs/healpix_n128_plane_100x100_Ks_22012026"
)

NSIDE = 128
OUTDIR = ROOT_DIR

# Zoom box in Galactic coordinates (l_min, l_max, b_min, b_max)
# If you already have a "box" from your plane grid script, paste it here.
#BOX = (0.0, 60.0, -5.0, 5.0)   # example: a slice around the plane
OUT_CSV = ROOT_DIR / "results_recalc.csv"

EBF_GLOB = "*.ebf"

# “Absolute magnitude” column used for the reconstruction.
# (You currently use /dcmc_ks here; if you later switch back to /mag0, just change this.)
MAG_ABS_KEY = "/dcmc_ks"

DIST_KPC_KEY = "/rad"
EBV_SCHLEGEL_KEY = "/exbv_schlegel"
EBV_SOLAR_KEY = "/exbv_solar"

# extinction coefficient: A_band / E(B-V)
F_BAND = 0.367

# scale exbv_solar -> physical E(B-V)
EBV_SOLAR_SCALE = 0.53

# NSIDE=128 pixel area
AREA_DEG2_DEFAULT = 0.2098234113

# crowding model
P_CRIT = 0.10
FWHM_ARCSEC = 0.2
K_SCALE = 1.0

# diagnostics
DMAG = 0.1
MAKE_PLOTS = True
MODE = "recalc"
P_BLEND = P_CRIT   # same threshold; keeps your title format consistent


# =========================
# HELPERS
# =========================

def extract_ipix(path: Path) -> int | None:
    """Extract ipix from directory names like hp_097065 or filename stems like hp097065_..."""
    for p in path.parents:
        m = re.match(r"hp[_-]?(\d+)", p.name)
        if m:
            return int(m.group(1))
    m = re.match(r"hp[_-]?(\d+)", path.stem)
    if m:
        return int(m.group(1))
    return None


def find_ebf_in_dir(hp_dir: Path) -> Path | None:
    cands = sorted(hp_dir.glob(EBF_GLOB))
    return cands[0] if cands else None


def safe_read(fn: Path, key: str) -> np.ndarray | None:
    try:
        return np.asarray(ebf.read(str(fn), key))
    except Exception:
        return None


def compute_abs_and_corrected_mags_from_ebf(
    ebf_file: Path,
    abs_mag_key: str = MAG_ABS_KEY,
    dist_kpc_key: str = DIST_KPC_KEY,
    ebv_schlegel_key: str = EBV_SCHLEGEL_KEY,
    ebv_solar_key: str = EBV_SOLAR_KEY,
    f_band: float = F_BAND,
    ebv_solar_scale: float = EBV_SOLAR_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Returns:
      M_abs: absolute mags (finite, matched)
      m_corr: corrected apparent mags (finite, matched)
      r_kpc: distances (kpc, matched)
      ebv_phys: physical E(B-V) used (matched)
      ebv_source: string
    """
    M = safe_read(ebf_file, abs_mag_key)
    r_kpc = safe_read(ebf_file, dist_kpc_key)
    if M is None or r_kpc is None:
        raise RuntimeError(f"Missing required keys: {abs_mag_key} or {dist_kpc_key}")

    ebv_s = safe_read(ebf_file, ebv_schlegel_key)
    ebv_sol = safe_read(ebf_file, ebv_solar_key)

    if ebv_s is not None:
        ebv = ebv_s
        ebv_source = "exbv_schlegel"
    elif ebv_sol is not None:
        ebv = ebv_solar_scale * ebv_sol
        ebv_source = f"{ebv_solar_scale}*exbv_solar"
    else:
        raise RuntimeError("No extinction column found (exbv_schlegel or exbv_solar).")

    ok = np.isfinite(M) & np.isfinite(r_kpc) & (r_kpc > 0) & np.isfinite(ebv)

    M_ok = M[ok]
    r_ok = r_kpc[ok]
    ebv_ok = ebv[ok]

    # DM = 5*log10(100*r_kpc) for r in kpc
    DM = 5.0 * np.log10(100.0 * r_ok)
    A = f_band * ebv_ok

    m_corr = M_ok + DM + A
    return M_ok, m_corr, r_ok, ebv_ok, ebv_source


def crowding_limit_from_mags(
    mags: np.ndarray,
    area_deg2: float,
    fwhm_arcsec: float,
    k_scale: float,
    pcrit: float,
) -> float | None:
    """
    Compute crowding limit magnitude where P(>=1 neighbor) >= pcrit using Poisson model.
    """
    mags = mags[np.isfinite(mags)]
    if mags.size == 0:
        return None

    mags_sorted = np.sort(mags)
    n = mags_sorted.size

    area_arcsec2 = area_deg2 * (3600.0 ** 2)
    r_arcsec = k_scale * fwhm_arcsec
    A = math.pi * (r_arcsec ** 2)

    Ncum = np.arange(1, n + 1, dtype=float)
    rho_cum = Ncum / area_arcsec2
    P = 1.0 - np.exp(-rho_cum * A)

    idx = np.where(P >= pcrit)[0]
    if idx.size == 0:
        return None
    return float(mags_sorted[int(idx[0])])


def _make_bins(x: np.ndarray, dx: float):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([0, 1])
    xmin = np.floor(x.min() / dx) * dx
    xmax = np.ceil(x.max() / dx) * dx
    return np.arange(xmin, xmax + dx, dx)


def plot_mag_2panel(mags: np.ndarray, outpng: Path, title: str, mlim: float | None, dx: float = DMAG):
    mags = mags[np.isfinite(mags)]
    if mags.size == 0:
        return

    bins = _make_bins(mags, dx)
    hist, edges = np.histogram(mags, bins=bins)
    cum = np.cumsum(hist)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

    ax1.bar(centers, hist, width=dx, align="center")
    ax1.set_yscale("log")
    ax1.set_xlabel("Magnitude")
    ax1.set_ylabel(r"$\mathrm{d}N/\mathrm{d}m$")
    ax1.set_title("Differential")
    ax1.grid(True, alpha=0.3)

    ax2.step(centers, cum, where="mid")
    ax2.set_xlabel("Magnitude")
    ax2.set_ylabel(r"$N(<m)$")
    ax2.set_title("Cumulative")
    ax2.grid(True, alpha=0.3)

    ax1.set_ylim(1, max(hist) * 1.2 if hist.size else 10)
    ax2.set_ylim(0, max(cum) * 1.05 if cum.size else 1)

    if mlim is not None:
        for ax in (ax1, ax2):
            ax.axvline(mlim, linestyle="--", linewidth=1.5)
        ax2.text(mlim, ax2.get_ylim()[1] * 0.95, f"m_lim={mlim:.2f}",
                 rotation=90, va="top", ha="right")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(outpng, dpi=220)
    plt.close(fig)

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
    l_min, l_max, b_min, b_max = box

    hp.cartview(
        healpix_map,
        coord="G",
        lonra=[l_min, l_max],
        latra=[b_min, b_max],
        title=title,
        unit=unit,
        min=min_val,
        max=max_val,
        badcolor="lightgray",
        notext=False,
        cbar=True,
        return_projected_map=False,
    )
    hp.graticule()

    if out_png:
        plt.savefig(out_png, dpi=250, bbox_inches="tight")
        plt.close()
        print(f"Wrote zoom plot: {out_png}")


def make_healpix_map_and_plot(
    *,
    nside: int,
    df: pd.DataFrame,
    value_col: str = "m_lim",
    title: str = "Crowding-limited depth",
    out_png: str | None = None,
    unit: str = "mag",
    min_val: float | None = None,
    max_val: float | None = None,
):
    npix = hp.nside2npix(nside)
    m = np.full(npix, hp.UNSEEN, dtype=float)

    for _, row in df.iterrows():
        ipix = int(row["ipix"])
        val = float(row[value_col])
        if np.isfinite(val):
            m[ipix] = val

    hp.mollview(
        m,
        title=title,
        unit=unit,
        min=min_val,
        max=max_val,
        badcolor="lightgray",
    )
    hp.graticule()

    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Wrote plot: {out_png}")

    return m

def plot_dist_ebv_2x2(r_kpc: np.ndarray, ebv: np.ndarray, outpng: Path, title: str,
                     dr: float = 0.2, debv: float = 0.05):
    r = r_kpc[np.isfinite(r_kpc)]
    e = ebv[np.isfinite(ebv)]
    if r.size == 0 or e.size == 0:
        return

    bins_r = _make_bins(r, dr)
    bins_e = _make_bins(e, debv)

    hr, er = np.histogram(r, bins=bins_r)
    cr = np.cumsum(hr)
    rc = 0.5 * (er[:-1] + er[1:])

    he, ee = np.histogram(e, bins=bins_e)
    ce = np.cumsum(he)
    ec = 0.5 * (ee[:-1] + ee[1:])

    fig, axs = plt.subplots(2, 2, figsize=(11, 7))

    axs[0, 0].bar(rc, hr, width=dr, align="center")
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_xlabel("Distance (kpc)")
    axs[0, 0].set_ylabel("dN")
    axs[0, 0].set_title("Distance distribution")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].step(rc, cr, where="mid")
    axs[0, 1].set_xlabel("Distance (kpc)")
    axs[0, 1].set_ylabel("N(<r)")
    axs[0, 1].set_title("Distance cumulative")
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].bar(ec, he, width=debv, align="center")
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_xlabel(r"$E(B-V)$")
    axs[1, 0].set_ylabel("dN")
    axs[1, 0].set_title("Reddening distribution")
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].step(ec, ce, where="mid")
    axs[1, 1].set_xlabel(r"$E(B-V)$")
    axs[1, 1].set_ylabel(r"$N(<E)$")
    axs[1, 1].set_title("Reddening cumulative")
    axs[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpng, dpi=220)
    plt.close(fig)


# =========================
# MAIN
# =========================

def main() -> None:
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"ROOT_DIR not found: {ROOT_DIR}")

    hp_dirs = sorted([p for p in ROOT_DIR.iterdir() if p.is_dir() and re.match(r"hp[_-]?\d+", p.name)])
    if not hp_dirs:
        raise RuntimeError(f"No hp_* directories found under {ROOT_DIR}")

    rows: list[dict] = []
    l_grid, b_grid, BOX = make_plane_grid_10x10(
        l_center=30.0,
        b_center=0.0,
        l_width=20.0,
        b_width=10.0,
        n_l=100,
        n_b=100,
    )
    print("Grid box (l_min, l_max, b_min, b_max) =", BOX)
    for hp_dir in hp_dirs:
        print("working on", hp_dir)
        ebf_file = find_ebf_in_dir(hp_dir)
        if ebf_file is None:
            continue

        ipix = extract_ipix(hp_dir) or extract_ipix(ebf_file)
        area_deg2 = AREA_DEG2_DEFAULT

        try:
            M_abs, m_corr, r_kpc, ebv_phys, ebv_source = compute_abs_and_corrected_mags_from_ebf(
                ebf_file=ebf_file,
                abs_mag_key=MAG_ABS_KEY,
                dist_kpc_key=DIST_KPC_KEY,
                f_band=F_BAND,
                ebv_solar_scale=EBV_SOLAR_SCALE,
            )
        except Exception as e:
            rows.append({
                "ipix": ipix,
                "ebf_path": str(ebf_file),
                "status": "fail",
                "reason": str(e),
            })
            continue

        # crowding limits in both flavors
        m_lim_abs = crowding_limit_from_mags(M_abs, area_deg2, FWHM_ARCSEC, K_SCALE, P_CRIT)
        m_lim_corr = crowding_limit_from_mags(m_corr, area_deg2, FWHM_ARCSEC, K_SCALE, P_CRIT)

        # keep legacy-style m_lim as corrected
        m_lim = m_lim_corr

        nstars = int(len(M_abs))
        density_deg2 = (nstars / area_deg2) if area_deg2 > 0 else np.nan
        density_arcsec2 = density_deg2 / (3600.0 ** 2)
        theta, phi = hp.pix2ang(NSIDE, ipix)  # theta=colatitude [0..pi], phi=lon [0..2pi]
        l_deg = float(np.degrees(phi))
        b_deg = float(90.0 - np.degrees(theta))

        row = {
            "ipix": ipix,
            "nside": NSIDE,
            "l_deg": l_deg,
            "b_deg": b_deg,
            "ebf_path": str(ebf_file),
            "status": "ok",
            "nstars": nstars,
            "area_deg2": area_deg2,
            "density_deg2": density_deg2,
            "density_arcsec2": density_arcsec2,

            "m_lim_abs": (np.nan if m_lim_abs is None else m_lim_abs),
            "m_lim_corr": (np.nan if m_lim_corr is None else m_lim_corr),
            "m_lim": (np.nan if m_lim is None else m_lim),

            "pcrit": P_CRIT,
            "fwhm_arcsec": FWHM_ARCSEC,
            "k_scale": K_SCALE,

            "mag_abs_key": MAG_ABS_KEY,
            "dist_key": DIST_KPC_KEY,
            "ebv_source": ebv_source,
            "f_band": F_BAND,
            "ebv_solar_scale": EBV_SOLAR_SCALE,
        }
        rows.append(row)

        if MAKE_PLOTS:
            stem = ebf_file.stem

            # 1) absolute mags
            out_abs = hp_dir / f"{stem}_abs_magdiag.png"
            title_abs = (
                f"{hp_dir.name}  ipix={ipix}  N={nstars:,}  area={area_deg2:.4f} deg$^2$\n"
                f"ABS mags from {MAG_ABS_KEY} | crowding: P≥{P_CRIT}, FWHM={FWHM_ARCSEC}\", k={K_SCALE}"
            )
            plot_mag_2panel(M_abs, out_abs, title_abs, m_lim_abs, dx=DMAG)

            # 2) corrected mags
            out_corr = hp_dir / f"{stem}_corr_magdiag.png"
            title_corr = (
                f"{hp_dir.name}  ipix={ipix}  N={nstars:,}  area={area_deg2:.4f} deg$^2$\n"
                f"CORR mags = {MAG_ABS_KEY} + DM + {F_BAND:.3f}*E(B-V) [{ebv_source}] | "
                f"crowding: P≥{P_CRIT}, FWHM={FWHM_ARCSEC}\", k={K_SCALE}"
            )
            plot_mag_2panel(m_corr, out_corr, title_corr, m_lim_corr, dx=DMAG)

            # 3) distance + reddening
            out_de = hp_dir / f"{stem}_dist_ebv.png"
            title_de = (
                f"{hp_dir.name}  ipix={ipix} | Distance + reddening diagnostics [{ebv_source}]"
            )
            plot_dist_ebv_2x2(r_kpc, ebv_phys, out_de, title_de)

    df = pd.DataFrame(rows)
    df.sort_values(by=["status", "ipix"], inplace=True, kind="mergesort")
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV}  (rows={len(df)})")

    # ---- HEALPix maps: ABS + CORR ----
    if "m_lim_abs" in df.columns:
        out_png_allsky_abs = str(Path(OUTDIR) / "m_lim_abs_healpix_mollview.png")
        m_abs = make_healpix_map_and_plot(
            nside=NSIDE,
            df=df,
            value_col="m_lim_abs",
            title=f"Crowding-limited depth (ABS mags, {MODE}, p={P_BLEND})  NSIDE={NSIDE}",
            out_png=out_png_allsky_abs,
            unit="mag",
        )

        out_png_zoom_abs = str(Path(OUTDIR) / "m_lim_abs_healpix_cartview_zoom.png")
        make_zoom_cartview_plot(
            healpix_map=m_abs,
            box=BOX,
            title=f"Zoom: Crowding-limited depth (ABS mags, {MODE}, p={P_BLEND})",
            out_png=out_png_zoom_abs,
            unit="mag",
        )

    if "m_lim_corr" in df.columns:
        out_png_allsky_corr = str(Path(OUTDIR) / "m_lim_corr_healpix_mollview.png")
        m_corr = make_healpix_map_and_plot(
            nside=NSIDE,
            df=df,
            value_col="m_lim_corr",
            title=f"Crowding-limited depth (CORR mags, {MODE}, p={P_BLEND})  NSIDE={NSIDE}",
            out_png=out_png_allsky_corr,
            unit="mag",
        )

        out_png_zoom_corr = str(Path(OUTDIR) / "m_lim_corr_healpix_cartview_zoom.png")
        make_zoom_cartview_plot(
            healpix_map=m_corr,
            box=BOX,
            title=f"Zoom: Crowding-limited depth (CORR mags, {MODE}, p={P_BLEND})",
            out_png=out_png_zoom_corr,
            unit="mag",
        )

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
recalc_crowding_from_ebf.py

Recompute crowding-limited magnitude per HEALPix directory from Galaxia EBF outputs,
using distance modulus + extinction correction (Galaxia website prescription).

Outputs:
- results_recalc.csv (one row per ipix)
- per-healpix diagnostic PNG histograms with vertical line at m_lim

Assumptions / intent:
- Galaxia stored band magnitude in mag0 (likely absolute magnitude in the selected band).
- Apparent magnitude is reconstructed via:
    m = M + 5*log10(100*r_kpc) + f_band * E(B-V)
- E(B-V) is taken as:
    exbv_schlegel (preferred)  OR  0.53 * exbv_solar
- Crowding limit defined by P(>=1 neighbor) >= Pcrit with Poisson model:
    P = 1 - exp(-rho(<m) * A)
"""

from __future__ import annotations

from pathlib import Path
import re
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # PNG only; avoids PyCharm backend issues
import matplotlib.pyplot as plt

import ebf


# =========================
# USER CONFIGURATION
# =========================

# Root directory containing hp_XXXXXX subdirectories
ROOT_DIR = Path(
    Path.home() / "Work/GaiaNIR/Galaxia/galaxia_runs/healpix_n128_plane_100x100_Ks_22012026"
)

# Output CSV path
OUT_CSV = ROOT_DIR / "results_recalc.csv"

# Which EBF file to use in each hp_* directory:
# - if multiple match, we pick the first one in sorted order
EBF_GLOB = "*.ebf"

# Magnitude reconstruction inputs
MAG_ABS_KEY = "/dcmc_ks"           # stored band magnitude (often absolute in Galaxia)
DIST_KPC_KEY = "/rad"           # heliocentric distance in kpc
EBV_SCHLEGEL_KEY = "/exbv_schlegel"
EBV_SOLAR_KEY = "/exbv_solar"

# Extinction coefficient f(i)=A(i)/E(B-V) for the band you are using.
# From the Galaxia webpage table: UKIRT K has A/E(B-V)=0.367
# (DCMC Ks is close enough for diagnostics; adjust if you prefer a different coefficient.)
F_BAND = 0.367

# Conversion for exbv_solar -> physical E(B-V) per Galaxia help text
EBV_SOLAR_SCALE = 0.53

# HEALPix pixel area for NSIDE=128 in deg^2 (you already used 0.2098234113)
AREA_DEG2_DEFAULT = 0.2098234113

# Crowding model parameters
P_CRIT = 0.10       # threshold for P(>=1 neighbor)
FWHM_ARCSEC = 0.2   # assumed PSF FWHM
K_SCALE = 1.0       # effective radius = K_SCALE * FWHM

# Histogram diagnostics
DMAG = 0.1          # bin width
MAKE_PLOTS = True
PLOT_FILENAME_SUFFIX = "_magdiag_recalc.png"


# =========================
# HELPERS
# =========================
def compute_corrected_mags_from_ebf(
    ebf_file: str,
    abs_mag_key: str = "/dcmc_ks",
    dist_kpc_key: str = "/rad",
    ebv_schlegel_key: str = "/exbv_schlegel",
    ebv_solar_key: str = "/exbv_solar",
    f_band: float = 0.367,          # A_band / E(B-V)  (UKIRT K from Galaxia table)
    ebv_solar_scale: float = 0.53,  # Galaxia help: convert exbv_solar to physical E(B-V)
):
    """
    Returns:
      M_abs : absolute magnitude array (finite only)
      m_app : corrected apparent magnitude array (finite only)
      r_kpc : distance array matched to returned mags
      ebv   : physical E(B-V) array matched to returned mags
      ebv_source : str describing which EBV was used
    """
    M = np.asarray(ebf.read(ebf_file, abs_mag_key))
    r_kpc = np.asarray(ebf.read(ebf_file, dist_kpc_key))

    # prefer physically calibrated value if present
    ebv_s = None
    ebv_sol = None
    try:
        ebv_s = np.asarray(ebf.read(ebf_file, ebv_schlegel_key))
    except Exception:
        ebv_s = None
    try:
        ebv_sol = np.asarray(ebf.read(ebf_file, ebv_solar_key))
    except Exception:
        ebv_sol = None

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

    # DM = 5*log10(d_pc/10), d_pc = 1000*r_kpc -> DM = 5*log10(100*r_kpc)
    DM = 5.0 * np.log10(100.0 * r_ok)
    A = f_band * ebv_ok

    m_app = M_ok + DM + A
    return M_ok, m_app, r_ok, ebv_ok, ebv_source


# ---------- Crowding limit (Poisson neighbor model) ----------
def crowding_limit_from_mags(
    mags: np.ndarray,
    area_deg2: float,
    fwhm_arcsec: float = 0.2,
    k_scale: float = 1.0,
    pcrit: float = 0.10,
):
    """
    Crowding criterion:
      P(>=1 neighbor) = 1 - exp(-rho(<m) * A) >= pcrit
    where rho(<m) is cumulative surface density and A = pi*(k*FWHM)^2

    Returns (m_lim or None, meta dict)
    """
    mags = mags[np.isfinite(mags)]
    if mags.size == 0:
        return None, {"reached": False, "reason": "no mags"}

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
        return None, {"reached": False, "p_max": float(P.max()), "A_arcsec2": A}

    i0 = int(idx[0])
    return float(mags_sorted[i0]), {"reached": True, "p_at_mlim": float(P[i0]), "A_arcsec2": A}


# ---------- Plotting helpers ----------
def _make_bins(x: np.ndarray, dx: float):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([0, 1])
    xmin = np.floor(x.min() / dx) * dx
    xmax = np.ceil(x.max() / dx) * dx
    return np.arange(xmin, xmax + dx, dx)


def plot_mag_2panel(mags: np.ndarray, outpng: str, title: str, mlim: float | None, dx: float = 0.1):
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


def plot_dist_ebv_2x2(r_kpc: np.ndarray, ebv: np.ndarray, outpng: str, title: str,
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
    if not cands:
        return None
    return cands[0]


def safe_read_ebf_array(fn: Path, key: str) -> np.ndarray | None:
    try:
        arr = np.asarray(ebf.read(str(fn), key))
        return arr
    except Exception:
        return None


def compute_apparent_mag(
    M: np.ndarray,
    r_kpc: np.ndarray,
    ebv_schlegel: np.ndarray | None,
    ebv_solar: np.ndarray | None,
    f_band: float,
    ebv_solar_scale: float,
) -> tuple[np.ndarray, str]:
    """
    Return apparent magnitudes m, plus a string describing which EBV source was used.
    m = M + 5*log10(100*r_kpc) + f_band * E(B-V)
    """
    if ebv_schlegel is not None:
        ebv = ebv_schlegel
        ebv_source = "exbv_schlegel"
    elif ebv_solar is not None:
        ebv = ebv_solar_scale * ebv_solar
        ebv_source = f"{ebv_solar_scale}*exbv_solar"
    else:
        raise RuntimeError("No E(B-V) column found (neither exbv_schlegel nor exbv_solar).")

    # mask valid
    ok = np.isfinite(M) & np.isfinite(r_kpc) & (r_kpc > 0) & np.isfinite(ebv)

    # distance modulus: DM = 5*log10(d_pc/10) with d_pc = 1000*r_kpc -> DM = 5*log10(100*r_kpc)
    DM = 5.0 * np.log10(100.0 * r_kpc[ok])
    A = f_band * ebv[ok]
    m = M[ok] + DM + A
    return m, ebv_source


def crowding_limit_from_mags(
    mags: np.ndarray,
    area_deg2: float,
    fwhm_arcsec: float,
    k_scale: float,
    pcrit: float,
) -> tuple[float | None, dict]:
    """
    Compute crowding limit magnitude where P(>=1 neighbor) >= pcrit using Poisson model.
    Uses cumulative density rho(<m) = N(<m) / area_arcsec2
    P = 1 - exp(-rho * A)
    """
    if mags.size == 0:
        return None, {"reason": "no mags"}

    mags = mags[np.isfinite(mags)]
    if mags.size == 0:
        return None, {"reason": "no finite mags"}

    mags_sorted = np.sort(mags)
    n = mags_sorted.size

    area_arcsec2 = area_deg2 * (3600.0 ** 2)
    r_arcsec = k_scale * fwhm_arcsec
    A = math.pi * (r_arcsec ** 2)

    Ncum = np.arange(1, n + 1, dtype=float)
    rho_cum = Ncum / area_arcsec2  # stars / arcsec^2
    P = 1.0 - np.exp(-rho_cum * A)

    idx = np.where(P >= pcrit)[0]
    if idx.size == 0:
        return None, {
            "area_arcsec2": area_arcsec2,
            "A_arcsec2": A,
            "r_arcsec": r_arcsec,
            "pcrit": pcrit,
            "reached": False,
            "p_max": float(P.max()),
        }

    i0 = int(idx[0])
    mlim = float(mags_sorted[i0])
    return mlim, {
        "area_arcsec2": area_arcsec2,
        "A_arcsec2": A,
        "r_arcsec": r_arcsec,
        "pcrit": pcrit,
        "reached": True,
        "p_at_mlim": float(P[i0]),
        "p_max": float(P.max()),
    }


def make_bins(mags: np.ndarray, dmag: float) -> np.ndarray:
    mmin = np.floor(mags.min() / dmag) * dmag
    mmax = np.ceil(mags.max() / dmag) * dmag
    return np.arange(mmin, mmax + dmag, dmag)


def plot_mag_diagnostics(outpng: Path, mags_app: np.ndarray, mlim: float | None, title: str) -> None:
    mags_app = mags_app[np.isfinite(mags_app)]
    if mags_app.size == 0:
        return

    bins = make_bins(mags_app, DMAG)
    hist, edges = np.histogram(mags_app, bins=bins)
    cum = np.cumsum(hist)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

    ax1.bar(centers, hist, width=DMAG, align="center")
    ax1.set_xlabel("Apparent magnitude")
    ax1.set_ylabel(r"$\mathrm{d}N/\mathrm{d}m$")
    ax1.set_title("Differential distribution")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")  # reveal faint tail

    ax2.step(centers, cum, where="mid")
    ax2.set_xlabel("Apparent magnitude")
    ax2.set_ylabel(r"$N(<m)$")
    ax2.set_title("Cumulative distribution")
    ax2.grid(True, alpha=0.3)

    # y limits for nice annotation
    ax1.set_ylim(1, max(hist) * 1.2 if hist.size else 10)
    ax2.set_ylim(0, max(cum) * 1.05 if cum.size else 1)

    if mlim is not None:
        for ax in (ax1, ax2):
            ax.axvline(mlim, linestyle="--", linewidth=1.5)
        ax2.text(
            mlim, ax2.get_ylim()[1] * 0.95,
            f"m_lim={mlim:.2f}",
            rotation=90, va="top", ha="right"
        )
    else:
        ax2.text(0.02, 0.95, "m_lim not reached", transform=ax2.transAxes, va="top")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
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

    for hp_dir in hp_dirs:
        print("working on ", hp_dir)
        ebf_file = find_ebf_in_dir(hp_dir)
        if ebf_file is None:
            continue

        ipix = extract_ipix(hp_dir) or extract_ipix(ebf_file)

        M = safe_read_ebf_array(ebf_file, MAG_ABS_KEY)
        r_kpc = safe_read_ebf_array(ebf_file, DIST_KPC_KEY)
        ebv_s = safe_read_ebf_array(ebf_file, EBV_SCHLEGEL_KEY)
        ebv_sol = safe_read_ebf_array(ebf_file, EBV_SOLAR_KEY)

        if M is None or r_kpc is None:
            rows.append({
                "ipix": ipix,
                "ebf": str(ebf_file),
                "status": "fail",
                "reason": "missing mag0 or rad",
            })
            continue

        try:
            m_app, ebv_source = compute_apparent_mag(
                M=M,
                r_kpc=r_kpc,
                ebv_schlegel=ebv_s,
                ebv_solar=ebv_sol,
                f_band=F_BAND,
                ebv_solar_scale=EBV_SOLAR_SCALE,
            )
        except Exception as e:
            rows.append({
                "ipix": ipix,
                "ebf": str(ebf_file),
                "status": "fail",
                "reason": f"EBV issue: {e}",
            })
            continue

        area_deg2 = AREA_DEG2_DEFAULT  # constant for NSIDE=128
        mlim, crowd_meta = crowding_limit_from_mags(
            mags=m_app,
            area_deg2=area_deg2,
            fwhm_arcsec=FWHM_ARCSEC,
            k_scale=K_SCALE,
            pcrit=P_CRIT,
        )

        nstars = int(np.isfinite(m_app).sum())
        mean_rho_deg2 = (nstars / area_deg2) if area_deg2 > 0 else np.nan
        mean_rho_arcsec2 = mean_rho_deg2 / (3600.0 ** 2)

        row = {
            "ipix": ipix,
            "ebf": str(ebf_file),
            "status": "ok",
            "nstars": nstars,
            "area_deg2": area_deg2,
            "mean_rho_deg2": mean_rho_deg2,
            "mean_rho_arcsec2": mean_rho_arcsec2,
            "m_lim": (np.nan if mlim is None else mlim),
            "pcrit": P_CRIT,
            "fwhm_arcsec": FWHM_ARCSEC,
            "k_scale": K_SCALE,
            "A_arcsec2": crowd_meta.get("A_arcsec2", np.nan),
            "r_eff_arcsec": crowd_meta.get("r_arcsec", np.nan),
            "p_max": crowd_meta.get("p_max", np.nan),
            "criterion_reached": crowd_meta.get("reached", False),
            "ebv_source": ebv_source,
            "f_band": F_BAND,
            "ebv_solar_scale": EBV_SOLAR_SCALE,
        }
        rows.append(row)

        # Per-healpix diagnostic plot
        if MAKE_PLOTS:
            outpng = hp_dir / f"{ebf_file.stem}{PLOT_FILENAME_SUFFIX}"
            title = (
                f"{hp_dir.name}  ipix={ipix}  N={nstars:,}  area={area_deg2:.4f} deg$^2$\n"
                f"m_app = mag0 + DM + {F_BAND:.3f}*E(B-V),  E(B-V) from {ebv_source}\n"
                f"crowding: P(>=1)≥{P_CRIT}, FWHM={FWHM_ARCSEC}\", k={K_SCALE}  "
                f"m_lim={'n/a' if mlim is None else f'{mlim:.2f}'}"
            )
            plot_mag_diagnostics(outpng, m_app, mlim, title)

    df = pd.DataFrame(rows)
    df.sort_values(by=["status", "ipix"], inplace=True, kind="mergesort")
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV}  (rows={len(df)})")


if __name__ == "__main__":
    main()

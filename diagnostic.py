#!/usr/bin/env python3
"""
Two-panel magnitude diagnostics for Galaxia outputs:
  (left)  differential magnitude distribution dN/dm
  (right) cumulative distribution N(<m)

Designed to run cleanly from PyCharm (no CLI args needed).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import ebf
import pandas as pd


# =========================
# USER CONFIGURATION
# =========================

# Path to Galaxia output EBF file
EBF_PATH = Path(
    Path.home() / "Work/GaiaNIR/Galaxia/galaxia_runs/healpix_n128_plane_100x100_Ks/hp_097065/hp097065_n128_Ks22.ebf"
)

# Magnitude key (choose one)
MAG_KEY = "/dcmc_ks"   # DCMC Ks
# MAG_KEY = "/ubv_v"   # UBV V

RESULTS_CSV = Path.home() / "Work/GaiaNIR/Galaxia/galaxia_runs/healpix_n128_plane_100x100_Ks/results_plane_10x10.csv"
IPIX = 97065  # <-- the pixel you are plotting (from hp_097065)


DMAG = 0.1             # bin width in magnitudes
OUTDIR = None          # None → same directory as EBF


# =========================
# IMPLEMENTATION
# =========================

def read_mlim_from_csv(csv_file: Path, ipix: int) -> float | None:
    df = pd.read_csv(csv_file)
    row = df.loc[df["ipix"] == ipix]
    if row.empty:
        return None
    m = row.iloc[0]["m_lim"]
    if np.isfinite(m):
        return float(m)
    return None

def load_magnitudes(ebf_file: Path, mag_key: str) -> np.ndarray:
    mags = np.asarray(ebf.read(str(ebf_file), mag_key))
    return mags[np.isfinite(mags)]


def make_bins(mags: np.ndarray, dmag: float) -> np.ndarray:
    mmin = np.floor(mags.min() / dmag) * dmag
    mmax = np.ceil(mags.max() / dmag) * dmag
    return np.arange(mmin, mmax + dmag, dmag)


def main() -> None:
    if not EBF_PATH.exists():
        raise FileNotFoundError(f"EBF file not found: {EBF_PATH}")

    outdir = OUTDIR if OUTDIR else EBF_PATH.parent
    outdir.mkdir(parents=True, exist_ok=True)

    mags = load_magnitudes(EBF_PATH, MAG_KEY)
    if mags.size == 0:
        raise RuntimeError(f"No finite magnitudes found for key {MAG_KEY}")

    bins = make_bins(mags, DMAG)
    hist, edges = np.histogram(mags, bins=bins)
    cum = np.cumsum(hist)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # -------------------------
    # TWO-PANEL FIGURE
    # -------------------------
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.5), sharex=True
    )

    # Left panel: differential LF
    ax1.bar(centers, hist, width=DMAG, align="center")
    ax1.set_xlabel("Magnitude")
    ax1.set_ylabel(r"$\mathrm{d}N/\mathrm{d}m$")
    ax1.set_title("Differential magnitude distribution")
    ax1.grid(True, alpha=0.3)

    # Right panel: cumulative counts
    ax2.step(centers, cum, where="mid")
    ax2.set_xlabel("Magnitude")
    ax2.set_ylabel(r"$N(<m)$")
    ax2.set_title("Cumulative magnitude distribution")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"{EBF_PATH.name} — {MAG_KEY}\n"
        f"N = {mags.size:,},  range = "
        f"[{mags.min():.2f}, {mags.max():.2f}]",
        fontsize=11,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    band = MAG_KEY.strip("/")
    outpng = outdir / f"{EBF_PATH.stem}_{band}_mag_diagnostics.png"
    fig.savefig(outpng, dpi=220)

    print(f"Wrote: {outpng}")
    #plt.show()   # nice when running interactively in PyCharm


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import numpy as np

import matplotlib
matplotlib.use("Agg")  # safe for PyCharm / batch use
import matplotlib.pyplot as plt

import ebf
import pandas as pd


# =========================
# USER CONFIGURATION
# =========================
EBF_PATH = Path(
    Path.home() / "Work/GaiaNIR/Galaxia/galaxia_runs/healpix_n128_plane_100x100_Ks/"
    "hp_101172/hp101172_n128_Ks22.ebf"
)

MAG_KEY = "/dcmc_ks"
DMAG = 0.1

RESULTS_CSV = Path(
    Path.home() / "Work/GaiaNIR/Galaxia/galaxia_runs/"
    "healpix_n128_plane_100x100_Ks/results_plane_100x100.csv"
)


# =========================
# IMPLEMENTATION
# =========================
def extract_ipix(path: Path) -> int | None:
    """
    Extract HEALPix index from:
      - directory name: hp_097065
      - filename stem:  hp097065_...
    """
    # 1) check parent directories
    for p in path.parents:
        m = re.match(r"hp[_-]?(\d+)", p.name)
        if m:
            return int(m.group(1))

    # 2) check filename stem
    m = re.match(r"hp[_-]?(\d+)", path.stem)
    if m:
        return int(m.group(1))

    return None


def load_magnitudes(ebf_file: Path, mag_key: str) -> np.ndarray:
    mags = np.asarray(ebf.read(str(ebf_file), mag_key))
    return mags[np.isfinite(mags)]


def make_bins(mags: np.ndarray, dmag: float) -> np.ndarray:
    mmin = np.floor(mags.min() / dmag) * dmag
    mmax = np.ceil(mags.max() / dmag) * dmag
    return np.arange(mmin, mmax + dmag, dmag)


def read_mlim_from_csv(csv_file: Path, ipix: int) -> float | None:
    if not csv_file.exists():
        return None
    df = pd.read_csv(csv_file)
    row = df.loc[df["ipix"] == ipix]
    if row.empty:
        return None
    m = row.iloc[0].get("m_lim", np.nan)
    return float(m) if np.isfinite(m) else None


def add_mlim_vline(ax, mlim: float, label: str) -> None:
    ax.axvline(mlim, linestyle="--", linewidth=1.5)
    ymax = ax.get_ylim()[1]
    ax.text(
        mlim,
        ymax * 0.95,
        f"{label} = {mlim:.2f}",
        rotation=90,
        va="top",
        ha="right",
    )


def main() -> None:
    if not EBF_PATH.exists():
        raise FileNotFoundError(f"EBF file not found: {EBF_PATH}")

    ipix = extract_ipix(EBF_PATH)
    if ipix is None:
        raise RuntimeError(f"Could not extract ipix from path: {EBF_PATH}")

    mags = load_magnitudes(EBF_PATH, MAG_KEY)
    if mags.size == 0:
        raise RuntimeError(f"No finite magnitudes found for key {MAG_KEY}")

    bins = make_bins(mags, DMAG)
    hist, edges = np.histogram(mags, bins=bins)
    cum = np.cumsum(hist)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mlim = read_mlim_from_csv(RESULTS_CSV, ipix)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

    # Left panel
    ax1.bar(centers, hist, width=DMAG, align="center")
    ax1.set_xlabel("Magnitude")
    ax1.set_ylabel(r"$\mathrm{d}N/\mathrm{d}m$")
    ax1.set_title("Differential magnitude distribution")
    ax1.grid(True, alpha=0.3)

    # Right panel
    ax2.step(centers, cum, where="mid")
    ax2.set_xlabel("Magnitude")
    ax2.set_ylabel(r"$N(<m)$")
    ax2.set_title("Cumulative magnitude distribution")
    ax2.grid(True, alpha=0.3)

    # y-limits first (for annotation placement)
    ax1.set_ylim(0, max(hist) * 1.05 if hist.size else 1)
    ax2.set_ylim(0, max(cum) * 1.05 if cum.size else 1)

    if mlim is not None:
        label = r"$K_{s,\rm lim}$" if "ks" in MAG_KEY.lower() else r"$m_{\rm lim}$"
        add_mlim_vline(ax1, mlim, label)
        add_mlim_vline(ax2, mlim, label)
    else:
        ax1.text(0.02, 0.98, "m_lim: not available", transform=ax1.transAxes, va="top")
        ax2.text(0.02, 0.98, "m_lim: not available", transform=ax2.transAxes, va="top")

    fig.suptitle(
        f"{EBF_PATH.name} — {MAG_KEY}\n"
        f"ipix={ipix}, N={mags.size:,}, "
        f"range=[{mags.min():.2f}, {mags.max():.2f}]",
        fontsize=11,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    outpng = EBF_PATH.parent / (
        f"{EBF_PATH.stem}_{MAG_KEY.strip('/')}_mag_diagnostics_mlim.png"
    )
    fig.savefig(outpng, dpi=220)
    print(f"Wrote: {outpng}")

    plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import numpy as np

import matplotlib
matplotlib.use("Agg")  # safe in PyCharm / batch use
import matplotlib.pyplot as plt

import ebf
import pandas as pd

# FITS export for TOPCAT
from astropy.table import Table


# =========================
# USER CONFIGURATION
# =========================
EBF_PATH = Path(
    "/home/balog/Work/GaiaNIR/Galaxia/galaxia_runs/healpix_n128_plane_100x100_Ks/"
    "hp_097065/hp097065_n128_Ks22.ebf"
)

MAG_KEY = "/dcmc_ks"
DMAG = 0.1

RESULTS_CSV = Path(
    "/home/balog/Work/GaiaNIR/Galaxia/galaxia_runs/"
    "healpix_n128_plane_100x100_Ks/results_plane_100x100.csv"
)

# FITS export toggle (TOPCAT)
EXPORT_FITS = True

# Keep FITS small: export only a subset of columns you care about
# Adjust to your needs. (Names must match EBF top-level keys.)
EXPORT_COLUMNS = [
    "px", "py", "pz",          # positions (Galactocentric Cartesian in Galaxia output)
    "vx", "vy", "vz",
    "rad",                     # distance-related field in Galaxia output (check meaning in your setup)
    "dcmc_j", "dcmc_h", "dcmc_ks",
    "teff", "grav", "feh", "age", "smass",
    "popid", "satid", "fieldid", "partid",
]


# =========================
# IMPLEMENTATION
# =========================
def extract_ipix(path: Path) -> int | None:
    """
    Extract HEALPix index from:
      - directory name: hp_097065
      - filename stem:  hp097065_...
    """
    for p in path.parents:
        m = re.match(r"hp[_-]?(\d+)", p.name)
        if m:
            return int(m.group(1))

    m = re.match(r"hp[_-]?(\d+)", path.stem)
    if m:
        return int(m.group(1))

    return None


def ebf_keys(ebf_file: Path) -> list[str]:
    root = ebf.read(str(ebf_file), "/")
    return sorted(list(root.keys()))


def load_magnitudes(ebf_file: Path, mag_key: str) -> np.ndarray:
    mags = np.asarray(ebf.read(str(ebf_file), mag_key))
    return mags[np.isfinite(mags)]


def make_bins(mags: np.ndarray, dmag: float) -> np.ndarray:
    mmin = np.floor(mags.min() / dmag) * dmag
    mmax = np.ceil(mags.max() / dmag) * dmag
    return np.arange(mmin, mmax + dmag, dmag)


def read_mlim_from_csv(csv_file: Path, ipix: int) -> float | None:
    if csv_file is None or (not csv_file.exists()):
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


def export_to_fits(ebf_file: Path, out_fits: Path, columns: list[str]) -> Path:
    """
    Export selected EBF top-level keys to FITS (for TOPCAT).
    """
    avail = set(ebf_keys(ebf_file))
    cols = [c for c in columns if c in avail]

    if not cols:
        raise RuntimeError("None of the requested EXPORT_COLUMNS exist in this EBF file.")

    data = {}
    for c in cols:
        arr = np.asarray(ebf.read(str(ebf_file), f"/{c}"))
        data[c] = arr

    t = Table(data)
    t.write(out_fits, overwrite=True)
    print(f"Wrote FITS for TOPCAT: {out_fits}")
    return out_fits


def main() -> None:
    if not EBF_PATH.exists():
        raise FileNotFoundError(f"EBF file not found: {EBF_PATH}")

    # auto ipix
    ipix = extract_ipix(EBF_PATH)
    if ipix is None:
        print(f"WARNING: Could not extract ipix from path: {EBF_PATH}")
    else:
        print(f"Detected ipix = {ipix}")

    # Optional FITS export
    if EXPORT_FITS:
        out_fits = EBF_PATH.with_suffix(".fits")
        export_to_fits(EBF_PATH, out_fits, EXPORT_COLUMNS)

    # Magnitude diagnostics
    mags = load_magnitudes(EBF_PATH, MAG_KEY)
    if mags.size == 0:
        raise RuntimeError(f"No finite magnitudes found for key {MAG_KEY}")

    bins = make_bins(mags, DMAG)
    hist, edges = np.histogram(mags, bins=bins)
    cum = np.cumsum(hist)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mlim = None
    if ipix is not None:
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

    # Vertical line for crowding limit
    if mlim is not None:
        label = r"$K_{s,\rm lim}$" if "ks" in MAG_KEY.lower() else r"$m_{\rm lim}$"
        add_mlim_vline(ax1, mlim, label)
        add_mlim_vline(ax2, mlim, label)
    else:
        ax1.text(0.02, 0.98, "m_lim: not available", transform=ax1.transAxes, va="top")
        ax2.text(0.02, 0.98, "m_lim: not available", transform=ax2.transAxes, va="top")

    info_ipix = f"ipix={ipix}, " if ipix is not None else ""
    fig.suptitle(
        f"{EBF_PATH.name} — {MAG_KEY}\n"
        f"{info_ipix}N={mags.size:,}, range=[{mags.min():.2f}, {mags.max():.2f}]",
        fontsize=11,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    outpng = EBF_PATH.parent / f"{EBF_PATH.stem}_{MAG_KEY.strip('/')}_mag_diagnostics_mlim.png"
    fig.savefig(outpng, dpi=220)
    print(f"Wrote PNG: {outpng}")

    plt.close(fig)


if __name__ == "__main__":
    main()

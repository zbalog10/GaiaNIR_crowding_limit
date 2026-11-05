#!/usr/bin/env python3
"""
Single-field diagnostics for a Galaxia EBF output (one HEALPix pixel or test field).

Replaces diagnostic.py, diagnostic_withCLD.py, diagnostic_withCLD_fitsconversion.py
and diagnostic_withCLD_fitsconversion_allcolumns.py.

Produces <ebf stem>_diagnostics.png next to the EBF file:
  (left)  differential magnitude distribution dN/dm of apparent Ks
          (with extinction; the extinction-free distribution is overplotted)
  (right) cumulative counts N(<m) and the blending probability P(>=1 neighbour),
          with the crowding limit m_lim marked

m_lim is recomputed from the EBF with the same function the pipeline uses
(fullRun_galaxia_crowding.crowding_limited_depth), so plot and limit always use
the same magnitudes. If a results CSV is given and contains this pixel, its m_lim
is marked too, for comparison.

Optionally writes <ebf stem>.fits for TOPCAT (FITS_EXPORT = "selected" or "all"),
including a computed column ks_app (apparent Ks with extinction).

Runs from PyCharm with the settings below, or from the command line:
    python diagnostic_pixel.py [--ebf FILE] [--csv FILE] [--fits none|selected|all]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # write PNGs only; safe in PyCharm and batch use
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import ebf

from fullRun_galaxia_crowding import EXT_COEFF_KS, crowding_limited_depth, load_apparent_mags


# =========================
# USER CONFIGURATION
# =========================
EBF_PATH = Path.home() / "Work/GaiaNIR/galaxia_runs/bulge_test/l0_b5_Ks25_area0p01.ebf"

# Optional results table with an m_lim per ipix (e.g. results_plane_100x100.csv
# from raunGalxiaHealpix10x10.py). None -> only the recomputed m_lim is shown.
RESULTS_CSV: Path | None = None

# Magnitudes: /mag0 = apparent Ks without extinction; A_Ks = EXT_COEFF * E(B-V) is added.
MAG_KEY = "/mag0"
EXT_COEFF = EXT_COEFF_KS          # set to None to ignore extinction
EBV_KEY = "/exbv_schlegel"

# Crowding model (same defaults as the pipeline)
FWHM_ARCSEC = 0.2
K_SCALE = 1.0
P_BLEND = 0.10

DMAG = 0.1                        # histogram bin width (mag)
LOG_Y = True                      # log y-axis for dN/dm

# FITS export for TOPCAT: "none", "selected" or "all"
FITS_EXPORT = "none"
FITS_COLUMNS = [
    "glon", "glat", "rad",
    "px", "py", "pz", "vx", "vy", "vz",
    "mag0", "mag1", "mag2",                    # apparent, no extinction
    "dcmc_j", "dcmc_h", "dcmc_ks",             # absolute
    "exbv_schlegel", "exbv_solar",
    "teff", "grav", "feh", "age", "smass", "mact", "lum",
    "popid", "satid", "fieldid", "partid",
]


# =========================
# HELPERS
# =========================
def extract_ipix(path: Path) -> int | None:
    """HEALPix index from a directory like hp_097065 or a filename like hp097065_..."""
    for name in [path.stem] + [p.name for p in path.parents]:
        m = re.match(r"hp[_-]?(\d+)", name)
        if m:
            return int(m.group(1))
    return None


def survey_area_deg2(ebf_file: Path) -> float:
    """
    Survey area of the run in deg^2.

    Read from the <stem>.params file Galaxia was run with; falls back to the
    HEALPix pixel area if the filename contains _n<NSIDE>_.
    """
    params = ebf_file.with_suffix(".params")
    if params.exists():
        for line in params.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "surveyArea":
                return float(parts[1])

    m = re.search(r"_n(\d+)_", ebf_file.name)
    if m:
        import healpy as hp
        return float(hp.nside2pixarea(int(m.group(1)), degrees=True))

    raise RuntimeError(
        f"Cannot determine survey area: no surveyArea in {params} and no _n<NSIDE>_ in the filename."
    )


def read_mlim_from_csv(csv_file: Path | None, ipix: int | None) -> tuple[float | None, str | None]:
    """Return (m_lim, mag_key used for it) for this ipix from a results CSV, if available."""
    if csv_file is None or ipix is None or not Path(csv_file).exists():
        return None, None
    df = pd.read_csv(csv_file)
    row = df.loc[df["ipix"] == ipix]
    if row.empty or "m_lim" not in row:
        return None, None
    m = float(row.iloc[0]["m_lim"])
    key = str(row.iloc[0]["mag_key"]) if "mag_key" in row else None
    return (m if np.isfinite(m) else None), key


def export_fits(ebf_file: Path, ks_app: np.ndarray, mode: str) -> Path:
    """Write <stem>.fits with the chosen EBF columns plus ks_app (for TOPCAT)."""
    from astropy.table import Table

    root = ebf.read(str(ebf_file), "/")
    n = len(ks_app)
    wanted = list(root.keys()) if mode == "all" else FITS_COLUMNS

    data = {}
    for k in wanted:
        if k not in root:
            continue
        arr = np.asarray(root[k])
        if arr.ndim == 1 and arr.size == n:   # skips 'log' and the 6-element 'center'
            data[k] = arr
    data["ks_app"] = ks_app

    out = ebf_file.with_suffix(".fits")
    Table(data).write(out, overwrite=True)
    print(f"Wrote FITS ({len(data)} columns, {n:,} rows): {out}")
    return out


def mark_mlim(ax, m: float, label: str, style: str) -> None:
    """Vertical line with a rotated label near the top of the axes."""
    ax.axvline(m, linestyle=style, linewidth=1.5, color="k")
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(m, 0.97, f"{label} = {m:.2f}", rotation=90, va="top", ha="right",
            transform=trans, fontsize=9)


# =========================
# MAIN
# =========================
def run(ebf_file: Path, results_csv: Path | None, fits_mode: str) -> Path:
    ebf_file = Path(ebf_file)
    if not ebf_file.exists():
        raise FileNotFoundError(f"EBF file not found: {ebf_file}")

    ipix = extract_ipix(ebf_file)
    area = survey_area_deg2(ebf_file)

    # Magnitudes (with and without extinction)
    m_app_all = load_apparent_mags(ebf_file, MAG_KEY, ext_coeff=EXT_COEFF, ebv_key=EBV_KEY)
    m_noext = load_apparent_mags(ebf_file, MAG_KEY, ext_coeff=None)
    ok = np.isfinite(m_app_all)
    m_app, m_noext = m_app_all[ok], m_noext[ok]
    if m_app.size == 0:
        raise RuntimeError(f"No finite magnitudes for {MAG_KEY} in {ebf_file}")

    # Crowding limit, recomputed with the pipeline's function
    res = crowding_limited_depth(
        ebf_file, MAG_KEY,
        survey_area_deg2=area, fwhm_arcsec=FWHM_ARCSEC, k=K_SCALE,
        mode="poisson", p_blend=P_BLEND, dmag=DMAG,
        ext_coeff=EXT_COEFF, ebv_key=EBV_KEY,
    )
    m_lim = res["m_lim"]
    m_lim_csv, csv_key = read_mlim_from_csv(results_csv, ipix)

    # Histograms on common bins
    lo = np.floor(min(m_app.min(), m_noext.min()) / DMAG) * DMAG
    hi = np.ceil(max(m_app.max(), m_noext.max()) / DMAG) * DMAG
    bins = np.arange(lo, hi + DMAG, DMAG)
    h_app, edges = np.histogram(m_app, bins=bins)
    h_noext, _ = np.histogram(m_noext, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    ext_label = f"+ {EXT_COEFF}·E(B−V)" if EXT_COEFF is not None else "no extinction"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, layout="constrained")

    # Left: dN/dm
    ax1.bar(centers, h_app, width=DMAG, align="center", alpha=0.6,
            label=f"apparent Ks ({ext_label})")
    if EXT_COEFF is not None:
        ax1.step(centers, h_noext, where="mid", linewidth=1.0, color="0.3",
                 label="apparent Ks, no extinction")
    if LOG_Y:
        ax1.set_yscale("log")
    ax1.set_xlabel(r"$K_s$ (apparent)")
    ax1.set_ylabel(r"$\mathrm{d}N/\mathrm{d}m$ per %.2g mag" % DMAG)
    ax1.set_title("Differential magnitude distribution")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: N(<m) and P(>=1 neighbour)
    ax2.step(centers, np.cumsum(h_app), where="mid", label=r"$N(<m)$")
    ax2.set_xlabel(r"$K_s$ (apparent)")
    ax2.set_ylabel(r"$N(<m)$")
    ax2.set_title("Cumulative counts and blending probability")
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.plot(res["m_centers"], res["p_ge1"], color="tab:red", linewidth=1.2,
              label=r"$P(\geq1$ neighbour$)$")
    ax2b.axhline(P_BLEND, color="tab:red", linestyle=":", linewidth=1)
    ax2b.set_ylim(0, 1)
    ax2b.set_ylabel(r"$P(\geq 1$ neighbour within $r)$", color="tab:red")
    ax2b.tick_params(axis="y", colors="tab:red")

    # Crowding-limit markers
    for ax in (ax1, ax2):
        if m_lim is not None:
            mark_mlim(ax, m_lim, r"$K_{s,\rm lim}$", "--")
        # Mark the CSV value only if it differs from the recomputed one (avoids overlapping labels)
        if m_lim_csv is not None and (m_lim is None or abs(m_lim_csv - m_lim) > 0.005):
            mark_mlim(ax, m_lim_csv, "CSV", ":")
    if m_lim is None:
        ax2.text(0.02, 0.97, f"P never reaches {P_BLEND}: m_lim not reached",
                 transform=ax2.transAxes, va="top", fontsize=9)
    if m_lim_csv is not None and csv_key is not None and not csv_key.startswith("/mag"):
        print(f"WARNING: CSV m_lim was computed from {csv_key} (absolute magnitudes); "
              "it is not comparable with the apparent-magnitude axis.")

    dens = m_app.size / area
    fig.suptitle(
        f"{ebf_file.name}" + (f"  (ipix={ipix})" if ipix is not None else "") + "\n"
        f"N={m_app.size:,}, area={area:g} deg², density={dens:.3g}/deg² "
        f"({dens / 3600**2:.3g}/arcsec²); FWHM={FWHM_ARCSEC}\", k={K_SCALE}, "
        f"P_blend={P_BLEND} → m_lim="
        + (f"{m_lim:.2f}" if m_lim is not None else "not reached"),
        fontsize=10,
    )

    out_png = ebf_file.with_name(f"{ebf_file.stem}_diagnostics.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Wrote PNG: {out_png}")
    print(f"m_lim = {m_lim if m_lim is None else round(m_lim, 2)}"
          + (f"   (CSV: {m_lim_csv:.2f})" if m_lim_csv is not None else ""))

    if fits_mode != "none":
        export_fits(ebf_file, m_app_all, fits_mode)

    return out_png


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ebf", type=Path, default=EBF_PATH, help="Galaxia .ebf file")
    ap.add_argument("--csv", type=Path, default=RESULTS_CSV, help="results CSV with ipix,m_lim")
    ap.add_argument("--fits", choices=["none", "selected", "all"], default=FITS_EXPORT,
                    help="FITS export for TOPCAT")
    args = ap.parse_args()
    run(args.ebf, args.csv, args.fits)


if __name__ == "__main__":
    main()

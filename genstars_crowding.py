#!/usr/bin/env python3
"""
genstars wrapper + crowding-limited depth (bulge region).

genstars (Koshimoto 2022, https://github.com/nkoshimoto/genstars) generates stars with
the Koshimoto, Baba & Bennett (2021) Galactic model + nuclear stellar disk, inside a
rectangular (l, b) box. Differences from the Galaxia pipeline that matter here:

- Sky coverage: only -9.5 < l < 9.5 and -10.0 < b < 4.5 (VVV extinction map).
- Output: whitespace-separated text on stdout ('#' lines are headers/footers).
- Magnitudes are APPARENT WITH EXTINCTION (mag = M + DM + A_lambda, VVV E(J-Ks) map
  + Nishiyama+09 law); the extinction in the reference band is the A<band> column.
- Subsampling: only fSIMU x (model star count) stars are written, so the true
  surface density is N / (fSIMU * area). Each 0.025 x 0.025 deg extinction cell gets
  round(area_cell * rho * fSIMU) stars, so keep fSIMU high enough for >> 1 star/cell.
- Area: genstars uses the flat-sky box area dl * db (no cos b); we use the same so
  that N / (fSIMU * dl * db) reproduces the model density.
- With Magrange set, only stars (no remnants) inside the range of the reference band
  (iMag) are generated.

Usage:
    .venv/bin/python genstars_crowding.py          # runs the example field below
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fullRun_galaxia_crowding import crowding_limited_depth_from_mags


GENSTARS_DIR = Path.home() / "PycharmProjects/software/genstars"   # must contain input_files/
GENSTARS_EXE = GENSTARS_DIR / "genstars"

L_RANGE_ALLOWED = (-9.5, 9.5)
B_RANGE_ALLOWED = (-10.0, 4.5)
EXT_CELL_DEG = 0.025          # genstars extinction-grid cell size

BANDS = {0: ["V", "I", "J", "H", "Ks"],                    # ROMAN 0 (PRIME mode)
         1: ["J", "H", "Ks", "Z087", "W146", "F213"]}      # ROMAN 1 (Roman mode)
DEFAULT_IMAG = {0: 3, 1: 4}   # genstars defaults: H (PRIME), W146 (Roman)


# -----------------------------
# 1) Running genstars
# -----------------------------
def box_area_deg2(l_range: tuple[float, float], b_range: tuple[float, float]) -> float:
    """Flat-sky box area dl*db in deg^2 (the convention genstars uses)."""
    return float((l_range[1] - l_range[0]) * (b_range[1] - b_range[0]))


def run_genstars(
    out_file: str | Path,
    *,
    l_range: tuple[float, float],
    b_range: tuple[float, float],
    i_mag: int = 4,                                   # PRIME mode: 4 = Ks
    mag_range: tuple[float, float] | None = (-10.0, 30.0),
    f_simu: float = 0.01,
    seed: int = 12,
    roman: int = 0,
    verbosity: int = 1,
    extra_args: dict[str, object] | None = None,
    genstars_exe: str | Path = GENSTARS_EXE,
    genstars_dir: str | Path = GENSTARS_DIR,
    skip_existing: bool = True,
) -> Path:
    """
    Run genstars for an (l, b) box and write its stdout to out_file.

    Also writes <out_file stem>.json with the exact command and run settings
    (the equivalent of Galaxia's .params file), which read_genstars() uses.
    mag_range=None omits Magrange, so remnants (mag=99) are generated too.
    """
    out_file = Path(out_file)
    meta_file = out_file.with_suffix(".json")
    if skip_existing and out_file.exists() and meta_file.exists():
        print(f"Reusing existing genstars output: {out_file}")
        return out_file

    (l1, l2), (b1, b2) = l_range, b_range
    if not (l1 < l2 and b1 < b2):
        raise ValueError(f"Need l1 < l2 and b1 < b2, got l={l_range}, b={b_range}")
    if not (L_RANGE_ALLOWED[0] < l1 and l2 < L_RANGE_ALLOWED[1]
            and B_RANGE_ALLOWED[0] < b1 and b2 < B_RANGE_ALLOWED[1]):
        raise ValueError(
            f"Box l={l_range}, b={b_range} is outside the genstars extinction map "
            f"({L_RANGE_ALLOWED[0]} < l < {L_RANGE_ALLOWED[1]}, "
            f"{B_RANGE_ALLOWED[0]} < b < {B_RANGE_ALLOWED[1]})"
        )

    args: dict[str, object] = {
        "l": f"{l1:.6f} {l2:.6f}",
        "b": f"{b1:.6f} {b2:.6f}",
        "ROMAN": roman,
        "iMag": i_mag,
        "fSIMU": f"{f_simu:.10g}",
        "VERBOSITY": verbosity,
        "seed": seed,
    }
    if mag_range is not None:
        args["Magrange"] = f"{mag_range[0]:g} {mag_range[1]:g}"
    if extra_args:
        args.update(extra_args)
    cmd = [str(genstars_exe)] + [tok for k, v in args.items() for tok in (k, *str(v).split())]

    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running: {shlex.join(cmd)}\n  (cwd={genstars_dir}) -> {out_file}")
    t0 = time.time()
    with open(out_file, "w") as fh:
        proc = subprocess.run(cmd, cwd=str(genstars_dir), stdout=fh,
                              stderr=subprocess.PIPE, text=True)
    elapsed = time.time() - t0

    tail = _tail_lines(out_file, 5)
    if proc.returncode != 0 or not any(l.startswith("# (n_BD") for l in tail):
        raise RuntimeError(
            f"genstars failed (exit {proc.returncode}).\nstderr:\n{proc.stderr}\n"
            "last lines of output:\n" + "\n".join(tail)
        )

    meta = {
        "command": shlex.join(cmd),
        "genstars_dir": str(genstars_dir),
        "l_range": [l1, l2], "b_range": [b1, b2],
        "area_deg2": box_area_deg2(l_range, b_range),
        "f_simu": f_simu, "i_mag": i_mag, "roman": roman, "verbosity": verbosity,
        "mag_range": list(mag_range) if mag_range is not None else None,
        "seed": seed, "elapsed_s": round(elapsed, 1),
    }
    meta_file.write_text(json.dumps(meta, indent=2))
    print(f"  done in {elapsed:.0f} s")
    return out_file


def _tail_lines(path: Path, n: int) -> list[str]:
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - 8192))
        return fh.read().decode(errors="replace").splitlines()[-n:]


# -----------------------------
# 2) Reading genstars output
# -----------------------------
def genstars_columns(roman: int, i_mag: int, verbosity: int) -> list[str]:
    """Column names of genstars per-star output (BINARY 0, HWBAND 0), from genstars.c."""
    bands = BANDS[roman]
    cols = list(bands)
    if verbosity == 3:
        cols += [f"A{b}" for b in bands]
        cols += ["Mass", "Radius", "Dist", "mu_l", "mu_b", "l", "b", "cls", "fREM"]
    elif verbosity in (1, 2):
        cols += ["Mass", "Radius", "Dist", "mu_l", "mu_b", f"A{bands[i_mag]}", "l", "b", "cls", "fREM"]
    else:
        raise ValueError("VERBOSITY 0 writes no per-star rows")
    if verbosity >= 2:
        cols += ["InitialMass", "v_x", "v_y", "v_z"]
    return cols


def read_genstars(out_file: str | Path) -> tuple[pd.DataFrame, dict]:
    """
    Read a genstars output file written by run_genstars().

    Returns (stars, meta): stars has one row per generated star with the columns of
    genstars_columns(); meta is the run's .json plus the footer star counts.
    Magnitudes of 99 (remnants / undefined bands) are converted to NaN.
    """
    out_file = Path(out_file)
    meta = json.loads(out_file.with_suffix(".json").read_text())
    cols = genstars_columns(meta["roman"], meta["i_mag"], meta["verbosity"])

    stars = pd.read_csv(out_file, sep=r"\s+", comment="#", header=None,
                        names=cols, index_col=False, engine="c")
    if stars.shape[1] != len(cols) or stars[cols[-1]].isna().any():
        raise RuntimeError(f"Unexpected column layout in {out_file}; expected {cols}")
    for b in BANDS[meta["roman"]]:
        stars.loc[stars[b] >= 99, b] = np.nan

    # footer summary: "# (n_BD n_MS n_WD n_NS n_BH)/n_all= ( ... ) / N = ( ... )"
    for line in _tail_lines(out_file, 5):
        m = re.match(r"# \(n_BD n_MS n_WD n_NS n_BH\)/n_all= \(([\d\s]+)\) / *(\d+)", line)
        if m:
            meta["n_BD_MS_WD_NS_BH"] = [int(x) for x in m.group(1).split()]
            meta["n_all_footer"] = int(m.group(2))
    return stars, meta


# -----------------------------
# 3) Crowding limit
# -----------------------------
def genstars_crowding(
    out_file: str | Path,
    *,
    band: str = "Ks",
    fwhm_arcsec: float = 0.2,
    k: float = 1.0,
    p_blend: float = 0.10,
    dmag: float = 0.1,
    min_stars_per_cell: float = 20.0,
) -> dict:
    """
    Crowding-limited depth from a genstars output file (same model as Galaxia pipeline).

    Uses the effective area fSIMU * dl * db, so N / area is the model surface density.
    Warns if fSIMU leaves too few stars per 0.025-deg extinction cell (rounding bias).
    """
    stars, meta = read_genstars(out_file)
    area = meta["area_deg2"]
    f_simu = meta["f_simu"]
    mags = stars[band].to_numpy()

    n_cells = max(1.0, area / EXT_CELL_DEG**2)
    per_cell = np.isfinite(mags).sum() / n_cells
    if per_cell < min_stars_per_cell:
        print(f"WARNING: only {per_cell:.1f} stars per {EXT_CELL_DEG} deg cell; genstars rounds "
              "the count per cell, so densities may be biased. Increase fSIMU.")

    res = crowding_limited_depth_from_mags(
        mags, survey_area_deg2=f_simu * area, fwhm_arcsec=fwhm_arcsec, k=k,
        mode="poisson", p_blend=p_blend, dmag=dmag,
    )
    a_col = f"A{band}"
    res.update({
        "model": "genstars",
        "band": band,
        "out_file": str(out_file),
        "l_range": meta["l_range"], "b_range": meta["b_range"],
        "area_deg2": area, "f_simu": f_simu,
        "N_generated": int(np.isfinite(mags).sum()),
        "density_deg2": float(np.isfinite(mags).sum() / (f_simu * area)),
        "stars_per_cell": float(per_cell),
        "median_A_band": float(np.nanmedian(stars[a_col])) if a_col in stars else None,
    })
    return res


def run_genstars_crowding(
    out_dir: str | Path,
    *,
    l_center: float,
    b_center: float,
    half_width: float = 0.1,
    band: str = "Ks",
    mag_max: float = 30.0,
    f_simu: float = 0.01,
    seed: int = 12,
    fwhm_arcsec: float = 0.2,
    k: float = 1.0,
    p_blend: float = 0.10,
) -> dict:
    """End-to-end: genstars for a square box around (l, b), then the crowding limit."""
    i_mag = BANDS[0].index(band)
    l_range = (l_center - half_width, l_center + half_width)
    b_range = (b_center - half_width, b_center + half_width)
    stem = f"gs_l{l_center:+.3f}_b{b_center:+.3f}_w{2 * half_width:g}_{band}{mag_max:g}_f{f_simu:g}"
    out_file = run_genstars(
        Path(out_dir) / f"{stem}.dat", l_range=l_range, b_range=b_range,
        i_mag=i_mag, mag_range=(-10.0, mag_max), f_simu=f_simu, seed=seed,
    )
    res = genstars_crowding(out_file, band=band, fwhm_arcsec=fwhm_arcsec, k=k, p_blend=p_blend)

    print("=== genstars summary ===")
    print(f"Box l={l_range}, b={b_range}  area={res['area_deg2']:.4g} deg^2  fSIMU={f_simu}")
    print(f"Generated stars: {res['N_generated']:,}  ({res['stars_per_cell']:.0f} per 0.025-deg cell)")
    print(f"Model density ({band}<{mag_max:g}): {res['density_deg2']:.3e} /deg^2 "
          f"= {res['density_deg2'] / 3600**2:.3g} /arcsec^2;  median A_{band} = {res['median_A_band']:.2f}")
    m = res["m_lim"]
    print(f"Crowding-limited depth ({res['criterion']}): "
          + (f"{m:.2f}" if m is not None else "not reached") + f" in {band} (apparent, with extinction)")
    return res


# -----------------------------
# Example: one bulge field
# -----------------------------
if __name__ == "__main__":
    run_genstars_crowding(
        Path.home() / "genstars_runs/test",
        l_center=1.0, b_center=-2.0,   # inside the genstars box; bulge-dominated
        half_width=0.1,                # 0.2 x 0.2 deg box
        band="Ks", mag_max=30.0,
        f_simu=0.01,
        fwhm_arcsec=0.2, k=1.0, p_blend=0.10,
    )

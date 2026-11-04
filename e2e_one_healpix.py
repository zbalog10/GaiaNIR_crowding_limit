#!/usr/bin/env python3
"""
End-to-end test of the crowding pipeline for ONE HEALPix pixel.

Steps (production settings: NSIDE=128, Ks<30, FWHM=0.2", P_blend=0.1):
  1. Run Galaxia for the pixel and compute m_lim
     (raunGalxiaHealpix10x10.run_crowding_for_ipix_list) -> results CSV
  2. Check the Galaxia output (params, star positions, magnitude conventions)
  3. Run the single-pixel diagnostics (diagnostic_pixel.run) and check that its
     recomputed m_lim matches the CSV
  4. Run the independent recalculation (recalc_crowding_ebvcorrect.main) on the
     same output and check that its extinction-corrected m_lim agrees

Prints PASS/FAIL per check and exits non-zero if any check fails.
Re-running reuses the existing EBF (skip_existing=True); delete OUTDIR to start fresh.

    .venv/bin/python e2e_one_healpix.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")   # PNG output only, no windows

import numpy as np
import pandas as pd
import healpy as hp
import ebf

import raunGalxiaHealpix10x10 as runner
import diagnostic_pixel
import recalc_crowding_ebvcorrect as recalc


# =========================
# SETTINGS
# =========================
NSIDE = 128
L_DEG, B_DEG = 30.0, 0.0        # pixel containing this direction (centre of the production grid)
K_MIN, K_MAX = -100.0, 30.0     # as in raunGalxiaHealpix10x10.py
FWHM_ARCSEC, K_SCALE, P_BLEND = 0.2, 1.0, 0.10
MAG_KEY = "/mag0"
OUTDIR = Path.home() / "galaxia_runs/e2e_test_n128"


# =========================
# CHECK BOOKKEEPING
# =========================
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(ok)


def main() -> int:
    theta, phi = runner.lb_to_theta_phi(L_DEG, B_DEG)
    ipix = int(hp.ang2pix(NSIDE, theta, phi))
    area = runner.healpix_area_deg2(NSIDE)
    l_c, b_c = hp.pix2ang(NSIDE, ipix, lonlat=True)
    print(f"Pixel: NSIDE={NSIDE} ipix={ipix} centre (l,b)=({l_c:.3f},{b_c:.3f}) area={area:.4f} deg^2")
    print(f"Output: {OUTDIR}\n")

    # ---- 1. Galaxia + crowding limit -------------------------------------
    print("Step 1: Galaxia run + crowding limit")
    t0 = time.time()
    df = runner.run_crowding_for_ipix_list(
        nside=NSIDE, ipix_list=np.array([ipix]), outdir=str(OUTDIR),
        k_max=K_MAX, k_min=K_MIN, mag_key=MAG_KEY,
        fwhm_arcsec=FWHM_ARCSEC, k=K_SCALE, mode="poisson", p_blend=P_BLEND,
        skip_existing=True,
    )
    csv = OUTDIR / "results_e2e.csv"
    df.to_csv(csv, index=False)
    row = df.iloc[0]
    ebf_path = Path(row["ebf_path"])
    m_lim = float(row["m_lim"])
    print(f"  took {time.time() - t0:.0f} s; wrote {csv}")
    check("EBF written", ebf_path.exists(), str(ebf_path))
    check("stars generated", row["N_total"] > 0, f"N={int(row['N_total']):,}")
    check("m_lim reached", np.isfinite(m_lim), f"m_lim={m_lim:.2f}")
    check("CSV records apparent magnitudes", str(row["mag_key"]).startswith("/mag"), str(row["mag_key"]))

    # ---- 2. Galaxia output sanity ---------------------------------------
    print("\nStep 2: Galaxia output checks")
    params = ebf_path.with_suffix(".params").read_text()
    area_par = float(next(l.split()[1] for l in params.splitlines() if l.startswith("surveyArea")))
    check("params surveyArea = pixel area", abs(area_par - area) < 1e-6, f"{area_par:.6f}")

    f = str(ebf_path)
    glon, glat = np.asarray(ebf.read(f, "/glon")), np.asarray(ebf.read(f, "/glat"))
    dl = (glon - l_c + 180.0) % 360.0 - 180.0
    r_field = np.sqrt(area / np.pi)   # radius (deg) of the circular field Galaxia samples
    check("stars centred on the pixel", abs(np.median(dl)) < 0.05 and abs(np.median(glat) - b_c) < 0.05,
          f"median offset dl={np.median(dl):+.3f}, db={np.median(glat) - b_c:+.3f} deg")
    check("stars within the field radius", np.max(np.hypot(dl, glat - b_c)) < 1.05 * r_field,
          f"max sep {np.max(np.hypot(dl, glat - b_c)):.3f} deg, cone radius {r_field:.3f} deg")

    M, rad, m0 = (np.asarray(ebf.read(f, k)) for k in ("/dcmc_ks", "/rad", "/mag0"))
    d = m0 - (M + 5 * np.log10(100 * rad))
    check("mag0 = dcmc_ks + distance modulus", np.max(np.abs(d)) < 1e-3, f"max |diff| {np.max(np.abs(d)):.1e}")
    check("apparent-magnitude cut applied", m0.max() <= K_MAX + 1e-6, f"max mag0 {m0.max():.2f}")
    ebv = np.asarray(ebf.read(f, "/exbv_schlegel"))
    check("reddening available", np.all(np.isfinite(ebv)) and ebv.max() > 0,
          f"median A_Ks {0.367 * np.median(ebv):.2f} mag")

    # ---- 3. Diagnostics --------------------------------------------------
    print("\nStep 3: diagnostic_pixel")
    png = diagnostic_pixel.run(ebf_path, csv, "none")
    m_lim_diag = diagnostic_pixel.crowding_limited_depth(
        ebf_path, MAG_KEY, survey_area_deg2=diagnostic_pixel.survey_area_deg2(ebf_path),
        fwhm_arcsec=FWHM_ARCSEC, k=K_SCALE, p_blend=P_BLEND,
    )["m_lim"]
    check("diagnostic PNG written", png.exists(), str(png))
    check("diagnostic m_lim = pipeline m_lim", m_lim_diag is not None and abs(m_lim_diag - m_lim) < 1e-6,
          f"{m_lim_diag} vs {m_lim:.2f}")

    # ---- 4. Independent recalculation -----------------------------------
    print("\nStep 4: recalc_crowding_ebvcorrect")
    recalc.ROOT_DIR = OUTDIR
    recalc.OUTDIR = OUTDIR
    recalc.OUT_CSV = OUTDIR / "results_recalc.csv"
    recalc.main()
    rc = pd.read_csv(recalc.OUT_CSV)
    rrow = rc.loc[rc["ipix"] == ipix].iloc[0]
    check("recalc status ok", rrow["status"] == "ok", str(rrow["status"]))
    check("recalc used exbv_schlegel", rrow["ebv_source"] == "exbv_schlegel", str(rrow["ebv_source"]))
    # pipeline bins in 0.1 mag (bin centres); recalc uses the exact sorted magnitudes
    check("recalc m_lim_corr agrees with pipeline (within 0.1 mag)", abs(rrow["m_lim_corr"] - m_lim) <= 0.1,
          f"{rrow['m_lim_corr']:.3f} vs {m_lim:.2f}")

    # ---- Summary ---------------------------------------------------------
    n_fail = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    for name, ok, detail in results:
        if not ok:
            print(f"  FAILED: {name}  ({detail})")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

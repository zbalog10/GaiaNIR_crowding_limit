#!/usr/bin/env python3
"""
Galaxia vs genstars for one bulge field: star counts, luminosity function,
extinction and crowding-limited depth, computed with the same crowding model.

Field: centre (L, B), area AREA_DEG2. genstars samples a square box of that area,
Galaxia a circular field of the same area (geometryOption 1) around the same centre.
Both are subsampled (genstars fSIMU, Galaxia fSample); densities use the effective
area f * AREA, so N / area is the model surface density.

Magnitudes (apparent Ks, with extinction):
  genstars : Ks column (VVV E(J-Ks) map + Nishiyama+09 law, applied by genstars)
  Galaxia  : mag0 + 0.367 * exbv_schlegel        (pipeline default)
             mag0 + 0.367 * 0.53 * exbv_solar    (alternative, for comparison)

Outputs in OUT_DIR: comparison.csv, comparison.png, plus both models' raw output.
Re-running reuses existing model output.

    .venv/bin/python compare_galaxia_genstars.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ebf

from fullRun_galaxia_crowding import EXT_COEFF_KS, crowding_limited_depth_from_mags, run_galaxia_stream
import genstars_crowding as gs


# =========================
# SETTINGS
# =========================
L, B = 1.0, -2.0                 # field centre (deg); must be inside the genstars footprint
AREA_DEG2 = 0.04                 # 0.2 x 0.2 deg box for genstars; same-area circle for Galaxia
KS_MIN, KS_MAX = -10.0, 30.0     # apparent Ks range generated

GS_FSIMU = 0.01                  # genstars subsampling
GX_FSAMPLE = 0.05                # Galaxia subsampling (fraction of stars generated)
SEED = 12

FWHM_ARCSEC, K_SCALE, P_BLEND, DMAG = 0.2, 1.0, 0.10, 0.1
EBV_SOLAR_SCALE = 0.53

OUT_DIR = Path.home() / f"model_comparison/l{L:+.2f}_b{B:+.2f}".replace(".", "p")


def _stem_safe(x: float) -> str:
    return f"{x:+.3f}".replace("+", "p").replace("-", "m").replace(".", "d")


# =========================
# MODEL RUNS
# =========================
def run_galaxia_field() -> dict:
    workdir = OUT_DIR / "galaxia"
    stem = f"gx_l{_stem_safe(L)}_b{_stem_safe(B)}_Ks{KS_MAX:g}_f{GX_FSAMPLE:g}".replace(".", "d")
    ebf_path = workdir / f"{stem}.ebf"
    if ebf_path.exists():
        print(f"Reusing existing Galaxia output: {ebf_path}")
    else:
        ebf_path, _, _ = run_galaxia_stream(
            workdir=workdir, output_file_stem=stem, l_deg=L, b_deg=B,
            survey_area_deg2=AREA_DEG2, k_min=KS_MIN, k_max=KS_MAX,
            f_sample=GX_FSAMPLE, seed=SEED,
        )
    f = str(ebf_path)
    rd = lambda k: np.asarray(ebf.read(f, k), dtype=float)
    m0 = rd("/mag0")
    a_schlegel = EXT_COEFF_KS * rd("/exbv_schlegel")
    a_solar = EXT_COEFF_KS * EBV_SOLAR_SCALE * rd("/exbv_solar")
    return {
        "model": "Galaxia",
        "f": GX_FSAMPLE,
        "dist_kpc": rd("/rad"),
        "bulge": rd("/popid") == 9,
        "variants": {
            "Galaxia (Schlegel)": (m0 + a_schlegel, a_schlegel),
            "Galaxia (0.53 solar)": (m0 + a_solar, a_solar),
        },
        "ks_noext": m0,
        "path": f,
    }


def run_genstars_field() -> dict:
    hw = np.sqrt(AREA_DEG2) / 2
    stem = f"gs_l{_stem_safe(L)}_b{_stem_safe(B)}_Ks{KS_MAX:g}_f{GS_FSIMU:g}".replace(".", "d")
    out = gs.run_genstars(
        OUT_DIR / "genstars" / f"{stem}.dat",
        l_range=(L - hw, L + hw), b_range=(B - hw, B + hw),
        i_mag=gs.BANDS[0].index("Ks"), mag_range=(KS_MIN, KS_MAX),
        f_simu=GS_FSIMU, seed=SEED,
    )
    stars, meta = gs.read_genstars(out)
    ks = stars["Ks"].to_numpy(float)
    a = stars["AKs"].to_numpy(float)
    return {
        "model": "genstars",
        "f": GS_FSIMU,
        "dist_kpc": stars["Dist"].to_numpy(float) / 1000.0,
        "bulge": stars["cls"].to_numpy() == 8,
        "variants": {"genstars (VVV)": (ks, a)},
        "ks_noext": ks - a,
        "path": str(out),
        "area_check": meta["area_deg2"],
    }


# =========================
# ANALYSIS
# =========================
def summarise(run: dict) -> list[dict]:
    area_eff = run["f"] * AREA_DEG2
    rows = []
    for name, (ks, a) in run["variants"].items():
        ok = np.isfinite(ks)
        res = crowding_limited_depth_from_mags(
            ks[ok], survey_area_deg2=area_eff, fwhm_arcsec=FWHM_ARCSEC, k=K_SCALE,
            p_blend=P_BLEND, dmag=DMAG,
        )
        res0 = crowding_limited_depth_from_mags(
            run["ks_noext"], survey_area_deg2=area_eff, fwhm_arcsec=FWHM_ARCSEC, k=K_SCALE,
            p_blend=P_BLEND, dmag=DMAG,
        )
        m_lim = res["m_lim"]
        rows.append({
            "variant": name,
            "N_generated": int(ok.sum()),
            "sampling_f": run["f"],
            "density_deg2_Ks<30": ok.sum() / area_eff,
            "density_deg2_Ks<20": (ks < 20).sum() / area_eff,
            "density_deg2_Ks<m_lim": (ks < m_lim).sum() / area_eff if m_lim else np.nan,
            "N_brighter_than_m_lim": int((ks < m_lim).sum()) if m_lim else 0,
            "median_A_Ks": float(np.median(a)),
            "p95_A_Ks": float(np.percentile(a, 95)),
            "bulge_fraction": float(run["bulge"].mean()),
            "median_dist_kpc": float(np.median(run["dist_kpc"])),
            "m_lim": m_lim if m_lim is not None else np.nan,
            "m_lim_no_extinction": res0["m_lim"] if res0["m_lim"] is not None else np.nan,
            "P_max": float(np.max(res["p_ge1"])),
            "_curve": (res["m_centers"], res["p_ge1"]),
            "_ks": ks[ok], "_a": a, "_area_eff": area_eff,
        })
    return rows


def plot(rows: list[dict], gx: dict, gsr: dict, out_png: Path) -> None:
    colors = {"Galaxia (Schlegel)": "tab:blue", "Galaxia (0.53 solar)": "tab:cyan",
              "genstars (VVV)": "tab:red"}
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    bins = np.arange(KS_MIN if KS_MIN > 0 else 5.0, KS_MAX + 6, DMAG * 2)

    ax = axs[0, 0]
    for r in rows:
        h, e = np.histogram(r["_ks"], bins=bins)
        c = 0.5 * (e[:-1] + e[1:])
        ax.step(c, h / r["_area_eff"] / np.diff(e), where="mid", color=colors[r["variant"]],
                label=r["variant"])
    ax.set_yscale("log")
    ax.set_xlabel(r"$K_s$ (apparent, with extinction)")
    ax.set_ylabel(r"stars deg$^{-2}$ mag$^{-1}$")
    ax.set_title("Luminosity function")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axs[0, 1]
    for r in rows:
        m, p = r["_curve"]
        ax.plot(m, p, color=colors[r["variant"]], label=f"{r['variant']}: m_lim={r['m_lim']:.2f}")
        if np.isfinite(r["m_lim"]):
            ax.axvline(r["m_lim"], color=colors[r["variant"]], linestyle="--", linewidth=1)
    ax.axhline(P_BLEND, color="k", linestyle=":", linewidth=1)
    ax.set_xlim(10, KS_MAX + 3)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$K_s$")
    ax.set_ylabel(r"$P(\geq 1$ neighbour within %.2g$''$)" % (K_SCALE * FWHM_ARCSEC))
    ax.set_title("Blending probability and crowding limit")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    ax = axs[1, 0]
    abins = np.linspace(0, max(np.percentile(r["_a"], 99.5) for r in rows), 80)
    for r in rows:
        ax.hist(r["_a"], bins=abins, histtype="step", density=True, color=colors[r["variant"]],
                label=f"{r['variant']}: median {r['median_A_Ks']:.2f}")
    ax.set_xlabel(r"$A_{K_s}$ (mag)")
    ax.set_ylabel("normalised")
    ax.set_title("Extinction per star")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axs[1, 1]
    dbins = np.linspace(0, 20, 81)
    for run, col in ((gx, "tab:blue"), (gsr, "tab:red")):
        area_eff = run["f"] * AREA_DEG2
        h, e = np.histogram(run["dist_kpc"], bins=dbins)
        ax.step(0.5 * (e[:-1] + e[1:]), h / area_eff / np.diff(e), where="mid", color=col,
                label=f"{run['model']} (bulge fraction {run['bulge'].mean():.2f})")
    ax.set_yscale("log")
    ax.set_xlabel("distance (kpc)")
    ax.set_ylabel(r"stars deg$^{-2}$ kpc$^{-1}$")
    ax.set_title("Distance distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Galaxia vs genstars at (l, b) = ({L:g}, {B:g}), area {AREA_DEG2:g} deg², "
        f"FWHM={FWHM_ARCSEC}\", k={K_SCALE}, P_blend={P_BLEND}", fontsize=11)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Field (l,b)=({L},{B}), area={AREA_DEG2} deg^2, Ks in [{KS_MIN},{KS_MAX}] -> {OUT_DIR}\n")
    gsr = run_genstars_field()
    gx = run_galaxia_field()

    rows = summarise(gx) + summarise(gsr)
    table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    table.insert(0, "l", L)
    table.insert(1, "b", B)
    table.insert(2, "area_deg2", AREA_DEG2)
    table.to_csv(OUT_DIR / "comparison.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", "{:.4g}".format):
        print("\n" + table.drop(columns=["l", "b", "area_deg2"]).set_index("variant").T.to_string())
    plot(rows, gx, gsr, OUT_DIR / "comparison.png")
    print(f"Wrote {OUT_DIR / 'comparison.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_full_galaxia_crowding.py

End-to-end:
- Write Galaxia v0.2 parameter file
- Run Galaxia with live stdout streaming
- Read output EBF
- Compute crowding-limited depth vs magnitude for a chosen band

Tested against Galaxia v0.2 parameter style (as in Examples/myparameterfile).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import ebf


# -----------------------------
# 1) Galaxia runner (v0.2, -r)
# -----------------------------

def _format_value(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return f"{v:.10g}"
    return str(v)


def write_galaxia_param_file(path: Union[str, Path], params: Dict[str, object]) -> Path:
    path = Path(path)
    lines = [f"{k:<35} {_format_value(v)}" for k, v in params.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_galaxia_stream(
    workdir: Union[str, Path],
    output_file_stem: str,
    *,
    l_deg: float,
    b_deg: float,
    survey_area_deg2: float,
    k_min: float = 0.0,
    k_max: float = 20.0,
    photo_sys: str = "DCMC",
    magcolor_names: str = "Ks,J-Ks",
    seed: int = 12,
    f_sample: float = 1.0,
    galaxia_exe: str = str(Path.home() / ".local/bin/galaxia"),
    extra_params: Optional[Dict[str, object]] = None,
    env: Optional[Dict[str, str]] = None,
) -> tuple[Path, str, int]:
    """
    Run Galaxia and stream its output live.

    Returns
    -------
    (ebf_path, stdout_text, return_code)
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    params = {
        "outputFile": output_file_stem,
        "outputDir": str(workdir) + "/",
        "photoSys": photo_sys,
        "magcolorNames": magcolor_names,
        "appMagLimits[0]": k_min,
        "appMagLimits[1]": k_max,
        "absMagLimits[0]": -1000,
        "absMagLimits[1]": 1000,
        "colorLimits[0]": -1000,
        "colorLimits[1]": 1000,
        "geometryOption": 1,   # Galactic l,b
        "longitude": l_deg,
        "latitude": b_deg,
        "surveyArea": survey_area_deg2,
        "fSample": f_sample,
        "popID": -1,
        "warpFlareOn": 1,
        "seed": seed,
        "r_max": 1000,
        "starType": 0,
        "photoError": 0,
    }
    if extra_params:
        params.update(extra_params)

    param_path = workdir / f"{output_file_stem}.params"
    write_galaxia_param_file(param_path, params)

    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    print(f"\n=== Running Galaxia: {output_file_stem} ===")
    print(f"Command: {galaxia_exe} -r {param_path}\n")

    proc = subprocess.Popen(
        [galaxia_exe, "-r", str(param_path)],
        cwd=str(workdir),
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    stdout_lines: list[str] = []
    assert proc.stdout is not None

    for line in proc.stdout:
        print(line, end="")       # live stream
        stdout_lines.append(line)

    proc.wait()
    stdout_text = "".join(stdout_lines)
    rc = int(proc.returncode or 0)

    ebf_path = workdir / f"{output_file_stem}.ebf"

    # Galaxia v0.2 sometimes returns 1 even though the EBF is written.
    if rc != 0 and not ebf_path.exists():
        raise RuntimeError(
            "Galaxia failed and did not produce an output .ebf.\n"
            f"Exit code: {rc}\n"
            "---- Galaxia output ----\n"
            f"{stdout_text}"
        )

    if not ebf_path.exists():
        # fallback: try to parse the "File written-" line
        for line in reversed(stdout_lines):
            if "File written-" in line:
                maybe = line.split("File written-")[-1].strip()
                if maybe:
                    p = Path(maybe)
                    if p.exists():
                        ebf_path = p
                        break

    if not ebf_path.exists():
        raise FileNotFoundError(
            "Galaxia finished but no .ebf file was found.\n"
            f"Exit code: {rc}\n"
            f"{stdout_text}"
        )

    print(f"\n=== Galaxia done. Output: {ebf_path} (exit code {rc}) ===\n")
    return ebf_path, stdout_text, rc


# ----------------------------------------
# 2) Crowding-limited depth calculator
# ----------------------------------------

# Extinction coefficient A_Ks / E(B-V) (Galaxia documentation, UKIRT K).
EXT_COEFF_KS = 0.367


def load_apparent_mags(
    ebf_file: Union[str, Path],
    mag_key: str = "/mag0",
    *,
    ext_coeff: float | None = EXT_COEFF_KS,
    ebv_key: str = "/exbv_schlegel",
) -> np.ndarray:
    """
    Return extinction-corrected apparent magnitudes from a Galaxia EBF file.

    Galaxia column conventions (verified against Galaxia 0.7.2 output):
    - /dcmc_ks, /dcmc_j, ... (photometric-system columns) are ABSOLUTE magnitudes.
    - /mag0, /mag1, /mag2 are APPARENT magnitudes WITHOUT extinction,
      mag0 = M + 5*log10(100*rad), for the bands in magcolorNames
      ("Ks,J-Ks" -> mag0 = Ks, mag1 = J, mag2 = Ks).
    - Extinction is not applied by Galaxia; add A = ext_coeff * E(B-V) here.

    Set ext_coeff=None to skip the extinction term. ext_coeff must match the
    band in mag_key (the default 0.367 is for Ks).
    """
    ebf_file = str(ebf_file)
    if not mag_key.startswith("/mag"):
        print(
            f"WARNING: {mag_key} is a photometric-system column, i.e. an ABSOLUTE "
            "magnitude in Galaxia output; use /mag0 (apparent) for crowding limits."
        )

    mags = np.asarray(ebf.read(ebf_file, mag_key), dtype=float)
    if ext_coeff is not None:
        ebv = np.asarray(ebf.read(ebf_file, ebv_key), dtype=float)
        mags = mags + ext_coeff * ebv
    return mags


def crowding_limited_depth(
    ebf_file: Union[str, Path],
    mag_key: str = "/mag0",
    *,
    survey_area_deg2: float,
    fwhm_arcsec: float,
    k: float = 1.0,
    dmag: float = 0.1,
    mode: str = "poisson",          # "poisson" or "beams"
    p_blend: float = 0.10,          # used if mode="poisson"
    beams_per_source: float = 30.0, # used if mode="beams"
    mag_min: float | None = None,
    mag_max: float | None = None,
    ext_coeff: float | None = EXT_COEFF_KS,
    ebv_key: str = "/exbv_schlegel",
) -> Dict[str, object]:
    """
    Compute crowding-limited depth using:
    - Poisson blending probability threshold, or
    - beams-per-source threshold.

    Uses cumulative density rho(<m) derived from extinction-corrected apparent
    magnitudes (see load_apparent_mags).
    """
    ebf_file = Path(ebf_file)

    mags = load_apparent_mags(ebf_file, mag_key, ext_coeff=ext_coeff, ebv_key=ebv_key)
    if not np.isfinite(mags).any():
        raise ValueError(f"No finite magnitudes found at {mag_key} in {ebf_file}")

    res = crowding_limited_depth_from_mags(
        mags,
        survey_area_deg2=survey_area_deg2, fwhm_arcsec=fwhm_arcsec, k=k, dmag=dmag,
        mode=mode, p_blend=p_blend, beams_per_source=beams_per_source,
        mag_min=mag_min, mag_max=mag_max,
    )
    res.update({
        "mag_key": mag_key,
        "ext_coeff": ext_coeff,
        "ebv_key": ebv_key if ext_coeff is not None else None,
    })
    return res


def crowding_limited_depth_from_mags(
    mags: np.ndarray,
    *,
    survey_area_deg2: float,
    fwhm_arcsec: float,
    k: float = 1.0,
    dmag: float = 0.1,
    mode: str = "poisson",          # "poisson" or "beams"
    p_blend: float = 0.10,          # used if mode="poisson"
    beams_per_source: float = 30.0, # used if mode="beams"
    mag_min: float | None = None,
    mag_max: float | None = None,
) -> Dict[str, object]:
    """
    Model-independent core of the crowding calculation (Galaxia, genstars, ...).

    mags must be apparent magnitudes (extinction included) of the stars in a field of
    survey_area_deg2. For a subsampled catalogue (e.g. genstars fSIMU < 1) pass the
    effective area fSIMU * area, so that N / area is the true surface density.
    """
    mags = np.asarray(mags, dtype=float)
    mags = mags[np.isfinite(mags)]
    if mags.size == 0:
        raise ValueError("No finite magnitudes")

    if mag_min is None:
        mag_min = float(np.floor(np.min(mags)))
    if mag_max is None:
        mag_max = float(np.ceil(np.max(mags)))

    # confusion area
    r_arcsec = k * fwhm_arcsec
    A_arcsec2 = np.pi * r_arcsec**2

    area_arcsec2 = survey_area_deg2 * (3600.0**2)

    bins = np.arange(mag_min, mag_max + dmag, dmag)
    hist, edges = np.histogram(mags, bins=bins)
    cum_counts = np.cumsum(hist)
    m_centers = 0.5 * (edges[:-1] + edges[1:])

    rho_cum = cum_counts / area_arcsec2                 # stars/arcsec^2
    lam = rho_cum * A_arcsec2
    p_ge1 = 1.0 - np.exp(-lam)
    #bps = np.where(rho_cum > 0, 1.0 / (rho_cum * A_arcsec2), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        bps = 1.0 / (rho_cum * A_arcsec2)
    bps[~np.isfinite(bps)] = np.inf

    mode = mode.lower().strip()
    if mode == "poisson":
        max_p = float(np.max(p_ge1))
        idx = np.where(p_ge1 >= p_blend)[0]
        if len(idx) == 0:
            m_lim = None
            crit_val = max_p
        else:
            m_lim = float(m_centers[idx[0]])
            crit_val = float(p_ge1[idx[0]])
        crit = f"P(>=1 neighbor) >= {p_blend} (max reached={max_p:.3g})"
    #if mode == "poisson":
    #    idx = np.where(p_ge1 >= p_blend)[0]
    #    m_lim = float(m_centers[idx[0]]) if len(idx) else float("nan")
    #    crit = f"P(>=1 neighbor) >= {p_blend}"
    #    crit_val = float(p_ge1[idx[0]]) if len(idx) else float("nan")
    elif mode == "beams":
        idx = np.where(bps <= beams_per_source)[0]
        m_lim = float(m_centers[idx[0]]) if len(idx) else float("nan")
        crit = f"beams/source <= {beams_per_source}"
        crit_val = float(bps[idx[0]]) if len(idx) else float("nan")
    else:
        raise ValueError("mode must be 'poisson' or 'beams'")

    return {
        "m_lim": m_lim,
        "criterion": crit,
        "criterion_value_at_m_lim": crit_val,
        "fwhm_arcsec": fwhm_arcsec,
        "k": k,
        "r_arcsec": r_arcsec,
        "A_arcsec2": A_arcsec2,
        "survey_area_deg2": survey_area_deg2,
        # curves (for plotting / debugging)
        "m_centers": m_centers,
        "rho_cum_arcsec2": rho_cum,
        "p_ge1": p_ge1,
        "beams_per_source": bps,
        "N_total": int(mags.size),
    }


# -----------------------------
# 3) End-to-end analysis driver
# -----------------------------

def run_full_analysis(
    *,
    workdir: Union[str, Path],
    output_file_stem: str,
    l_deg: float,
    b_deg: float,
    survey_area_deg2: float,
    k_max: float,
    fwhm_arcsec: float,
    k: float = 1.0,
    mode: str = "poisson",
    p_blend: float = 0.10,
    beams_per_source: float = 30.0,
    mag_key: str = "/mag0",
    ext_coeff: float | None = EXT_COEFF_KS,
    ebv_key: str = "/exbv_schlegel",
) -> Dict[str, object]:
    ebf_path, stdout_text, rc = run_galaxia_stream(
        workdir=workdir,
        output_file_stem=output_file_stem,
        l_deg=l_deg,
        b_deg=b_deg,
        survey_area_deg2=survey_area_deg2,
        k_max=k_max,
    )

    res = crowding_limited_depth(
        ebf_file=ebf_path,
        mag_key=mag_key,
        survey_area_deg2=survey_area_deg2,
        fwhm_arcsec=fwhm_arcsec,
        k=k,
        mode=mode,
        p_blend=p_blend,
        beams_per_source=beams_per_source,
        dmag=0.1,
        ext_coeff=ext_coeff,
        ebv_key=ebv_key,
    )

    # Print a compact summary
    dens_deg2 = res["N_total"] / survey_area_deg2
    dens_arcsec2 = dens_deg2 / (3600.0**2)

    print("=== Summary ===")
    print(f"Field (l,b) = ({l_deg:.2f}, {b_deg:.2f})  area={survey_area_deg2} deg^2")
    print(f"Total stars in file: {res['N_total']:,}")
    print(f"Mean surface density: {dens_deg2:,.1f} /deg^2 = {dens_arcsec2:.3e} /arcsec^2")
    ext_desc = f"+ {ext_coeff}*{ebv_key.lstrip('/')}" if ext_coeff is not None else "(no extinction)"
    if res["m_lim"] is None:
        print(f"Crowding-limited depth ({res['criterion']}): not reached within simulated mags")
    else:
        print(f"Crowding-limited depth ({res['criterion']}): {res['m_lim']:.2f} using {mag_key} {ext_desc}")

    print(f"Assumed FWHM={fwhm_arcsec}\"  k={k}  => r={res['r_arcsec']:.3f}\"  A={res['A_arcsec2']:.4g} arcsec^2")
    print(f"Galaxia exit code: {rc} (EBF exists => accepted)\n")

    return {
        "ebf_path": str(ebf_path),
        "galaxia_exit_code": rc,
        "galaxia_stdout": stdout_text,
        **res,
    }


# -----------------------------
# Main: run one test field
# -----------------------------
if __name__ == "__main__":
    # Example: bulge field just above the Galactic Centre (l=0, b=+5; very high density).
    # A small area keeps the test quick: crowding depends on surface density,
    # so m_lim is close to the 1 deg^2 result, just noisier.
    result = run_full_analysis(
        workdir=str(Path.home() / "galaxia_runs/bulge_test"),
        output_file_stem="l0_b5_Ks25_area0p01",
        l_deg=0.0,
        b_deg=5.0,
        survey_area_deg2=0.01,
        k_max=25.0,
        fwhm_arcsec=0.20,   # set your effective resolution here
        k=1.0,
        mode="poisson",
        p_blend=0.10,
        mag_key="/mag0",          # apparent Ks; extinction added from exbv_schlegel
    )

    # If you want the beams/source criterion instead:
    # result = run_full_analysis(..., mode="beams", beams_per_source=30.0)

import os
from pathlib import Path
import numpy as np
import pandas as pd
import healpy as hp
from fullRun_galaxia_crowding import run_galaxia_stream, crowding_limited_depth

# assume these already exist in your script:
# - run_galaxia_stream(...)
# - crowding_limited_depth(...)

def healpix_centers_galactic(nside: int, nest: bool = False):
    """
    Return arrays (ipix, l_deg, b_deg) for all HEALPix pixels at nside.
    """
    npix = hp.nside2npix(nside)
    ipix = np.arange(npix, dtype=int)

    # healpy.pix2ang gives theta,phi in radians:
    # theta = colatitude (0 at north pole), phi = longitude
    theta, phi = hp.pix2ang(nside, ipix, nest=nest)
    l_deg = np.degrees(phi)
    b_deg = 90.0 - np.degrees(theta)
    return ipix, l_deg, b_deg


def healpix_pixel_area_deg2(nside: int) -> float:
    """
    HEALPix pixel area in deg^2.
    """
    area_sr = hp.nside2pixarea(nside)            # steradians
    area_deg2 = area_sr * (180.0 / np.pi) ** 2
    return float(area_deg2)


def run_crowding_per_healpix(
    *,
    outdir: str,
    nside: int = 16,
    nest: bool = False,
    mag_key: str = "/dcmc_ks",
    k_max: float = 22.0,
    fwhm_arcsec: float = 0.2,
    k: float = 1.0,
    mode: str = "poisson",
    p_blend: float = 0.10,
    beams_per_source: float = 30.0,
    b_min: float | None = None,
    b_max: float | None = None,
    max_pixels: int | None = None,
    skip_existing: bool = True,
    galaxia_exe: str = str(Path.home() / ".local/bin/galaxia"),
) -> pd.DataFrame:
    """
    Run Galaxia + crowding limit for each HEALPix pixel center.

    Options:
      - b_min/b_max: restrict to latitude band (e.g. b_min=-5, b_max=+5 for plane strip)
      - max_pixels: run only first N pixels (quick test)
      - skip_existing: if EBF exists for pixel, skip Galaxia and just analyze it

    Returns a pandas DataFrame with one row per pixel.
    """
    outdir = str(Path(outdir))
    Path(outdir).mkdir(parents=True, exist_ok=True)

    survey_area_deg2 = healpix_pixel_area_deg2(nside)
    ipix, l_deg, b_deg = healpix_centers_galactic(nside, nest=nest)

    # Optional band selection
    mask = np.ones_like(ipix, dtype=bool)
    if b_min is not None:
        mask &= (b_deg >= b_min)
    if b_max is not None:
        mask &= (b_deg <= b_max)

    ipix = ipix[mask]
    l_deg = l_deg[mask]
    b_deg = b_deg[mask]

    if max_pixels is not None:
        ipix = ipix[:max_pixels]
        l_deg = l_deg[:max_pixels]
        b_deg = b_deg[:max_pixels]

    rows = []

    for i, l, b in zip(ipix, l_deg, b_deg):
        stem = f"hp_n{nside}_{'nest' if nest else 'ring'}_{i:06d}_Ks{k_max:g}"
        workdir = Path(outdir) / f"hp_{i:06d}"
        workdir.mkdir(parents=True, exist_ok=True)

        ebf_path = workdir / f"{stem}.ebf"

        # Either run Galaxia or reuse existing output
        if skip_existing and ebf_path.exists():
            stdout_text = ""
            rc = 0
        else:
            ebf_path, stdout_text, rc = run_galaxia_stream(
                workdir=str(workdir),
                output_file_stem=stem,
                l_deg=float(l),
                b_deg=float(b),
                survey_area_deg2=survey_area_deg2,
                k_max=float(k_max),
                galaxia_exe=galaxia_exe,
            )

        # Crowding calculation
        res = crowding_limited_depth(
            ebf_file=str(ebf_path),
            mag_key=mag_key,
            survey_area_deg2=survey_area_deg2,
            fwhm_arcsec=float(fwhm_arcsec),
            k=float(k),
            mode=mode,
            p_blend=float(p_blend),
            beams_per_source=float(beams_per_source),
            dmag=0.1,
            # optional: enforce consistent range
            mag_min=0.0,
            mag_max=float(k_max),
        )

        dens_deg2 = res["N_total"] / survey_area_deg2
        dens_arcsec2 = dens_deg2 / (3600.0**2)

        rows.append({
            "nside": nside,
            "nest": nest,
            "ipix": int(i),
            "l_deg": float(l),
            "b_deg": float(b),
            "survey_area_deg2": survey_area_deg2,
            "k_max": float(k_max),
            "mag_key": mag_key,
            "N_total": int(res["N_total"]),
            "density_deg2": float(dens_deg2),
            "density_arcsec2": float(dens_arcsec2),
            "mode": mode,
            "p_blend": float(p_blend) if mode == "poisson" else np.nan,
            "beams_per_source": float(beams_per_source) if mode == "beams" else np.nan,
            "fwhm_arcsec": float(fwhm_arcsec),
            "k": float(k),
            "A_arcsec2": float(res["A_arcsec2"]),
            "m_lim": res["m_lim"],   # may be None if not reached
            "galaxia_exit_code": int(rc),
            "ebf_path": str(ebf_path),
        })

        # Small progress ping
        print(f"[hp {i:06d}] (l,b)=({l:.2f},{b:.2f})  N={res['N_total']:,}  m_lim={res['m_lim']}")

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    # Good first test: a narrow Galactic plane strip to keep runtime manageable
    df = run_crowding_per_healpix(
        outdir=str(Path.home() / "galaxia_runs/healpix_plane_n16"),
        nside=8,
        nest=False,
        k_max=22.0,
        fwhm_arcsec=0.2,
        k=1.0,
        mode="poisson",
        p_blend=0.10,
        mag_key="/dcmc_ks",
        b_min=-5.0,
        b_max=+5.0,
        # max_pixels=50,   # uncomment for a quick smoke test
        skip_existing=True,
    )

    out_csv = str(Path.home() / "galaxia_runs/healpix_plane_n16/results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}")

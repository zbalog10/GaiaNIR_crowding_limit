import numpy as np
import ebf

def crowding_limited_depth_from_ebf(
    ebf_file: str,
    mag_key: str,
    survey_area_deg2: float,
    fwhm_arcsec: float,
    k: float = 1.0,
    p_blend: float = 0.10,
    mag_min: float | None = None,
    mag_max: float | None = None,
    dmag: float = 0.1,
    mode: str = "poisson",         # "poisson" or "beams"
    beams_per_source: float = 30.0 # used if mode="beams"
):
    """
    Compute crowding-limited depth using a Poisson blending model or beams/source criterion.

    Parameters
    ----------
    ebf_file : str
        Path to Galaxia output EBF.
    mag_key : str
        Dataset path for magnitude, e.g. "/ubv_v".
    survey_area_deg2 : float
        The simulated sky area in deg^2 (from your Galaxia parameter file surveyArea).
    fwhm_arcsec : float
        PSF FWHM in arcsec (or an effective resolution scale).
    k : float
        Multiplier to define the effective confusion radius: r = k * FWHM.
        Typical: k=0.5..1.0 depending on what you call “blended”.
    p_blend : float
        Blending probability threshold for mode="poisson" (e.g., 0.1, 0.3, 0.5).
    dmag : float
        Magnitude bin size used to compute cumulative densities.
    mode : str
        "poisson": find m where P(>=1 neighbor) >= p_blend
        "beams":   find m where beams/source <= beams_per_source
    beams_per_source : float
        Typical confusion criteria: 30–50 beams per source.

    Returns
    -------
    result : dict
        Contains limiting magnitude, confusion area, and the curves.
    """
    # Read magnitudes
    mags = ebf.read(ebf_file, mag_key)
    mags = np.asarray(mags)
    mags = mags[np.isfinite(mags)]

    if mag_min is None:
        mag_min = float(np.floor(np.min(mags)))
    if mag_max is None:
        mag_max = float(np.ceil(np.max(mags)))

    # Effective confusion area per source (arcsec^2)
    # Define "confusion radius" r = k * FWHM
    r_arcsec = k * fwhm_arcsec
    A_arcsec2 = np.pi * r_arcsec**2

    # Convert simulated area to arcsec^2
    area_arcsec2 = survey_area_deg2 * (3600.0**2)
    print(area_arcsec2)

    # Histogram & cumulative counts
    bins = np.arange(mag_min, mag_max + dmag, dmag)
    hist, edges = np.histogram(mags, bins=bins)
    cum_counts = np.cumsum(hist)  # N(<m_bin_upper)
    print(cum_counts)

    # Use bin centers for reporting
    m_centers = 0.5 * (edges[:-1] + edges[1:])

    # Cumulative surface density rho(<m) [stars/arcsec^2]
    rho_cum = cum_counts / area_arcsec2

    # Confusion metrics
    lam = rho_cum * A_arcsec2
    p_ge1 = 1.0 - np.exp(-lam)
    bps = np.where(rho_cum > 0, 1.0 / (rho_cum * A_arcsec2), np.inf)  # beams/source

    if mode.lower() == "poisson":
        target = p_blend
        idx = np.where(p_ge1 >= target)[0]
        m_lim = float(m_centers[idx[0]]) if len(idx) else np.nan
        criterion_value = float(p_ge1[idx[0]]) if len(idx) else np.nan
        criterion_name = f"P(>=1 neighbor) >= {target}"
    elif mode.lower() == "beams":
        target = beams_per_source
        idx = np.where(bps <= target)[0]
        m_lim = float(m_centers[idx[0]]) if len(idx) else np.nan
        criterion_value = float(bps[idx[0]]) if len(idx) else np.nan
        criterion_name = f"beams/source <= {target}"
    else:
        raise ValueError("mode must be 'poisson' or 'beams'")

    return {
        "m_lim": m_lim,
        "criterion": criterion_name,
        "criterion_value_at_m_lim": criterion_value,
        "A_arcsec2": A_arcsec2,
        "r_arcsec": r_arcsec,
        "m_centers": m_centers,
        "rho_cum_arcsec2": rho_cum,
        "p_ge1": p_ge1,
        "beams_per_source": bps,
    }


# ---- Example usage (your case) ----
if __name__ == "__main__":
    ebf_file = "/home/balog/galaxia_test/galaxy1.ebf"
    survey_area_deg2 = 1.0          # set to your 'surveyArea' value
    fwhm_arcsec = 0.20              # example: 0.2" (tune to your instrument)
    mag_key = "/ubv_v"

    res = crowding_limited_depth_from_ebf(
        ebf_file=ebf_file,
        mag_key=mag_key,
        survey_area_deg2=survey_area_deg2,
        fwhm_arcsec=fwhm_arcsec,
        k=1.0,              # confusion radius = 1*FWHM
        mode="poisson",
        p_blend=0.10,       # 10% blended probability
        dmag=0.1
    )

    print("Effective confusion radius [arcsec]:", res["r_arcsec"])
    print("Confusion area A [arcsec^2]:", res["A_arcsec2"])
    print("Crowding-limited depth m_lim:", res["m_lim"])
    print("Criterion:", res["criterion"])

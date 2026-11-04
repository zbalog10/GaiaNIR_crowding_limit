import numpy as np
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------
# USER SETTINGS
# -----------------------
CSV_FILE = "/home/balog/galaxia_runs/healpix_n128_plane_10x10/results_plane_10x10.csv"
NSIDE = 128

# Column to plot:
# choose ONE of these (both should exist in your CSV):
VALUE_COL = "density_arcsec2"   # stars / arcsec^2
# VALUE_COL = "density_deg2"    # stars / deg^2

# Zoom box (same as used in the run)
L_MIN, L_MAX = 25.0, 35.0
B_MIN, B_MAX = -2.5, +2.5

OUTDIR = Path(CSV_FILE).parent
OUTDIR.mkdir(parents=True, exist_ok=True)


# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv(CSV_FILE)

npix = hp.nside2npix(NSIDE)
m = np.full(npix, hp.UNSEEN, dtype=float)

for _, row in df.iterrows():
    ipix = int(row["ipix"])
    val = row[VALUE_COL]
    if np.isfinite(val):
        m[ipix] = val


# -----------------------
# 1) FULL-SKY VIEW
# -----------------------
plt.figure(figsize=(10, 6))
hp.mollview(
    m,
    title=f"Stellar surface density ({VALUE_COL})  NSIDE={NSIDE}",
    unit=VALUE_COL,
    norm="log",            # IMPORTANT: densities span many orders of magnitude
    badcolor="lightgray",
)
hp.graticule()

out_allsky = OUTDIR / f"density_{VALUE_COL}_mollview.png"
plt.savefig(out_allsky, dpi=200, bbox_inches="tight")
print(f"Wrote: {out_allsky}")


# -----------------------
# 2) ZOOMED CARTESIAN VIEW
# -----------------------
plt.figure(figsize=(8, 6))
hp.cartview(
    m,
    coord="G",
    lonra=[L_MIN, L_MAX],
    latra=[B_MIN, B_MAX],
    title=f"Zoom: stellar surface density ({VALUE_COL})",
    unit=VALUE_COL,
    norm="log",
    badcolor="lightgray",
    cbar=True,
)
hp.graticule()

out_zoom = OUTDIR / f"density_{VALUE_COL}_cartview_zoom.png"
plt.savefig(out_zoom, dpi=250, bbox_inches="tight")
print(f"Wrote: {out_zoom}")

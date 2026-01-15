from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np

import healpy as hp
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from astroquery.besancon import Besancon


@dataclass(frozen=True)
class HealpixQuery:
    nside: int
    ipix: int
    nest: bool = False
    frame: str = "galactic"  # "galactic" or "icrs"

    def center_galactic_lonlat_deg(self) -> tuple[float, float]:
        """
        Return (l, b) in degrees for the pixel center.
        If frame="icrs", ipix is interpreted as RA/Dec pixelization and converted to Galactic.
        """
        lon, lat = hp.pix2ang(self.nside, self.ipix, nest=self.nest, lonlat=True)

        if self.frame.lower() == "galactic":
            l_deg, b_deg = float(lon), float(lat)
        elif self.frame.lower() == "icrs":
            c = SkyCoord(ra=lon * u.deg, dec=lat * u.deg, frame="icrs").galactic
            l_deg, b_deg = float(c.l.deg), float(c.b.deg)
        else:
            raise ValueError("frame must be 'galactic' or 'icrs'")

        # normalize l into [0, 360)
        l_deg = l_deg % 360.0
        return l_deg, b_deg

    def pixel_area_deg2(self) -> float:
        """HEALPix pixel area in square degrees."""
        area_sr = hp.nside2pixarea(self.nside)  # steradians
        return float(area_sr * (180.0 / np.pi) ** 2)


def _cache_key(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def query_besancon_for_healpix(
    hpq: HealpixQuery,
    email: str,
    cache_dir: str | Path = ".bgm_cache",
    retrieve_file: bool = True,
    **besancon_kwargs,
) -> Table:
    """
    Query Besançon using astroquery for a HEALPix pixel (approximate: circular field with same area).

    Returns:
        astropy.table.Table: Besançon simulated catalog.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    l_deg, b_deg = hpq.center_galactic_lonlat_deg()
    area_deg2 = hpq.pixel_area_deg2()

    # Minimal parameter set; you can pass more via besancon_kwargs if needed.
    params = {
        "glon": round(l_deg, 8),
        "glat": round(b_deg, 8),
        "area": round(area_deg2, 12),
        "smallfield": True,
        "retrieve_file": retrieve_file,
        # include any extra kwargs in cache key:
        "extra": besancon_kwargs,
    }
    key = _cache_key(params)
    cache_path = cache_dir / f"bgm_{key}.ecsv"

    if cache_path.exists():
        return Table.read(cache_path, format="ascii.ecsv")

    # Submit query (this can take minutes; astroquery polls + downloads the result)
    #tbl = Besancon.query(
    #    glon=l_deg,
    #    glat=b_deg,
    #    area=area_deg2,
    #    smallfield=True,
    #    email=email,
    #    retrieve_file=retrieve_file,
    #    **besancon_kwargs,
    #)
    tbl = besancon_query_with_debug(
        glon=l_deg,
        glat=b_deg,
        area=area_deg2,
        smallfield=True,
        email=email,
        retrieve_file=retrieve_file,
        **besancon_kwargs,
    )

    # Cache for re-use
    tbl.write(cache_path, format="ascii.ecsv", overwrite=True)
    return tbl

def besancon_query_with_debug(**kwargs):
    """
    Run Besancon query, and if astroquery fails to parse, print the raw server
    response so we can see what changed (login page, maintenance, etc.).
    """
    try:
        return Besancon.query(**kwargs)
    except Exception as e:
        # Try to get the raw response via the async interface
        try:
            resp = Besancon.query_async(**kwargs)
            text = resp.text if hasattr(resp, "text") else str(resp)
            print("\n--- Besancon RAW RESPONSE (first 2000 chars) ---\n")
            print(text[:2000])
            print("\n--- end raw response ---\n")
        except Exception as e2:
            print("\nAlso failed to retrieve raw response via query_async:", repr(e2))
        raise  # re-raise original error

def count_stars_in_healpix(
    hpq: HealpixQuery,
    email: str,
    mag_band: str | None = None,   # e.g. "V", "G", "K" (depends on chosen photometric system)
    mag_max: float | None = None,  # e.g. 20.0
    **kwargs,
) -> int:
    """
    Get Besançon catalog for the pixel and count stars, optionally applying a mag cut.
    """
    tbl = query_besancon_for_healpix(hpq, email=email, **kwargs)
    #tbl = besancon_query_with_debug(hpq, email=email, **kwargs)

    if mag_band is None or mag_max is None:
        return len(tbl)

    if mag_band not in tbl.colnames:
        raise KeyError(
            f"Requested mag band '{mag_band}' not in returned table columns: {tbl.colnames}"
        )

    return int(np.count_nonzero(np.isfinite(tbl[mag_band]) & (tbl[mag_band] <= mag_max)))


# ---- Example usage ----
if __name__ == "__main__":
    email = 'zbalog@uni-heidelberg.de'
    hpq = HealpixQuery(nside=128, ipix=123456, nest=False, frame="galactic")

    n_all = count_stars_in_healpix(hpq, email=email)
    n_v20 = count_stars_in_healpix(hpq, email=email, mag_band="V", mag_max=20.0)

    print("Total simulated stars in pixel-area field:", n_all)
    print("Stars with V<=20:", n_v20)

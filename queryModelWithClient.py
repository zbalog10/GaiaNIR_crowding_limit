from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import random
import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError

import numpy as np
import healpy as hp
import xml.etree.ElementTree as ET

import json
import hashlib
from pathlib import Path
from typing import Any

from astropy.table import Table
#from dotenv import load_dotenv
import dotenv
dotenv.load_dotenv()

XML_NS = {
    "uws": "http://www.ivoa.net/xml/UWS/v1.0",
    "xlink": "http://www.w3.org/1999/xlink",
}

def canonicalize_params(params: dict[str, Any]) -> dict[str, str]:
    """
    Convert params to a stable {str: str} dict (what will actually be sent to the server).
    """
    out: dict[str, str] = {}
    for k, v in params.items():
        if v is None:
            continue
        out[str(k)] = str(v).strip()
    return out


def make_cache_key(
    *,
    url: str,
    nside: int,
    ipix: int,
    nest: bool,
    frame: str,
    params: dict[str, Any],
) -> str:
    """
    Stable key for "same request => same cached result".
    """
    payload = {
        "service_url": url.rstrip("/"),
        "healpix": {"nside": nside, "ipix": ipix, "nest": nest, "frame": frame},
        "params": canonicalize_params(params),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    """
    Returns (table_path, meta_path).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"bgm_{key}.ecsv", cache_dir / f"bgm_{key}.json"


def _xml_text(elem: Optional[ET.Element]) -> str:
    return "" if elem is None or elem.text is None else elem.text.strip()


def _find_phase(job_xml: ET.Element) -> str:
    return _xml_text(job_xml.find("uws:phase", XML_NS)).upper()


def _job_id(job_xml: ET.Element) -> int:
    return int(_xml_text(job_xml.find("uws:jobId", XML_NS)))


def _list_results(job_xml: ET.Element) -> list[tuple[str, str]]:
    """
    Returns list of (result_id, href_url).
    """
    out = []
    results = job_xml.find("uws:results", XML_NS)
    if results is None:
        return out
    for res in results.findall("uws:result", XML_NS):
        rid = res.get("id")
        href = res.get(f"{{{XML_NS['xlink']}}}href")
        if rid and href:
            out.append((rid, href))
    return out


@dataclass
class BGMUWS:
    """
    Minimal programmatic wrapper around the official Besançon UWS workflow.
    """
    url: str = "https://model.obs-besancon.fr/ws"
    user: str = ""
    password: str = ""
    timeout_s: int = 60
    min_delay_s: float = 0.5  # base politeness between requests
    max_delay_s: float = 30.0  # cap for backoff
    max_retries: int = 8  # total retries on transient failure

    def __post_init__(self):
        self._sess = requests.Session()
        self._sess.headers.update({"User-Agent": "GaiaNIR-BGMUWS/0.1"})
        self._auth = (self.user, self.password)

    def _get_xml(self, url: str) -> ET.Element:
        r = self._request("GET", url)
        r.raise_for_status()
        return ET.fromstring(r.text)
        #r = self._sess.get(url, auth=self._auth, timeout=self.timeout_s)

        #return ET.fromstring(r.text)

    def create_job(self) -> ET.Element:
        #r = self._sess.post(f"{self.url}/jobs", auth=self._auth, timeout=self.timeout_s)
        r = self._request("POST", f"{self.url}/jobs")
        r.raise_for_status()
        # Many UWS services return XML immediately; if not, follow Location.
        try:
            return ET.fromstring(r.text)
        except ET.ParseError:
            loc = r.headers.get("Location")
            if not loc:
                raise RuntimeError(f"Could not parse create_job response and no Location header.\n{r.text[:1000]}")
            return self._get_xml(loc)

    def set_params(self, job_id: int, params: dict[str, str]) -> ET.Element:
        r = self._sess.post(
            f"{self.url}/jobs/{job_id}/parameters",
            auth=self._auth,
            data=params,
            timeout=self.timeout_s,
        )
        r = self._sess.post(
            f"{self.url}/jobs/{job_id}/parameters",
            auth=self._auth,
            data=params,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return ET.fromstring(r.text)

    def run_job(self, job_id: int) -> ET.Element:
        r = self._sess.post(
            f"{self.url}/jobs/{job_id}/phase",
            auth=self._auth,
            data={"PHASE": "RUN"},
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return ET.fromstring(r.text)

    def get_job(self, job_id: int) -> ET.Element:
        return self._get_xml(f"{self.url}/jobs/{job_id}")

    def wait(self, job_id: int, poll_s: float = 5.0, max_wait_s: int = 1800) -> ET.Element:
        t0 = time.time()
        while True:
            job = self.get_job(job_id)
            phase = _find_phase(job)
            if phase in {"COMPLETED", "ERROR", "ABORTED"}:
                return job
            if time.time() - t0 > max_wait_s:
                raise TimeoutError(f"Job {job_id} did not finish in {max_wait_s}s (last phase={phase}).")
            time.sleep(poll_s)

    def download_result(self, href: str, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        r = self._sess.get(href, auth=self._auth, timeout=self.timeout_s)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return out_path

    def _sleep_polite(self, delay: float | None = None) -> None:
        time.sleep(self.min_delay_s if delay is None else delay)

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Robust request with exponential backoff + jitter on transient failures.
        Retries on: ConnectionError, Timeout, and HTTP 429/5xx.
        """
        last_exc = None
        for attempt in range(self.max_retries + 1):
            # A small baseline pause to avoid rapid-fire bursts
            self._sleep_polite()

            try:
                r = self._sess.request(
                    method, url,
                    auth=self._auth,
                    timeout=self.timeout_s,
                    **kwargs
                )

                # Retry on rate limit / transient server errors
                if r.status_code in (429, 500, 502, 503, 504):
                    # honor Retry-After if present
                    ra = r.headers.get("Retry-After")
                    if ra is not None:
                        try:
                            wait_s = float(ra)
                        except ValueError:
                            wait_s = None
                    else:
                        wait_s = None

                    if attempt >= self.max_retries:
                        r.raise_for_status()

                    # exponential backoff with jitter
                    backoff = min(self.max_delay_s, (2 ** attempt) + random.random())
                    self._sleep_polite(wait_s if wait_s is not None else backoff)
                    continue

                r.raise_for_status()
                return r

            except (ConnectionError, Timeout) as e:
                last_exc = e
                if attempt >= self.max_retries:
                    raise
                backoff = min(self.max_delay_s, (2 ** attempt) + random.random())
                self._sleep_polite(backoff)
                continue

            except HTTPError as e:
                # Non-retriable HTTP errors fall through immediately
                raise

        # Should not reach here
        raise last_exc if last_exc else RuntimeError("Request failed without exception.")

@dataclass(frozen=True)
class HealpixQuery:
    nside: int
    ipix: int
    nest: bool = False
    frame: str = "galactic"  # this UWS example assumes galactic coords for Coor1/Coor2

    def lonlat_corners_deg(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns arrays (lon_deg, lat_deg) for pixel corners.
        """
        # boundaries returns 3xN vectors in Cartesian for the polygon vertices
        vec = hp.boundaries(self.nside, self.ipix, step=1, nest=self.nest)  # shape (3, Nv)
        lon, lat = hp.vec2ang(vec.T, lonlat=True)  # lon/lat in degrees
        return np.asarray(lon), np.asarray(lat)


def _lon_bounds_handle_wrap(lon_deg: np.ndarray) -> tuple[float, float]:
    """
    Compute lon min/max robustly when the polygon crosses 0/360.
    Strategy: compute bounds in both [-180, 180] and [0, 360] and pick the smaller span.
    """
    lon0 = (lon_deg % 360.0)
    span0 = lon0.max() - lon0.min()

    lon180 = ((lon0 + 180.0) % 360.0) - 180.0
    span180 = lon180.max() - lon180.min()

    if span180 < span0:
        lo, hi = float(lon180.min()), float(lon180.max())
        # convert back to [0,360) style bounds for service if needed:
        lo = (lo + 360.0) % 360.0
        hi = (hi + 360.0) % 360.0
        # NOTE: if this crosses 0, the service might not accept wrap ranges; in that case split into two queries.
        return lo, hi

    return float(lon0.min()), float(lon0.max())

def count_stars_in_healpix_bgm(
    hpq: HealpixQuery,
    bgm: BGMUWS,
    cache_dir: Path = Path(".bgm_ws_cache"),
    extra_params: dict[str, str] | None = None,
    mag_col: str | None = None,
    mag_max: float | None = None,
    force: bool = False,          # <--- add this
) -> int:
    cache_dir = Path(cache_dir)
    # --- build bounds (your existing code) ---
    lon, lat = hpq.lonlat_corners_deg()
    lon_min, lon_max = _lon_bounds_handle_wrap(lon)
    lat_min, lat_max = float(lat.min()), float(lat.max())

    # --- base params (what you actually send to server) ---
    params = {
        "Coor1_min": str(lon_min),
        "Coor1_max": str(lon_max),
        "Coor2_min": str(lat_min),
        "Coor2_max": str(lat_max),
        "SOLI": "1",
        "ref_filter": "V",
        "band_max": "20.,99.,99.,99.,99.,99.,99.,99.,99.",
    }
    if extra_params:
        params.update(extra_params)

    # --- CACHE KEY ---
    key = make_cache_key(
        url=bgm.url,
        nside=hpq.nside,
        ipix=hpq.ipix,
        nest=hpq.nest,
        frame=hpq.frame,
        params=params,
    )
    table_path, meta_path = cache_paths(cache_dir, key)

    # --- CACHE HIT ---
    if (not force) and table_path.exists():
        tbl = Table.read(table_path, format="ascii.ecsv")
        return _count_from_table(tbl, mag_col, mag_max)

    # --- CACHE MISS: run job ---
    job_xml = bgm.create_job()
    job_id = _job_id(job_xml)

    bgm.set_params(job_id, canonicalize_params(params))
    bgm.run_job(job_id)

    job_done = bgm.wait(job_id, poll_s=5.0, max_wait_s=3600)
    phase = _find_phase(job_done)
    if phase != "COMPLETED":
        err = job_done.find("uws:errorSummary", XML_NS)
        msg = ""
        if err is not None:
            m = err.find("uws:message", XML_NS)
            msg = _xml_text(m)
        raise RuntimeError(f"BGM job {job_id} ended in phase={phase}. {msg}")

    results = _list_results(job_done)
    if not results:
        raise RuntimeError(f"BGM job {job_id} completed but returned no results.")

    rid, href = results[0]
    raw_path = cache_dir / f"raw_{key}_{rid}"  # raw download file
    raw_path = bgm.download_result(href, raw_path)

    # Read the raw download robustly (use your sniffing code if needed)
    #tbl = read_bgm_result_table(raw_path)  # <-- implement with sniff_table_format/gzip logic
    tbl = Table.read(raw_path, format='ascii')

    # Save canonical cached table
    tbl.write(table_path, format="ascii.ecsv", overwrite=True)

    # Save metadata for traceability
    meta = {
        "cache_key": key,
        "job_id": job_id,
        "result_id": rid,
        "result_href": href,
        "service_url": bgm.url,
        "healpix": {"nside": hpq.nside, "ipix": hpq.ipix, "nest": hpq.nest, "frame": hpq.frame},
        "params": canonicalize_params(params),
        "raw_file": str(raw_path),
        "table_file": str(table_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    return _count_from_table(tbl, mag_col, mag_max)


def _count_from_table(tbl: Table, mag_col: str | None, mag_max: float | None) -> int:
    import numpy as np
    if mag_col is None or mag_max is None:
        return len(tbl)
    if mag_col not in tbl.colnames:
        raise KeyError(f"{mag_col=} not found. Available columns: {tbl.colnames}")
    m = np.asarray(tbl[mag_col])
    return int(np.count_nonzero(np.isfinite(m) & (m <= mag_max)))

def count_stars_in_healpix_bgm_orig(
    hpq: HealpixQuery,
    bgm: BGMUWS,
    cache_dir: Path = Path(".bgm_ws_cache"),
    extra_params: Optional[dict[str, str]] = None,
    mag_col: Optional[str] = None,
    mag_max: Optional[float] = None,
) -> int:
    """
    Submit a BGM job for a bounding box around the HEALPix pixel and count returned rows.
    """
    if hpq.frame.lower() != "galactic":
        raise ValueError("This simple version assumes Coor1/Coor2 are Galactic lon/lat. Convert beforehand if needed.")

    lon, lat = hpq.lonlat_corners_deg()
    lon_min, lon_max = _lon_bounds_handle_wrap(lon)
    lat_min, lat_max = float(lat.min()), float(lat.max())

    # Create job
    job_xml = bgm.create_job()
    job_id = _job_id(job_xml)

    # Minimal parameter set based on the example in the official client header.
    params = {
        "Coor1_min": str(lon_min),
        "Coor1_max": str(lon_max),
        "Coor2_min": str(lat_min),
        "Coor2_max": str(lat_max),
        "SOLI": "1",
        "ref_filter": "V",
        # Example: band_max expects a comma-separated list; keep as in their usage
        "band_max": "20.,99.,99.,99.,99.,99.,99.,99.,99.",
    }
    if extra_params:
        params.update(extra_params)

    bgm.set_params(job_id, params)
    bgm.run_job(job_id)

    job_done = bgm.wait(job_id, poll_s=5.0, max_wait_s=3600)
    phase = _find_phase(job_done)

    if phase != "COMPLETED":
        # Try to surface error messages if present
        err = job_done.find("uws:errorSummary", XML_NS)
        msg = ""
        if err is not None:
            m = err.find("uws:message", XML_NS)
            msg = _xml_text(m)
        raise RuntimeError(f"BGM job {job_id} ended in phase={phase}. {msg}")

    results = _list_results(job_done)
    if not results:
        raise RuntimeError(f"BGM job {job_id} completed but returned no results.")

    # Download first result
    rid, href = results[0]
    out = cache_dir / f"job{job_id}_{rid}"
    bgm.download_result(href, out)

    # Let astropy infer format (VOTable/CSV/etc.)
    tbl = Table.read(out,format='ascii')

    if mag_col is None or mag_max is None:
        return len(tbl)

    if mag_col not in tbl.colnames:
        raise KeyError(f"{mag_col=} not found. Available columns: {tbl.colnames}")

    m = np.asarray(tbl[mag_col])
    return int(np.count_nonzero(np.isfinite(m) & (m <= mag_max)))


# Example usage:
if __name__ == "__main__":
    bgm = BGMUWS(
        url="https://model.obs-besancon.fr/ws",
        user=os.getenv("BGM_USER"),
        password=os.getenv("BGM_PASS"),
    )
    hpq = HealpixQuery(nside=128, ipix=123457, nest=False, frame="galactic")
    n = count_stars_in_healpix_bgm(hpq, bgm)
    print("Stars (bounding box around pixel):", n)


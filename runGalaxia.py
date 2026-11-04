from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, Union, Optional


def _format_value(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return f"{v:.10g}"
    return str(v)


def write_galaxia_param_file(
    path: Union[str, Path],
    params: Dict[str, object],
) -> Path:
    """
    Write a Galaxia v0.2 parameter file (fixed-width style).
    """
    path = Path(path)
    lines = []
    for k, v in params.items():
        lines.append(f"{k:<35} {_format_value(v)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

def run_galaxia(
    workdir,
    output_file_stem,
    *,
    l_deg,
    b_deg,
    survey_area_deg2,
    vmag_min=0.0,
    vmag_max=20.0,
    photo_sys="DCMC",
    magcolor_names="Ks,J-Ks",
    seed=12,
    f_sample=1.0,
    galaxia_exe=str(Path.home() / ".local/bin/galaxia"),
    extra_params=None,
    env=None,
):
    from pathlib import Path
    import os
    import subprocess

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    params = {
        "outputFile": output_file_stem,
        "outputDir": str(workdir) + "/",
        "photoSys": photo_sys,
        "magcolorNames": magcolor_names,
        "appMagLimits[0]": vmag_min,
        "appMagLimits[1]": vmag_max,
        "absMagLimits[0]": -1000,
        "absMagLimits[1]": 1000,
        "colorLimits[0]": -1000,
        "colorLimits[1]": 1000,
        "geometryOption": 1,
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

    print(f"\n--- Running Galaxia: {output_file_stem} ---")
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

    stdout_lines = []

    for line in proc.stdout:
        print(line, end="")       # live output
        stdout_lines.append(line)

    proc.wait()
    stdout_text = "".join(stdout_lines)

    ebf_path = workdir / f"{output_file_stem}.ebf"

    if proc.returncode != 0 and not ebf_path.exists():
        raise RuntimeError(
            "Galaxia failed and did not produce output.\n"
            f"Exit code: {proc.returncode}\n"
            "---- Galaxia output ----\n"
            f"{stdout_text}"
        )

    if ebf_path.exists():
        print(f"\n--- Galaxia finished, output written to {ebf_path} ---")
        return ebf_path

    raise FileNotFoundError(
        "Galaxia finished but no .ebf file was found.\n"
        f"Galaxia output:\n{stdout_text}"
    )

def run_galaxia_old(
    workdir: Union[str, Path],
    output_file_stem: str,
    *,
    l_deg: float,
    b_deg: float,
    survey_area_deg2: float,
    vmag_min: float = 0.0,
    vmag_max: float = 20.0,
    photo_sys: str = "DCMC",
    magcolor_names: str = "K,J-K",
    seed: int = 12,
    f_sample: float = 1.0,
    galaxia_exe: str = str(Path.home() / ".local/bin/galaxia"),
    extra_params: Optional[Dict[str, object]] = None,
    env: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Run Galaxia v0.2 using the required '-r' switch.

    Returns
    -------
    Path to the produced .ebf file.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    params = {
        "outputFile": output_file_stem,
        "outputDir": str(workdir) + "/",
        "photoSys": photo_sys,
        "magcolorNames": magcolor_names,
        "appMagLimits[0]": vmag_min,
        "appMagLimits[1]": vmag_max,
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

    # ---- THE IMPORTANT BIT: -r switch ----
    proc = subprocess.run(
        [galaxia_exe, "-r", str(param_path)],
        cwd=str(workdir),
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    ebf_path = workdir / f"{output_file_stem}.ebf"

    # If Galaxia produced the file, accept it as success even if return code is non-zero.
    if proc.returncode != 0 and not ebf_path.exists():
        raise RuntimeError(
            "Galaxia failed and did not produce an output .ebf.\n"
            f"Command: {galaxia_exe} -r {param_path}\n"
            f"Exit code: {proc.returncode}\n"
            "---- Galaxia output ----\n"
            f"{proc.stdout}"
        )

    if ebf_path.exists():
        return ebf_path
    #if proc.returncode != 0:
    #    raise RuntimeError(
    #        "Galaxia failed.\n"
    #        f"Command: {galaxia_exe} -r {param_path}\n"
    #        f"Exit code: {proc.returncode}\n"
    #        "---- Galaxia output ----\n"
    #        f"{proc.stdout}"
    #    )

    ebf_path = workdir / f"{output_file_stem}.ebf"
    if ebf_path.exists():
        return ebf_path

    # Fallback: try matching files
    matches = sorted(workdir.glob(f"{output_file_stem}*.ebf"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        "Galaxia finished without error, but no .ebf file was found.\n"
        f"Galaxia output:\n{proc.stdout}"
    )


# ---- Minimal example ----
if __name__ == "__main__":
    ebf = run_galaxia(
        workdir=str(Path.home() / "galaxia_runs/pole_test"),
        output_file_stem="pole_K20_area2",
        l_deg=0.0,
        b_deg=90.0,
        survey_area_deg2=0.2,
        vmag_max=20.0,
    )
    print("EBF written to:", ebf)

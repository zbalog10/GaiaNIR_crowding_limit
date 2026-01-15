import numpy as np
import ebf

fn = "/home/zbalog/galaxia_test/galaxy1.ebf"     # <-- change
survey_area_deg2 = 1000.0     # <-- set to your surveyArea from params

V = ebf.read(fn, "/ubv_v")  # apparent V magnitudes

# Basic counts
N = V.size
density_deg2 = N / survey_area_deg2
density_arcsec2 = density_deg2 / (3600.0**2)

print(f"N stars: {N:,}")
print(f"Surface density: {density_deg2:,.2f} stars/deg^2")
print(f"Surface density: {density_arcsec2:.3e} stars/arcsec^2")

# Histogram (e.g. 0–25 mag in 0.5 mag bins)
bins = np.arange(0, 25.0 + 0.5, 0.5)
hist, edges = np.histogram(V, bins=bins)

# Convert to densities per deg^2 per mag-bin
bin_width = np.diff(edges)
dens_per_deg2 = hist / survey_area_deg2
dens_per_deg2_per_mag = dens_per_deg2 / bin_width

# Print a compact table for non-empty bins
print("\nVmag_bin_center  N_in_bin  dens[deg^-2]  dens[deg^-2 mag^-1]")
centers = 0.5 * (edges[:-1] + edges[1:])
for c, n, d, dm in zip(centers, hist, dens_per_deg2, dens_per_deg2_per_mag):
    if n > 0:
        print(f"{c:6.2f}          {n:7d}   {d:10.2f}      {dm:10.2f}")
import ebf

fn = "/home/balog/galaxia_runs/healpix_n128_plane_10x10_Ks/hp_093477/hp093477_n128_Ks22.ebf"
#fn = "/home/balog/galaxia_runs/healpix_n128_plane_100x100_Ks/hp_096036/hp096036_n128_Ks22.ebf"
root = ebf.read(fn)

print("Top-level keys:")
for k in list(root.keys())[:30]:
    print(" ", k)
print("Total keys:", len(root))
#print(root.min(), root.max(), root.size)
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 

def read_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype('float32')
        transform = src.transform
    return arr, transform

def calc_ndwi(green, nir):
    ndwi = (green - nir) / (green + nir + 1e-8)
    return ndwi

def water_mask_from_ndwi(ndwi, threshold = 0.2):
    return ndwi > threshold

def area_from_mask(mask, pixel_area_m2 = 100.0):
    return mask.sum() * pixel_area_m2

def save_mask_image(mask, out_path):
    plt.figure(figsize=(4,4))
    plt.imshow(mask, cmap='Blues')
    plt.axis('off')
    Path(out_path).parent.mkdir(parents = True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def foam_heuristic(red, green, blue, nir, water_mask, bright_thr=0.45, nir_thr=0.15):
    red_norm = red/(red.max() + 1e-8)
    green_norm = green/(green.max() + 1e-8)
    blue_norm = blue/(blue.max() + 1e-8)
    nir_norm = nir/(nir.max() + 1e-8)

    vis = (red_norm + green_norm + blue_norm)/3.0
    foam_mask = (water_mask) & (vis > bright_thr) & (nir_norm < nir_thr)
    foam_fraction = foam_mask.sum()/(water_mask.sum() + 1e-8)
    return foam_mask, foam_fraction


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python process_tile.py <green_band.tif> <nir_band.tif> [<output_prefix>]")
        sys.exit(1)

    green_path = sys.argv[1]
    nir_path = sys.argv[2]
    out_prefix = sys.argv[3] if len(sys.argv) > 3 else "out/test"

    green, _ = read_band(green_path)
    nir, _ = read_band(nir_path)

    ndwi = calc_ndwi(green, nir)

    water_mask = water_mask_from_ndwi(ndwi, threshold = 0.2)
    
    area = area_from_mask(water_mask)
    print(f"Detected water area: {area:.2f} m^2")

    save_mask_image(water_mask, f"{out_prefix}_water_mask.png")
    print(f"Saved mask image to {out_prefix}_water_mask.png")

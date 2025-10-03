import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype('float32')
    return arr

def calc_ndwi(green, nir):
    return (green - nir) / (green + nir + 1e-8)

def water_mask_from_ndwi(ndwi, threshold=0.2):
    return ndwi > threshold

def foam_heuristic(red, green, blue, nir, water_mask, bright_thr=0.45, nir_thr=0.15):
    red_norm = red / (red.max() + 1e-8)
    green_norm = green / (green.max() + 1e-8)
    blue_norm = blue / (blue.max() + 1e-8)
    nir_norm = nir / (nir.max() + 1e-8)
    
    vis = (red_norm + green_norm + blue_norm) / 3.0
    foam_mask = (water_mask) & (vis > bright_thr) & (nir_norm < nir_thr)
    foam_fraction = foam_mask.sum() / (water_mask.sum() + 1e-8)
    return foam_mask, foam_fraction

def save_mask_image(mask, out_path, cmap='Blues'):
    plt.figure(figsize=(4,4))
    plt.imshow(mask, cmap=cmap)
    plt.axis('off')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def analyze_tile(red_path, green_path, blue_path, nir_path, out_prefix="out/test"):
    red = read_band(red_path)
    green = read_band(green_path)
    blue = read_band(blue_path)
    nir = read_band(nir_path)
    
    ndwi = calc_ndwi(green, nir)
    water_mask = water_mask_from_ndwi(ndwi, threshold=0.2)
    water_area = water_mask.sum() * 100  # m^2, assume 100m² per pixel
    
    foam_mask, foam_fraction = foam_heuristic(red, green, blue, nir, water_mask)
    
    save_mask_image(water_mask, f"{out_prefix}_water_mask.png", cmap='Blues')
    save_mask_image(foam_mask, f"{out_prefix}_foam_mask.png", cmap='Reds')
    
    return {
        "water_area_m2": water_area,
        "foam_fraction": foam_fraction,
        "water_mask_image": f"{out_prefix}_water_mask.png",
        "foam_mask_image": f"{out_prefix}_foam_mask.png"
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 6:
        print("Usage: python analyze_tile.py <red.tif> <green.tif> <blue.tif> <nir.tif> <output_prefix>")
        sys.exit(1)
    red, green, blue, nir, out_prefix = sys.argv[1:]
    result = analyze_tile(red, green, blue, nir, out_prefix)
    print(result)

import rasterio
import matplotlib.pyplot as plt
from process_tile import calc_ndwi, water_mask_from_ndwi

with rasterio.open("data/lake1/B03.tif") as src:
    green = src.read(1).astype("float32")
with rasterio.open("data/lake1/B08.tif") as src:
    nir = src.read(1).astype("float32")

ndwi  = calc_ndwi(green, nir)
water_mask = water_mask_from_ndwi(ndwi, threshold =0.2)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(green, cmap = "Greens")
plt.title("Green Band (B03)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(nir, cmap = "Reds")
plt.title("NIR Band (B08)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(water_mask, cmap="Blues")
plt.title("Detected Water Mask")
plt.axis("off")

plt.tight_layout()
plt.show()
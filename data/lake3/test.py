import numpy as np
import rasterio
import cv2
import matplotlib.pyplot as plt

def read_band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)

def make_rgb(red, green, blue, max_side=1024):
    # Stack into 3-channel
    rgb = np.dstack([red, green, blue])
    # Normalize robustly (ignore outliers)
    lo, hi = np.percentile(rgb[~np.isnan(rgb)], (2, 98))
    rgb_norm = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)

    # Resize for faster viewing
    h, w, _ = rgb_norm.shape
    scale = max_side / max(h, w) if max(h, w) > max_side else 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    rgb_resized = cv2.resize(rgb_norm, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return rgb_resized

def save_rgb_image(red_path, green_path, blue_path, out_path):
    red = read_band(red_path)
    green = read_band(green_path)
    blue = read_band(blue_path)

    rgb_img = make_rgb(red, green, blue)
    
    plt.figure(figsize=(8,8))
    plt.imshow(rgb_img)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved true-color image to {out_path}")


# Example usage
save_rgb_image("data/lake3/B04.jp2", "data/lake3/B03.jp2", "data/lake3/B02.jp2", "data/lake3/truecolor.png")

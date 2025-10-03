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

def area_from_mask(mask, pixel_area_m2=100.0):
    return mask.sum() * pixel_area_m2

def foam_heuristic(red, green, blue, nir, water_mask, bright_thr=0.4, nir_thr=0.2):
    red_norm = red / (red.max() + 1e-8)
    green_norm = green / (green.max() + 1e-8)
    blue_norm = blue / (blue.max() + 1e-8)
    nir_norm = nir / (nir.max() + 1e-8)

    vis = (red_norm + green_norm + blue_norm)/3.0
    foam_mask = (water_mask) & (vis > bright_thr) & (nir_norm < nir_thr)
    foam_fraction = foam_mask.sum() / (water_mask.sum() + 1e-8)
    return foam_mask, foam_fraction

def save_mask_image(mask, out_path, cmap='Blues'):
    plt.figure(figsize=(6,6))
    plt.imshow(mask.astype(np.float32), cmap=cmap, vmin=0, vmax=1)
    plt.axis('off')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# def save_combined_figure(green, nir, water_mask, foam_mask, out_path):
#     plt.figure(figsize=(12,4))

#     plt.subplot(1,4,1)
#     plt.imshow(green, cmap="Greens")
#     plt.title("Green Band (B03)")
#     plt.axis("off")

#     plt.subplot(1,4,2)
#     plt.imshow(nir, cmap="Reds")
#     plt.title("NIR Band (B08)")
#     plt.axis("off")

#     plt.subplot(1,4,3)
#     plt.imshow(water_mask.astype(np.float32), cmap="Blues", vmin=0, vmax=1)
#     plt.title("Water Mask")
#     plt.axis("off")

#     plt.subplot(1,4,4)
#     plt.imshow(foam_mask.astype(np.float32), cmap="Reds", vmin=0, vmax=1)
#     plt.title("Foam Mask")
#     plt.axis("off")

#     plt.tight_layout()
#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
#     plt.close()

# thumbnail + safe combined-figure saver (drop-in replacement)


def _downsample_numpy(a: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """
    Fast fallback downsample by simple block-averaging-ish using slicing.
    Not perfect, but very fast and dependency-free.
    """
    h, w = a.shape
    if new_h >= h and new_w >= w:
        # no resize needed
        return a.astype(np.float32)
    # compute integer step sizes
    step_y = max(1, int(np.ceil(h / new_h)))
    step_x = max(1, int(np.ceil(w / new_w)))
    # take a grid sample (fast)
    sampled = a[::step_y, ::step_x]
    # If sampled is still larger than desired (due to ceil), crop
    return sampled[:new_h, :new_w].astype(np.float32)

def make_thumbnail(arr, max_side=512, normalize=True):
    """Make a small thumbnail (uint8) from a big 2D array for visualization."""
    a = np.array(arr, dtype=np.float32)
    # handle NaNs
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    # robust normalization to 0..1
    if normalize:
        # use percentiles to ignore outliers
        flat = a.ravel()
        if flat.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo = np.percentile(flat, 1)
            hi = np.percentile(flat, 99)
            if hi - lo == 0:
                lo, hi = a.min(), a.max()
        if hi - lo <= 0:
            a_norm = np.clip((a - a.min()) / (a.max() - a.min() + 1e-8), 0.0, 1.0)
        else:
            a_norm = (a - lo) / (hi - lo)
            a_norm = np.clip(a_norm, 0.0, 1.0)
        a = a_norm
    else:
        # assume already 0..1 or 0/1 mask, scale to 0..1 anyway
        amin, amax = float(a.min()), float(a.max())
        if amax - amin > 1e-8:
            a = (a - amin) / (amax - amin)
        else:
            a = np.clip(a, 0.0, 1.0)

    h, w = a.shape
    scale = max_side / max(h, w) if max(h, w) > max_side else 1.0
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

    # Try OpenCV resize if available (faster & better interpolation)
    try:
        import cv2
        thumb = cv2.resize(a.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except Exception:
        # fallback: very fast numpy subsample/approximate resize
        thumb = _downsample_numpy(a, new_w, new_h)

    # convert to uint8 for visualization
    thumb_u8 = (np.clip(thumb, 0.0, 1.0) * 255).astype(np.uint8)
    return thumb_u8

def save_combined_figure_safe(green, nir, water_mask, foam_mask, out_path, thumb_size=512):
    """
    Create small thumbnails and assemble a combined image robustly.
    green, nir: 2D arrays (float)
    water_mask, foam_mask: boolean or 0/1 arrays
    out_path: path to save combined figure (PNG)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # prepare arrays (ensure 2D)
    green = np.array(green, dtype=np.float32)
    nir = np.array(nir, dtype=np.float32)
    water_mask = np.array(water_mask, dtype=np.float32)
    foam_mask = np.array(foam_mask, dtype=np.float32)

    # create thumbnails
    tg = make_thumbnail(green, max_side=thumb_size, normalize=True)
    tn = make_thumbnail(nir, max_side=thumb_size, normalize=True)
    tw = make_thumbnail(water_mask, max_side=thumb_size, normalize=False)  # mask 0/1
    tf = make_thumbnail(foam_mask, max_side=thumb_size, normalize=False)

    try:
        # Assemble with matplotlib (small images -> fast)
        fig, axes = plt.subplots(1, 4, figsize=(12, 4))
        axes[0].imshow(tg, cmap="Greens")
        axes[0].set_title("Green (thumb)")
        axes[1].imshow(tn, cmap="Reds")
        axes[1].set_title("NIR (thumb)")
        axes[2].imshow(tw, cmap="Blues", vmin=0, vmax=255)
        axes[2].set_title("Water Mask")
        axes[3].imshow(tf, cmap="Reds", vmin=0, vmax=255)
        axes[3].set_title("Foam Mask")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    except Exception as e:
        # Very safe fallback: save thumbnails individually with imageio
        try:
            import imageio
            imageio.imwrite(str(out_path.with_suffix(".green_thumb.png")), tg)
            imageio.imwrite(str(out_path.with_suffix(".nir_thumb.png")), tn)
            imageio.imwrite(str(out_path.with_suffix(".water_thumb.png")), tw)
            imageio.imwrite(str(out_path.with_suffix(".foam_thumb.png")), tf)
        except Exception as e2:
            # Last resort: write numpy arrays to .npy so you can inspect them
            np.save(out_path.with_suffix(".green_thumb.npy"), tg)
            np.save(out_path.with_suffix(".nir_thumb.npy"), tn)
            np.save(out_path.with_suffix(".water_thumb.npy"), tw)
            np.save(out_path.with_suffix(".foam_thumb.npy"), tf)
        # re-raise a clear error so you know plotting failed but fallback wrote files
        raise RuntimeError(f"Failed to create combined figure: {e}. Thumbnails saved as separate files.") from e


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 6:
        print("Usage: python foam_detection.py <red.jp2> <green.jp2> <blue.jp2> <nir.jp2> <output_prefix>")
        sys.exit(1)

    red_path, green_path, blue_path, nir_path, out_prefix = sys.argv[1:]

    red = read_band(red_path)
    green = read_band(green_path)
    blue = read_band(blue_path)
    nir = read_band(nir_path)
    print(red.sum(), green.sum(), blue.sum(), nir.sum())

    ndwi = calc_ndwi(green, nir)
    water_mask = water_mask_from_ndwi(ndwi, threshold=0.2)
    water_area = area_from_mask(water_mask)
    print(f"Detected water area: {water_area:.2f} m²")

    foam_mask, foam_fraction = foam_heuristic(red, green, blue, nir, water_mask)
    print(f"Foam fraction over water: {foam_fraction*100:.2f}%")

    save_mask_image(water_mask, f"{out_prefix}_water_mask.png", cmap='Blues')
    save_mask_image(foam_mask, f"{out_prefix}_foam_mask.png", cmap='Reds')
    save_combined_figure_safe(green, nir, water_mask, foam_mask, f"{out_prefix}_combined.png")
    print(f"Saved water mask, foam mask, and combined image with prefix: {out_prefix}")

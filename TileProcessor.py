import json
import os
import shutil 
import boto3
import rasterio
import numpy as np
import datetime 
from pathlib import Path
import gc 

s3 = boto3.client("s3")

REQUIRED_BANDS = ["B02.jp2", "B03.jp2", "B04.jp2", "B08.jp2"]

def read_band_local(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float16)
        transform = src.transform
    return arr, transform

def calc_ndwi(green, nir):
    return (green - nir) / (green + nir + 1e-8)

def calc_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-8)

def water_mask_from_ndwi(ndwi, thr=0.2):
=    return ndwi > thr

def foam_heuristic(red, green, blue, nir, water_mask, bright_thr=0.4, nir_thr=0.2):
    def norm(a):
        ma = np.nanmax(a)
        return a / (ma + 1e-8)
        
    red_n = norm(red.astype(np.float16)); green_n = norm(green.astype(np.float16)); 
    blue_n = norm(blue.astype(np.float16)); nir_n = norm(nir.astype(np.float16))
    
    vis = (red_n + green_n + blue_n) / 3.0
    
    foam_mask = (water_mask) & (vis > bright_thr) & (nir_n < nir_thr)
    foam_fraction = foam_mask.sum() / (water_mask.sum() + 1e-8)
    
    del red_n, green_n, blue_n, nir_n, vis
    
    return foam_mask, float(foam_fraction)

def area_from_mask(mask, transform):
    px_w = abs(transform.a) # Pixel width
    px_h = abs(transform.e) # Pixel height
    pixel_area = px_w * px_h
    return mask.sum() * pixel_area

def upload_file(local_path, bucket, key):
    s3.upload_file(local_path, bucket, key)

def extract_s3_event(record_body):
    try:
        ev = json.loads(record_body)
        if "Records" in ev and ev["Records"][0].get("s3"):
            rec = ev["Records"][0]
            bucket = rec["s3"]["bucket"]["name"]
            key = rec["s3"]["object"]["key"]
            return bucket, key
    except Exception:
        pass
    try:
        ev2 = json.loads(json.loads(record_body))
        rec = ev2["Records"][0]
        return rec["s3"]["bucket"]["name"], rec["s3"]["object"]["key"]
    except Exception:
        raise ValueError("Unable to parse S3 event from SQS record body")

def find_r10m_prefix(key):
    idx = key.find("/R10m/")
    if idx == -1:
        return None
    return key[:idx+len("R10m/")]

def parse_tile_and_date_from_key(prefix):
    parts = [p for p in prefix.strip("/").split("/") if p]
    
    if len(parts) < 5: 
        return None, None, None, None

    day = parts[-2]
    month = parts[-3]
    year = parts[-4]
    tile = parts[-5]
    
    return tile, year, month, day

def lambda_handler(event, context):
    results = []
    local_tmp_dirs = []

    for record in event.get("Records", []):
        body = record.get("body")
        local_tmp = None
        
        try:
            bucket, key = extract_s3_event(body)
        except Exception as e:
            print("Skipping record, parse error:", e)
            continue

        print("Processing S3 object:", bucket, key)
        r10m_prefix = find_r10m_prefix(key)
        if not r10m_prefix:
            print("Object not in R10m path; skipping:", key)
            continue

        try:
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=r10m_prefix)
            keys = [obj['Key'].split("/")[-1] for obj in resp.get('Contents', [])]
        except Exception as e:
            print("Error listing objects for prefix:", r10m_prefix, e)
            continue

        if not all(band in keys for band in REQUIRED_BANDS):
            print(f"Not all required bands present under {r10m_prefix}. Found: {keys}")
            results.append({"status":"waiting", "prefix": r10m_prefix})
            continue

        tile, year, month, day = parse_tile_and_date_from_key(r10m_prefix)

        if not all([tile, year, month, day]):
            print(f"Failed to parse Tile/Date from: {r10m_prefix}. Using unknown tile and current time.")
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            tile = "unknown_tile"
            year = str(now_utc.year)
            month = str(now_utc.month).zfill(2)
            day = str(now_utc.day).zfill(2)

        date_str = f"{year}-{month}-{day}"

        local_tmp = f"/tmp/{tile}_{year}_{month}_{day}_{context.aws_request_id}"
        local_tmp_dirs.append(local_tmp)
        Path(local_tmp).mkdir(parents=True, exist_ok=True)

        local_paths = {}
        for band in REQUIRED_BANDS:
            s3_key = f"{r10m_prefix.rstrip('/')}/{band}" 
            local_path = os.path.join(local_tmp, band)
            try:
                s3.download_file(bucket, s3_key, local_path)
                local_paths[band] = local_path
            except Exception as e:
                print(f"Failed to download {s3_key} {e}") 
                local_paths[band] = None

        if any(v is None for v in local_paths.values()):
            print("Missing one of the downloaded bands, skipping")
            results.append({"status":"failed_download", "prefix": r10m_prefix})
            shutil.rmtree(local_tmp)
            continue
        
        blue, trans_b = read_band_local(local_paths["B02.jp2"])
        green, trans_g = read_band_local(local_paths["B03.jp2"])
        red, trans_r = read_band_local(local_paths["B04.jp2"])
        nir, transform = read_band_local(local_paths["B08.jp2"])
        del trans_b, trans_g, trans_r 
        
        ndwi = calc_ndwi(green, nir)
        ndvi = calc_ndvi(nir, red)
        water_mask = water_mask_from_ndwi(ndwi, thr=0.2)
        del ndwi 
        
        water_area_m2 = float(area_from_mask(water_mask, transform))

        foam_mask, foam_fraction = foam_heuristic(red, green, blue, nir, water_mask, bright_thr=0.4, nir_thr=0.2)
        
        del blue, green, red, nir, water_mask, foam_mask
        
        gc.collect() 
        
        ndvi_mean = float(np.nanmean(ndvi))
        ndvi_median = float(np.nanmedian(ndvi))
        veg_status = "sparse/degraded"
        if ndvi_mean < 0.2:
            veg_status = "sparse/degraded"
        elif ndvi_mean < 0.5:
            veg_status = "moderate"
        else:
            veg_status = "healthy"
        
        del ndvi

        out_dir = os.path.join(local_tmp, "out")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        
        res_prefix = f"results/{tile}/{year}/{month}/{day}/"
        try:
            metrics = {
                "tile": tile,
                "date": date_str,
                "water_area_m2": water_area_m2,
                "foam_fraction": foam_fraction,
                "ndvi_mean": ndvi_mean,
                "ndvi_median": ndvi_median,
                "vegetation_status": veg_status
            }
            metrics_local = os.path.join(out_dir, "metrics.json")
            with open(metrics_local, "w") as f:
                json.dump(metrics, f)
                
            upload_file(metrics_local, bucket, res_prefix + "metrics.json")

            print(f"Processed tile {tile} date {date_str}: water_area={water_area_m2:.2f} m2 foam={foam_fraction:.3f} ndvi_mean={ndvi_mean:.3f}")
            results.append({"status":"processed","tile":tile,"date":date_str,"metrics":metrics})
            
        except Exception as e:
            print("Upload failed", e)
            results.append({"status":"upload_failed","tile":tile,"date":date_str, "error": str(e)})
            continue

    for tmp_dir in local_tmp_dirs:
        try:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
                print(f"Cleaned up {tmp_dir}")
        except Exception as e:
            print(f"Failed to clean up {tmp_dir}: {e}")

    gc.collect() 
    
    return {"statusCode": 200, "results": results}

import os
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from pathlib import Path 
from analyze_tile import analyze_tile

app = FastAPI(title = "River Agent API (local)")

class AnalyzeRequest(BaseModel): 
    red: str
    green: str
    blue: str
    nir: str
    out_prefix: str = "out/result"

@app.post("/analyze-paths")
async def analyze_paths(req: AnalyzeRequest):
    try:
        result = analyze_tile(req.red, req.green, req.blue, req.nir, out_prefix = req.out_prefix)
        summary = local_summarize(result)
        return {"result": result, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-upload")
async def analyze_upload(
    red_file: UploadFile = File(...),
    green_file: UploadFile = File(...),
    blue_file: UploadFile = File(...),
    nir_file: UploadFile = File(...),
    out_prefix: str = Form("out/uploaded")
):
    tmpdir = Path("tmp_uploads")
    tmpdir.mkdir(parents = True, exist_ok = True)
    red_path = tmpdir/red_file.filename
    green_path = tmpdir/green_file.filename
    blue_path = tmpdir/blue_file.filename
    nir_path = tmpdir/nir_file.filename

    for upload, p in [(red_file, red_path), (green_file, green_path), (blue_file, blue_path), (nir_file, nir_path)]:
        with open(p, "wb") as f:
            f.write(await upload.read())
    
    try:
        result = analyze_tile(str(red_path), str(green_path), str(blue_path), str(nir_path), out_prefix)
        summary = local_summarize(result)
        return {"result": result, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        pass


def local_summarize(result: dict) -> str:
    area_km2 = result["water_area_m2"]/1e6
    foam_pct = result["foam_fraction"] * 100
    s = (
        f"Detected water area = {result['water_area_m2']:.0f} m^2"
        f"({area_km2:.3f} km^2)."
        f"Foam fraction = {foam_pct:.2f}%."
    )    

    if foam_pct > 5:
        s += "High foam detected - suggest manual review and notify authorities."
    elif foam_pct > 0.5:
        s += "Moderate foam detected - consider monitoring."
    else:
        s += "No significant foam detected."
    return s


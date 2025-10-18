#Lambda function being invoked: WaterMonitoringBedrockAnalyzer

import os
import json
import boto3
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from botocore.exceptions import ClientError
from decimal import Decimal

BUCKET_NAME = os.environ.get("BUCKET_NAME", "water-monitoring-project")
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"

s3_client = boto3.client("s3")  
bedrock_runtime = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

def json_default_serializer(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def create_analysis_prompt(tile_id, raw_metrics_data):
    data_sample = raw_metrics_data[-12:]
    data_string = json.dumps(data_sample, default=json_default_serializer, indent=2)
    prompt = (
        f"You are an expert geospatial and environmental analyst. Analyze the following "
        f"time-series satellite metric data for Sentinel-2 tile {tile_id}.\n\n"
        f"RAW METRICS DATA (Last 12 months):\n{data_string}\n\n"
        "Provide a single, structured paragraph covering:\n"
        "1) Water body condition (changes in area/stability)\n"
        "2) Vegetation quality (trends/health)\n"
        "3) Flood/Drought patterns (seasonal risk)\n"
        "4) Recommendations or warnings for local authorities/users."
    )
    return prompt

def invoke_bedrock_analysis(tile_id, raw_metrics_data):
    if DRY_RUN:
        return "DRY_RUN: Bedrock invocation skipped."

    if not raw_metrics_data:
        return "AI Summary: No raw metrics data available to perform analysis."

    prompt = create_analysis_prompt(tile_id, raw_metrics_data)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 300,
            "temperature": 0.7
        }
    }

    try:
        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )

        body = response.get("body")
        if hasattr(body, "read"):
            raw_text = body.read().decode("utf-8")
            parsed = json.loads(raw_text)
        else:
            parsed = body

        ai_summary = parsed.get("output", {}).get("message", {}).get("content", [])
        if ai_summary and isinstance(ai_summary, list):
            first_item = ai_summary[0]
            text = first_item.get("text") if isinstance(first_item, dict) else None
            if text:
                return text.strip()
        return "AI Analysis failed: No usable response from model."

    except Exception as e:
        return f"AI Analysis failed: {e}"


def lambda_handler(event, context):
    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            },
            "body": ""
        }

    try:
        body = json.loads(event.get("body", "{}"))
        tile_name = body["tile"].upper()
    except Exception as e:
        return {"statusCode": 400, "body": json.dumps({"error": f"Invalid input: {e}"})}

    start_date = datetime.now(timezone.utc) - relativedelta(months=24)
    end_date = datetime.now(timezone.utc)

    all_metrics_data = []
    latest_thumbnail_key = None
    latest_timestamp = datetime.min.replace(tzinfo=timezone.utc)

    paginator = s3_client.get_paginator("list_objects_v2")
    current_date = start_date
    while current_date <= end_date:
        prefix = f"results/{tile_name}/{current_date.strftime('%Y/%m')}/"
        try:
            pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix)
            for page in pages:
                for obj in page.get("Contents", []):
                    key = obj.get("Key")
                    if key.endswith("metrics.json"):
                        try:
                            file_content = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)["Body"].read().decode("utf-8")
                            metrics = json.loads(file_content)
                            # Extract date from S3 path
                            parts = key.split("/")
                            date_str = None
                            for i in range(len(parts)-2):
                                y,m,d = parts[i], parts[i+1], parts[i+2]
                                if y.isdigit() and m.isdigit() and d.isdigit():
                                    date_str = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
                                    break
                            if not date_str:
                                lm = obj.get("LastModified")
                                if lm.tzinfo is None:
                                    lm = lm.replace(tzinfo=timezone.utc)
                                date_str = lm.strftime("%Y-%m-%d")
                            all_metrics_data.append({"date": date_str, "data": metrics})
                        except Exception as e:
                            print(f"Error reading metrics {key}: {e}")
                    elif key.endswith("thumb.png"):
                        lm = obj.get("LastModified")
                        if lm:
                            if lm.tzinfo is None:
                                lm = lm.replace(tzinfo=timezone.utc)
                            if lm > latest_timestamp:
                                latest_timestamp = lm
                                latest_thumbnail_key = key
        except ClientError as e:
            print(f"S3 error for prefix {prefix}: {e}")
        current_date += relativedelta(months=1)

    all_metrics_data.sort(key=lambda x: x["date"])

    if not all_metrics_data:
        return {"statusCode": 404, "headers": {"Access-Control-Allow-Origin": "*"}, "body": json.dumps({"error": f"No metrics data found for tile {tile_name}"})}

    thumbnail_url = None
    if latest_thumbnail_key:
        try:
            thumbnail_url = s3_client.generate_presigned_url("get_object", Params={"Bucket": BUCKET_NAME, "Key": latest_thumbnail_key}, ExpiresIn=3600)
        except ClientError as e:
            print(f"Error generating presigned URL: {e}")

    ai_summary = invoke_bedrock_analysis(tile_name, all_metrics_data)

    response_body = {
        "tile": tile_name,
        "analysis_period": f"{start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}",
        "most_recent_thumbnail_url": thumbnail_url,
        "raw_metrics_count": len(all_metrics_data),
        "latest_data_date": all_metrics_data[-1]["date"],
        "raw_metrics_data": all_metrics_data,
        "ai_summary": ai_summary
    }

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
        },
        "body": json.dumps(response_body, default=json_default_serializer)
    }

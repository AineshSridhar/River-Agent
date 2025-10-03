import os, json, boto3, logging, time

logger = logging.getLogger(__name__)

def summarize_with_bedrock(result_dict, model_id = "anthropic.claude-v1", max_tokens=200):
    if os.environ.get("ENABLE_BEDROCK", "false").lower() != "true":
        return "[Bedrock disabled]"
    
    prompt = (
        "You are an assistant that writes a concise, professional incident report. "
        "Input: JSON with water_area_m2, foam_fraction, water_mask_image, foam_mask_image. "
        f"Data: {json.dumps(result_dict, indent = None)[:4000]} \n"
        "Output a 3-4 sentence plain-text report, include recommended action and confidence."
    )

    client = boto3.client("bedrock-runtime")
    try:
        response = client.invoke_model(
            modelId = model_id,
            contentType = "application/json",
            accept = "application/json",
            body = json.dumps({"input": prompt, "max_tokens": max_tokens})
        )
        body = response["body"].read().decode("utf-8")
        return body
    except Exception as e: 
        logger.exception("Bedrock call failed")
        return f"[Bedrock error] {e}"
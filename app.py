# basic fastapi app
from datetime import datetime
from fastapi import FastAPI, Request
import boto3
from botocore.exceptions import ClientError
import uvicorn
import requests
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

client = boto3.client("bedrock-runtime", region_name="us-east-1")

app = FastAPI()


class CompleationModel(BaseModel):
    model: str
    messages: list[dict]
    temperature: float = 0.7


@app.get("/health")
def health():
    return {"status": "OK"}


@app.get("/models")
def models():
    # mistral - mistral.mistral-large-2402-v1:0
    # mistral.mistral-small-2402-v1:0
    return {
        "object": "list",
        "data": [
            # {
            #     "id": "mistral.mistral-large-2402-v1:0",
            #     "object": "model",
            #     "created": 1734112601,
            #     "owned_by": "system"
            # },
            {
                "id": "qwen2.5-coder:14b",
                "object": "model",
                "created": 1734112601,
                "owned_by": "system"
            }
        ]
    }

@app.post("/v1/chat/my-completions")
async def generate(inp: CompleationModel):
    model = inp.model
    messages = inp.messages
    temperature = inp.temperature

    # convert this into format for bedrock
    bedrock_messages = []
    for message in messages:
        bedrock_messages.append({
            "role": message.get("role"),
            "content": [{"text": message.get("content")}]
        })

    try:
        response = client.converse(
            modelId=model,
            messages=bedrock_messages,
            inferenceConfig={"maxTokens": 512, "temperature": temperature, "topP": 0.9},
        ) 
        response_text = response["output"]["message"]["content"][0]["text"]
        response = {
            "id": response["ResponseMetadata"]["RequestId"],
            "object": "chat.completion",
            "created": int(datetime.utcnow().timestamp()), # current timestamp
            "model": model,
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "index": 0,
                "message": {
                "role": "assistant",
                "content": response_text,
                },
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "service_tier": "default",
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21,
                "completion_tokens_details": {
                "reasoning_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0
                }
            }
        }
        return response
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model}'. Reason: {e}")
        exit(1)


TARGET_URL = "http://localhost:11434/v1/chat/completions"  # Change to the actual target service URL
@app.post("/v1/chat/completions")
async def proxy_endpoint(request: Request):
    payload = await request.json()  # Read incoming payload
    
    response = requests.post(TARGET_URL, json=payload)  # Forward the payload
    
    return response.json()  # Return the response from the target service

 # Change to the actual target service URL
@app.post("/chat/completions")
async def proxy_endpoint_v2(request: Request):
    payload = await request.json()  # Read incoming payload
    if payload.get("stream", False):
        response = requests.post(TARGET_URL, json=payload, stream=True)  # Forward the payload with stream
        return StreamingResponse(response.raw)  # Return the streaming response from the target service
    else:
        response = requests.post(TARGET_URL, json=payload)  # Forward the payload
        return response.json()  # Return the response from the target service



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


    






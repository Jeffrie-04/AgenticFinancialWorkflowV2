import boto3
import json
from botocore.config import Config

DEFAULT_CONFIG = Config(
    read_timeout=180,
    connect_timeout=60,
    retries={'max_attempts': 2}
)


def get_bedrock_client(config=DEFAULT_CONFIG):
    return boto3.client('bedrock-runtime', region_name='us-east-1', config=config)


def clean_json_text(text):
    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Find JSON in response
    start = text.find('{')
    end = text.rfind('}') + 1
    if start != -1 and end > start:
        text = text[start:end]
    else:
        print("Error: Could not find JSON in response")
        print("Raw response:", text[:500])
        exit(1)

    return text


def parse_json_response(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print("Cleaned text:", text[:500])
        exit(1)

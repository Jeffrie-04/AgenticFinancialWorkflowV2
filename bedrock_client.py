import os
import json
import boto3
from botocore.config import Config
from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency for local development
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

# Which backend call_model() actually uses. Default is "openai" (GPT-5.6 Terra
# via bedrock-mantle) since that's what currently has quota; set
# MODEL_PROVIDER=claude in the environment to fall back to Bedrock's native
# Claude Haiku invoke_model path once its quota clears — no code changes needed.
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "openai")

OPENAI_MODEL_ID = "us.openai.gpt-5.6-terra"
CLAUDE_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

CLAUDE_CONFIG = Config(read_timeout=180, connect_timeout=60, retries={'max_attempts': 2})


def _call_openai(prompt):
    client = OpenAI()  # reads OPENAI_API_KEY / OPENAI_BASE_URL from the environment
    response = client.chat.completions.create(
        model=OPENAI_MODEL_ID,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _call_claude(prompt):
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1', config=CLAUDE_CONFIG)
    response = bedrock.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        contentType='application/json',
        accept='application/json',
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8000,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }),
    )
    response_body = json.loads(response['body'].read())
    return response_body['content'][0]['text']


def call_model(prompt):
    if MODEL_PROVIDER == "claude":
        return _call_claude(prompt)
    return _call_openai(prompt)


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

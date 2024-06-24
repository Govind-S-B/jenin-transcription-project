import requests
import json
import regex as re
from typing import Dict, Any


def extract_json_from_string(input_string: str) -> Dict[str, Any]:
    pattern = r'\{(?:[^{}]|(?R))*\}'
    matches = re.findall(pattern, input_string)
    return json.loads(matches[0]) if matches else {}


def simple_llm_inference(prompt: str, provider: str, model: str, api_key: str) -> str:
    providers = {
        "together": {
            "endpoint": 'https://api.together.xyz/inference',
            "models": {
                "llama3-70b": "meta-llama/Llama-3-70b-chat-hf",
                "llama3-8b": "meta-llama/Llama-3-8b-chat-hf"
            }
        },
        "groq": {
            "endpoint": 'https://api.groq.com/openai/v1/chat/completions',
            "models": {
                "llama3-70b": "llama3-70b-8192",
                "llama3-8b": "llama3-8b-8192"
            }
        },
        "google": {
            "endpoint": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={api_key}",
            "models": {
                "gemini-1.5-pro": "gemini-1.5-pro-latest"
            }
        }
    }

    if provider not in providers or model not in providers[provider]["models"]:
        raise ValueError(f"Invalid provider or model: {provider}, {model}")

    endpoint = providers[provider]["endpoint"]
    model_name = providers[provider]["models"][model]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "max_tokens": 6000,
        "temperature": 0.2,
        "messages": [{"content": prompt, "role": "user"}]
    }

    if provider == "google":
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.4,
                "topK": 32,
                "topP": 1,
                "maxOutputTokens": 4096,
                "stopSequences": []
            }
        }
        headers = {"Content-Type": "application/json"}

    response = requests.post(endpoint, headers=headers, json=payload)
    response.raise_for_status()

    content = response.json()
    if provider == "together":
        return content["output"]["choices"][0]["text"]
    elif provider == "groq":
        return content["choices"][0]["message"]["content"]
    elif provider == "google":
        return content["candidates"][0]["content"]["parts"][0]["text"]

    raise ValueError(f"Unsupported provider: {provider}")

import requests
from typing import Dict, Any


def transcribe_audio(
        file_path: str,
        api_key: str) -> str:
    url = "https://api.groq.com/openai/v1/audio/transcriptions"

    headers = {
        "Authorization": f"bearer {api_key}"
    }

    files = {
        "file": (file_path.split("/")[-1], open(file_path, "rb"))
    }

    data = {
        "model": "whisper-large-v3",
        "temperature": 0,
        "response_format": "json"
    }

    try:
        response = requests.post(
            url, headers=headers, files=files, data=data)

        response.raise_for_status()
        return response.json()["text"]

    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")
    except IOError as e:
        raise Exception(f"File read error: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")

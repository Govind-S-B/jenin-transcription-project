import requests
from typing import Dict, Any


def transcribe_audio(file_path: str, api_key: str) -> str:
    url = 'https://api.deepgram.com/v1/listen'
    params = {
        'model': 'whisper-medium',
        'smart_format': 'true',
        'language': 'en'
    }
    headers = {
        'Authorization': f'Token {api_key}',
        'Content-Type': 'audio/wav'
    }

    try:
        with open(file_path, 'rb') as audio_file:
            response = requests.post(
                url, params=params, headers=headers, data=audio_file)

        response.raise_for_status()
        return response.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")
    except IOError as e:
        raise Exception(f"File read error: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")

import os
import requests
from pathlib import Path

# API endpoint
BASE_URL = "http://localhost:8000"  # Adjust if your server is running on a different host/port

def send_file_to_endpoint(endpoint, file_path):
    with open(file_path, "rb") as audio_file:
        files = {"audio_file": (file_path.name, audio_file, "audio/wav")}
        response = requests.post(f"{BASE_URL}/{endpoint}", files=files)
    
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code} - {response.text}"

def process_audio_files(directory):
    audio_dir = Path(directory)
    wav_files = list(audio_dir.glob("*.wav"))

    if not wav_files:
        print(f"No WAV files found in {directory}")
        return

    for wav_file in wav_files:
        print(f"\nProcessing: {wav_file.name}")
        
        # Transcribe
        print("Transcription:")
        transcription_result = send_file_to_endpoint("transcribe", wav_file)
        if isinstance(transcription_result, dict) and "transcription" in transcription_result:
            print(transcription_result["transcription"])
        else:
            print(transcription_result)

        # Summarize
        print("\nSummary:")
        summary_result = send_file_to_endpoint("summarize", wav_file)
        if isinstance(summary_result, dict) and "summary" in summary_result:
            print(summary_result["summary"])
        else:
            print(summary_result)

        print("\n" + "="*50)

if __name__ == "__main__":
    test_audio_dir = "test_audio"  # Change this if your test audio files are in a different directory
    process_audio_files(test_audio_dir)
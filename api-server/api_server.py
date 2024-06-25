import os
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from api_key_load_balancer import ApiKeyLoadBalancer
from llm_inference import simple_llm_inference, extract_json_from_string
from transcriber_inference import transcribe_audio
from prompt import SUMMARY_PROMPT

app = FastAPI()

DEEPGRAM_KEYS = ApiKeyLoadBalancer(path="./keys/deepgram_api_keys.json")
TOGETHER_KEYS = ApiKeyLoadBalancer(path="./keys/together_api_keys.json")
GROQ_KEYS = ApiKeyLoadBalancer(path="./keys/groq_api_keys.json")

AUDIO_DIR = "./audio"
os.makedirs(AUDIO_DIR, exist_ok=True)


def process_audio_file(audio_file: UploadFile) -> str:
    if not audio_file.filename.lower().endswith('.wav'):
        raise HTTPException(
            status_code=400, detail="Only WAV files are allowed")

    file_path = os.path.join(AUDIO_DIR, audio_file.filename)
    try:
        with open(file_path, "wb") as buffer:
            content = audio_file.file.read()
            buffer.write(content)
        return file_path
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/transcribe")
async def transcribe(audio_file: UploadFile = File(...)):
    file_path = None
    try:
        file_path = process_audio_file(audio_file)
        transcribed_text = transcribe_audio(
            file_path=file_path, api_key=GROQ_KEYS.get_key())
        return {"transcription": transcribed_text}
    finally:
        if file_path and os.path.exists(file_path):
            audio_file.file.close()
            time.sleep(0.1)
            try:
                os.remove(file_path)
            except PermissionError:
                print(f"Unable to remove file: {file_path}. It may still be in use.")


@app.post("/summarize")
async def summarize(audio_file: UploadFile = File(...)):
    file_path = None
    try:
        file_path = process_audio_file(audio_file)
        transcribed_text = transcribe_audio(
            file_path=file_path, api_key=GROQ_KEYS.get_key())

        prompt = SUMMARY_PROMPT.format(TEXT=transcribed_text)
        summary_response = simple_llm_inference(
            prompt=prompt, provider="groq", model="llama3-70b", api_key=GROQ_KEYS.get_key())
        summary = extract_json_from_string(summary_response)

        return summary
    finally:
        if file_path and os.path.exists(file_path):
            audio_file.file.close()
            time.sleep(0.1)
            try:
                os.remove(file_path)
            except PermissionError:
                print(f"Unable to remove file: {file_path}. It may still be in use.")

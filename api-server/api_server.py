import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from api_key_load_balancer import ApiKeyLoadBalancer
from llm_inference import simple_llm_inference, extract_json_from_string
from inference import transcribe_audio
from prompt import SUMMARY_PROMPT

app = FastAPI()

DEEPGRAM_KEYS = ApiKeyLoadBalancer(path="./keys/deepgram_api_keys.json")
TOGETHER_KEYS = ApiKeyLoadBalancer(path="./keys/together_api_keys.json")

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
    try:
        file_path = process_audio_file(audio_file)
        transcribed_text = transcribe_audio(file_path, DEEPGRAM_KEYS.get_key())
        return {"transcription": transcribed_text}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/summarize")
async def summarize(audio_file: UploadFile = File(...)):
    try:
        file_path = process_audio_file(audio_file)
        transcribed_text = transcribe_audio(file_path, DEEPGRAM_KEYS.get_key())

        prompt = SUMMARY_PROMPT.format(TEXT=transcribed_text)
        summary_response = simple_llm_inference(
            prompt=prompt, provider="together", model="llama3-70b", api_key=TOGETHER_KEYS.get_key())
        summary = extract_json_from_string(summary_response)["summary"]
        return {"summary": summary}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

import os
import time
import tempfile
import pygame
from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
from gtts import gTTS
import sounddevice as sd
import numpy as np
import wave
from openai import OpenAI
import uvicorn

app = FastAPI()

# Khởi tạo OpenAI Client
client = OpenAI(
    base_url="http://192.168.1.67:11434/v1",
    api_key='ollama',
)

model = WhisperModel("medium.en", device="cpu", compute_type="int8")

# @app.post("/record")
# def record_audio(duration: int = 5, samplerate: int = 16000):
#     filename = "temp_audio.wav"
#     print("🎙️ Recording...")
#     recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
#     sd.wait()
    
#     with wave.open(filename, "wb") as wf:
#         wf.setnchannels(1)
#         wf.setsampwidth(2)
#         wf.setframerate(samplerate)
#         wf.writeframes(recording.tobytes())
    
#     print("✅ Record Done.")
#     return {"message": "Recording saved", "file": filename}

@app.post("/speech-to-text")
def speech_to_text(audio: UploadFile = File(...)):
    temp_audio_path = f"temp_{audio.filename}"
    with open(temp_audio_path, "wb") as f:
        f.write(audio.file.read())
    
    segments, _ = model.transcribe(temp_audio_path)
    os.remove(temp_audio_path)
    text_result = " ".join(segment.text for segment in segments)
    
    return {"text": text_result}

@app.post("/chat")
def generate_response(text: str):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": "You are an AI assistant designed to support drivers with real-time information and advice. Provide clear, concise, and accurate responses in no more than two sentences. Prioritize safety and ensure easy-to-understand answers while avoiding distractions. If a user feels tired or drowsy, strongly recommend pulling over at a safe location to rest. Politely guide users back to driving-related topics if they ask unrelated questions. Keep your tone professional, informative, and friendly."},
            {"role": "user", "content": text}],
        model="llama3.2",
    )
    return {"response": chat_completion.choices[0].message.content}

@app.post("/text-to-speech")
def text_to_speech_and_play(text: str):
    temp_audio_path = tempfile.mktemp(suffix=".mp3")  
    tts = gTTS(text)
    tts.save(temp_audio_path) 

    pygame.mixer.init()
    pygame.mixer.music.load(temp_audio_path)  
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)  

    pygame.mixer.quit()
    time.sleep(0.5)

    os.remove(temp_audio_path)
    return {"message": "Audio played successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

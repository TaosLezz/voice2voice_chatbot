import os
import time
import tempfile
import pygame
from faster_whisper import WhisperModel
from gtts import gTTS
import sounddevice as sd
import numpy as np
import wave
from openai import OpenAI

# Khởi tạo OpenAI Client
client = OpenAI(
    base_url="http://192.168.1.67:11434/v1",
    api_key='ollama',
)

model = WhisperModel("base", device="cpu", compute_type="int8")

def record_audio(filename="temp_audio.wav", duration=5, samplerate=16000):
    print("🎙️ Đang ghi âm... (Nói trong 5 giây)")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(recording.tobytes())

    print("✅ Ghi âm hoàn tất.")
    return filename


def speech_to_text(audio_path):
    segments, _ = model.transcribe(audio_path)
    return " ".join(segment.text for segment in segments)

def generate_response(text):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": text}],
        model="llama3.2",
    )
    return chat_completion.choices[0].message.content

def text_to_speech_and_play(text):
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


def run_cmd_chatbot():
    print("🚀 Voice Chatbot bằng CMD - Nhấn Ctrl+C để thoát")
    while True:
        try:
            # Ghi âm giọng nói
            audio_file = record_audio()


            text_input = speech_to_text(audio_file)
            print(f"🗣️ Bạn: {text_input}")

            response_text = generate_response(text_input)
            print(f"🤖 AI: {response_text}")

            text_to_speech_and_play(response_text)

        except KeyboardInterrupt:
            print("\n🚪 Thoát chatbot.")
            break

if __name__ == "__main__":
    run_cmd_chatbot()

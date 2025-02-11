# -*- coding: utf-8 -*-
import gradio as gr
from faster_whisper import WhisperModel  # Import Faster-Whisper
from gtts import gTTS
import os
from openai import OpenAI
from tempfile import NamedTemporaryFile

# Initialize OpenAI Client
client = OpenAI(
    # base_url='http://localhost:11434/v1',
    base_url="http://192.168.1.67:11434/v1",
    api_key='ollama',  # required but unused
)

# Load the Faster-Whisper model
model = WhisperModel("base", device="cpu", compute_type="int8")  # Load on GPU

def speech_to_text(audio_path):
    segments, _ = model.transcribe(audio_path)
    transcription = " ".join(segment.text for segment in segments)
    return transcription

def generate_response(text):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": text}],
        model="llama3.2",
    )
    return chat_completion.choices[0].message.content

def text_to_speech(text):
    tts = gTTS(text)
    output_audio = NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.save(output_audio.name)
    return output_audio.name

def chatbot_pipeline(audio_path):
    try:
        # Step 1: Convert speech to text
        text_input = speech_to_text(audio_path)
        print("Text: ", text_input)
        
        # Step 2: Get response from LLaMA model
        response_text = generate_response(text_input)
        print("AI: ", response_text)
        # Step 3: Convert response text to speech
        # response_audio_path = text_to_speech(response_text)

        return response_text

    except Exception as e:
        return str(e), None

# Create Gradio Interface
iface = gr.Interface(
    fn=chatbot_pipeline,
    inputs=gr.Audio(type="filepath", label="Speak"),
    outputs=[
        gr.Textbox(label="Response Text"),
        # gr.Audio(label="Response Audio")
    ],
    title="Real-Time Voice-to-Voice Chatbot (Faster-Whisper)"
)

iface.launch(debug=True)  # Launch the interface in debug mode
# while True:
#     text_input = input("Nhap: ")
#     print("Text: ", text_input)

#     # Step 2: Get response from LLaMA model
#     response_text = generate_response(text_input)
#     print("AI:", response_text)
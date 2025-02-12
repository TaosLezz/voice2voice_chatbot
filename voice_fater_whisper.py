# # -*- coding: utf-8 -*-
# import gradio as gr
# from faster_whisper import WhisperModel  # Import Faster-Whisper
# from gtts import gTTS
# import os
# from openai import OpenAI
# from tempfile import NamedTemporaryFile

# # Initialize OpenAI Client
# client = OpenAI(
#     # base_url='http://localhost:11434/v1',
#     base_url="http://192.168.1.67:11434/v1",
#     api_key='ollama',  # required but unused
# )

# # Load the Faster-Whisper model
# model = WhisperModel("base", device="cpu", compute_type="int8")  # Load on GPU

# def speech_to_text(audio_path):
#     segments, _ = model.transcribe(audio_path)
#     transcription = " ".join(segment.text for segment in segments)
#     return transcription

# def generate_response(text):
#     chat_completion = client.chat.completions.create(
#         messages=[{"role": "user", "content": text}],
#         model="llama3.2",
#     )
#     return chat_completion.choices[0].message.content

# def text_to_speech(text):
#     tts = gTTS(text)
#     output_audio = NamedTemporaryFile(suffix=".mp3", delete=False)
#     tts.save(output_audio.name)
#     return output_audio.name

# def chatbot_pipeline(audio_path):
#     try:
#         # Step 1: Convert speech to text
#         text_input = speech_to_text(audio_path)
#         print("Text: ", text_input)
        
#         # Step 2: Get response from LLaMA model
#         response_text = generate_response(text_input)
#         print("AI: ", response_text)
#         # Step 3: Convert response text to speech
#         # response_audio_path = text_to_speech(response_text)

#         return response_text

#     except Exception as e:
#         return str(e), None

# # Create Gradio Interface
# iface = gr.Interface(
#     fn=chatbot_pipeline,
#     inputs=gr.Audio(type="filepath", label="Speak"),
#     outputs=[
#         gr.Textbox(label="Response Text"),
#         # gr.Audio(label="Response Audio")
#     ],
#     title="Real-Time Voice-to-Voice Chatbot (Faster-Whisper)"
# )

# iface.launch(debug=True)  # Launch the interface in debug mode
import gradio as gr
from faster_whisper import WhisperModel
from gtts import gTTS
import os
import tempfile
from openai import OpenAI
import time
import pygame

# Initialize OpenAI Client
client = OpenAI(
    base_url="http://192.168.1.67:11434/v1",
    api_key='ollama',
)

# Load Faster-Whisper model
model = WhisperModel("base", device="cpu", compute_type="int8")

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
    tts = gTTS(text)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    
    pygame.mixer.init()
    pygame.mixer.music.load(temp_audio.name)
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    
    os.unlink(temp_audio.name)

def chat_pipeline(audio_path, chat_history):
    try:
        # Convert speech to text
        text_input = speech_to_text(audio_path)
        chat_history.append({"role": "user", "content": text_input})
        
        # Generate AI response
        response_text = generate_response(text_input)
        chat_history.append({"role": "assistant", "content": response_text})
        
        # Convert response text to speech and play immediately
        text_to_speech_and_play(response_text)
        
        return chat_history
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return chat_history

# Create real-time chat interface
with gr.Blocks() as iface:
    gr.Markdown("# Real-Time Voice Chatbot")
    chatbox = gr.Chatbot(type="messages")
    audio_input = gr.Audio(type="filepath")
    
    audio_input.change(chat_pipeline, inputs=[audio_input, chatbox], outputs=[chatbox])
    
iface.launch(debug=True)
import streamlit as st
import sounddevice as sd
import numpy as np
import whisper
from scipy.io.wavfile import write
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI
import io
import time
import torch

def record_audio(duration, fs=44100):
    """Record system audio for the specified duration with a timer display."""
    st.write("Recording system audio...")
    
    # Initialize progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Start recording
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
    start_time = time.time()
    
    # Update timer and progress bar during recording
    while time.time() - start_time < duration:
        elapsed_time = time.time() - start_time
        remaining_time = duration - elapsed_time
        progress = int((elapsed_time / duration) * 100)
        
        progress_bar.progress(progress)
        status_text.write(f"Time remaining: {remaining_time:.1f} seconds")
        time.sleep(0.1)  # Small delay to prevent rapid updates
    
    # Finish recording
    sd.wait()
    progress_bar.progress(100)
    status_text.write("Recording complete.")
    
    return audio

#def record_audio(duration, fs=44100):
#    """Record system audio for the specified duration."""
#    st.write("Recording system audio...")
#    audio = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
#    sd.wait()  # Wait until recording is finished
#    st.write("Recording complete.")
#    return audio

def save_audio_to_file(audio, filename, fs=44100):
    """Save the recorded audio to a WAV file."""
    write(filename, fs, audio)
    st.write(f"Audio saved to {filename}")

@st.cache_resource
def load_whisper_model(model_name):
    """Load the Whisper model."""
    return whisper.load_model(model_name)

def transcribe_audio_with_whisper(filename, model):
    """Transcribe audio from a saved WAV file using Whisper."""
    result = model.transcribe(filename)
    return result["text"].strip()

LANGUAGE_OPTIONS = {
    "English (US)": "en-US",
    "Mandarin Chinese (Simplified)": "zh-CN",
    "Cantonese (Hong Kong)": "zh-HK",
    "Mandarin Chinese (Traditional)": "zh-TW",
    "Spanish": "es-ES",
    "French": "fr-FR",
    "German": "de-DE",
    "Japanese": "ja-JP",
    "Korean": "ko-KR"
}

GPT_MODEL_OPTIONS = {
    "gpt-3.5-turbo ($0.002 per 1K tokens)": ("gpt-3.5-turbo", 0.002),
    "gpt-4 ($0.03 per 1K tokens)": ("gpt-4", 0.03),
    "gpt-4o ($0.05 per 1K tokens)": ("gpt-4o", 0.05)
}

st.title("System Audio to Text Transcription (Multi-Language)")
st.write("Record system audio, save it, and transcribe it into text in your chosen language.")

# Sidebar settings
st.sidebar.title("Settings")
duration = st.sidebar.slider("Select duration for recording (seconds):", 30, 1800, 30, 10)
miniute_time = round(duration/60,2)
mini_bar = st.sidebar.write(str(miniute_time) + ' miniute')
model_name = st.sidebar.selectbox("Choose a Whisper model:", ["base", "medium", "large"], index=1)
language = st.sidebar.selectbox("Select Language for Transcription:", list(LANGUAGE_OPTIONS.keys()))
gpt_model_option = st.sidebar.selectbox("Select GPT Model:", list(GPT_MODEL_OPTIONS.keys()))

language_code = LANGUAGE_OPTIONS[language]
gpt_model, cost_per_1k_tokens = GPT_MODEL_OPTIONS[gpt_model_option]

total_cost = 0
model = load_whisper_model(model_name)
st.write(f"Loaded Whisper model: {model_name}")

# Initialize content for download
download_content = ""

if st.button("Start Recording"):
    audio_data = record_audio(duration)
    wav_filename = "recorded_audio.wav"
    save_audio_to_file(audio_data, wav_filename)

    transcription = transcribe_audio_with_whisper(wav_filename, model)
    st.subheader("Transcribed Text")
    st.write(transcription)

    # Append transcription to download content
    download_content += f"### Transcription:\n{transcription}\n\n"

    llm = ChatOpenAI(model_name=gpt_model, openai_api_key="xxxxxx")

    if language != "English (US)":
        translation_prompt = PromptTemplate(
            input_variables=["text"],
            template="Translate the following text to English: {text}"
        )
        translation_message = HumanMessage(content=translation_prompt.format(text=transcription))
        translation_response = llm([translation_message])
        translation_tokens = len(translation_message.content.split())
        translation_cost = (translation_tokens / 1000) * cost_per_1k_tokens
        total_cost += translation_cost

        st.subheader("Translated Text (to English for Reference)")
        st.write(translation_response.content)
        download_content += f"### Translated Text:\n{translation_response.content}\n\n"

    if transcription not in ["Could not understand audio.", ""]:
        summary_prompt = PromptTemplate(
            input_variables=["text"],
            template="Provide a concise summary of the following text, highlighting the key points: {text}"
        )
        summary_message = HumanMessage(content=summary_prompt.format(text=transcription))
        summary_response = llm([summary_message])
        st.subheader("Summary")
        st.write(summary_response.content)
        download_content += f"### Summary:\n{summary_response.content}\n\n"

        questions_prompt = PromptTemplate(
            input_variables=["text"],
            template="After reading the following text, suggest relevant questions for deeper exploration: {text}"
        )
        questions_message = HumanMessage(content=questions_prompt.format(text=transcription))
        questions_response = llm([questions_message])
        st.subheader("Recommended Questions")
        st.write(questions_response.content)
        download_content += f"### Recommended Questions:\n{questions_response.content}\n\n"

        insights_prompt = PromptTemplate(
            input_variables=["text"],
            template="Analyze the following text and provide actionable insights and key observations: {text}"
        )
        insights_message = HumanMessage(content=insights_prompt.format(text=transcription))
        insights_response = llm([insights_message])
        st.subheader("Insights")
        st.write(insights_response.content)
        download_content += f"### Insights:\n{insights_response.content}\n\n"

    st.sidebar.subheader(f"Estimated Cost")
    st.sidebar.write(f"Total cost: ${total_cost:.4f}")

# Always display the download button after the analysis
if download_content:
    st.download_button(
        label="Download All Transcripts",
        data=download_content,
        file_name="transcripts_and_analyses.txt",
        mime="text/plain"
    )

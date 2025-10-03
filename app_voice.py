# streamlit run system_audio_whisper.py

import os
import io
import time
import shutil
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

import torch
import whisper

# --- OPTIONAL: enable LLM value-add (translation/summaries) if key provided ---
try:
    from langchain.prompts import PromptTemplate
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# =========================
# UI SETUP
# =========================
st.set_page_config(page_title="System Audio → Whisper", page_icon="🎧", layout="centered")
st.title("🎧 System Audio Recorder → Whisper Transcriber (Windows WASAPI)")

st.write(
    "Record **system audio** (YouTube, players, calls) using Windows WASAPI loopback "
    "or fall back to the microphone. Then transcribe with OpenAI Whisper."
)

# =========================
# HELPERS
# =========================
def ensure_int16(audio_np: np.ndarray) -> np.ndarray:
    if audio_np.dtype == np.int16:
        return audio_np
    if audio_np.dtype.kind == "f":
        audio_np = np.clip(audio_np, -1.0, 1.0)
        return (audio_np * np.iinfo(np.int16).max).astype(np.int16)
    if np.issubdtype(audio_np.dtype, np.integer):
        info_in = np.iinfo(audio_np.dtype)
        x = audio_np.astype(np.float32)
        x /= max(abs(info_in.min), info_in.max)
        x = np.clip(x, -1.0, 1.0)
        return (x * np.iinfo(np.int16).max).astype(np.int16)
    return audio_np.astype(np.int16)

def wav_bytes_from_np(audio_np: np.ndarray, fs: int) -> bytes:
    buf = io.BytesIO()
    wav_write(buf, fs, audio_np)
    buf.seek(0)
    return buf.getvalue()

def list_wasapi_output_devices():
    """
    Return list of WASAPI output devices on Windows for loopback capture.
    Each item: {"index": int, "name": str}
    """
    try:
        apis = sd.query_hostapis()
        devs = sd.query_devices()
    except Exception:
        return []

    wasapi_idx = None
    for i, ha in enumerate(apis):
        if "WASAPI" in ha.get("name", ""):
            wasapi_idx = i
            break
    if wasapi_idx is None:
        return []

    out = []
    for i, d in enumerate(devs):
        if d.get("hostapi") == wasapi_idx and d.get("max_output_channels", 0) > 0:
            out.append({"index": i, "name": d.get("name", f"Device {i}")})
    return out

def record_system_audio_wasapi(duration, fs=44100, channels=2, device_index=None):
    """
    Record system output via WASAPI loopback (Windows). If loopback isn't available,
    this will raise and caller can fall back.
    """
    frames = int(duration * fs)
    # WASAPI loopback flag
    settings = sd.WasapiSettings(loopback=True)
    with sd.InputStream(samplerate=fs,
                        channels=channels,
                        dtype='int16',
                        device=device_index,
                        extra_settings=settings):
        audio = sd.rec(frames, samplerate=fs, channels=channels, dtype='int16')
        sd.wait()
    return audio

def record_any_input(duration, fs=44100, channels=2, dtype='int16', device_index=None):
    """
    Generic recorder (mic or chosen input). Used as fallback or for non-Windows.
    """
    frames = int(duration * fs)
    audio = sd.rec(frames, samplerate=fs, channels=channels, dtype=dtype, device=device_index)
    sd.wait()
    return audio

def save_audio_to_file(audio_np, fs=44100, out_dir=r"C:\record", filename=None):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if not filename:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"recording_{ts}_{fs}Hz.wav"
    full_path = out_path / filename
    audio_np = ensure_int16(audio_np)
    wav_write(str(full_path), fs, audio_np)
    return str(full_path)

@st.cache_resource
def load_whisper_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(model_name, device=device)

def transcribe_with_whisper(wav_path: str, model) -> str:
    # Whisper shells out to ffmpeg — make sure Python can see it on PATH.
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg not found in PATH for this Python process. "
            "Restart your IDE/Streamlit after installing, or prepend "
            r'os.environ["PATH"] = r"C:\Path\To\ffmpeg\bin" + os.pathsep + os.environ["PATH"]'
        )
    if not Path(wav_path).exists():
        raise FileNotFoundError(f"Audio file not found at {Path(wav_path).resolve()}")
    result = model.transcribe(str(wav_path))
    return result.get("text", "").strip()

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("Settings")

duration = st.sidebar.slider("Duration (seconds)", 5, 1800, 30, step=5)
fs = st.sidebar.selectbox("Sample rate", [16000, 22050, 32000, 44100, 48000], index=4)
channels = st.sidebar.selectbox("Channels", [1, 2], index=1)
out_dir = st.sidebar.text_input("Save folder", r"C:\record")

# Whisper model choice
whisper_name = st.sidebar.selectbox("Whisper model", ["tiny", "base", "small", "medium", "large"], index=2)
model = load_whisper_model(whisper_name)
st.sidebar.success(f"Whisper loaded: {whisper_name}")

# Device picker (WASAPI outputs)
wasapi_outs = list_wasapi_output_devices()
if platform.system() == "Windows" and wasapi_outs:
    device_label_list = [f'{d["index"]}: {d["name"]}' for d in wasapi_outs]
    dev_choice = st.sidebar.selectbox("Output device to capture (system audio)", device_label_list, index=0)
    chosen_device_index = int(dev_choice.split(":")[0])
    st.sidebar.info("Tip: If you don't hear anything in the recording, try a different WASAPI output device.")
else:
    st.sidebar.warning(
        "WASAPI output devices not found (or not Windows). "
        "Recording will use default input (microphone) unless you specify an input device index below."
    )
    chosen_device_index = None

# Optional: manual input device override
input_override = st.sidebar.text_input("Override input device index (advanced)", "")
if input_override.strip():
    try:
        chosen_device_index = int(input_override.strip())
    except ValueError:
        st.sidebar.error("Device index must be an integer.")

# Optional: OpenAI key for extras (translation/summaries)
openai_key = st.sidebar.text_input("OpenAI API Key (optional for translate/summarize)", type="password")

# Diagnostics
with st.sidebar.expander("Diagnostics"):
    st.write(f"OS: {platform.system()} {platform.release()}")
    st.write(f"Python sees ffmpeg? {bool(shutil.which('ffmpeg'))}")
    try:
        st.write("Default devices (in,out):", sd.default.device)
    except Exception as e:
        st.write("sounddevice error:", e)
    try:
        st.write("PortAudio ver:", sd.get_portaudio_version())
    except Exception:
        pass

# =========================
# MAIN ACTION
# =========================
rec_button = st.button("Start Recording", type="primary")

if rec_button:
    # Progress UI
    prog = st.progress(0)
    status = st.empty()
    start = time.time()
    frames = int(duration * fs)

    try:
        # Show which device we are using
        try:
            dev_name = sd.query_devices(chosen_device_index)["name"] if chosen_device_index is not None else "Default"
        except Exception:
            dev_name = "Unknown/Default"
        st.write(f"Recording from device: **{dev_name}**")

        # Preferred path: WASAPI loopback on Windows
        audio = None
        used_loopback = False
        if platform.system() == "Windows":
            try:
                # warm-up short sleep is optional
                time.sleep(0.05)
                # Use loopback; if chosen_device_index is an output device on WASAPI, this grabs system mix
                settings = sd.WasapiSettings(loopback=True)
                audio = sd.rec(frames, samplerate=fs, channels=channels, dtype='int16',
                               device=chosen_device_index, extra_settings=settings)
                used_loopback = True
            except Exception as e:
                st.warning(f"WASAPI loopback failed ({e}). Falling back to default input (microphone).")

        # Fallback: any input device (mic)
        if audio is None:
            audio = sd.rec(frames, samplerate=fs, channels=channels, dtype='int16',
                           device=chosen_device_index if chosen_device_index is not None else None)

        # Update progress bar while recording
        while True:
            elapsed = time.time() - start
            if elapsed >= duration:
                break
            pct = min(100, int(100 * elapsed / duration))
            prog.progress(pct)
            status.write(f"Time remaining: {max(0.0, duration - elapsed):.1f} s")
            time.sleep(0.1)

        sd.wait()
        prog.progress(100)
        status.write("Recording complete.")

        audio = ensure_int16(audio)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"recording_{'loopback' if used_loopback else 'input'}_{ts}_{fs}Hz_{channels}ch.wav"
        saved_path = save_audio_to_file(audio, fs=fs, out_dir=out_dir, filename=filename)
        st.success(f"Saved: {saved_path}")

        # Playback + download
        wav_bytes = wav_bytes_from_np(audio, fs)
        st.audio(wav_bytes, format="audio/wav")
        st.download_button("Download WAV", data=wav_bytes, file_name=Path(saved_path).name, mime="audio/wav")

        # Transcribe with Whisper
        with st.spinner("Transcribing with Whisper..."):
            transcript = transcribe_with_whisper(saved_path, model)
        st.subheader("Transcript")
        st.write(transcript if transcript else "_(empty)_")

        # Optional translate/summarize (only if key provided + langchain installed)
        if openai_key and LANGCHAIN_AVAILABLE:
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)
            # Translate to English (if non-English)
            translate_btn = st.button("Translate to English")
            if translate_btn:
                tpl = PromptTemplate.from_template("Translate to English:\n\n{text}")
                msg = HumanMessage(content=tpl.format(text=transcript))
                res = llm.invoke([msg])
                st.subheader("Translation (English)")
                st.write(res.content)

            # Summarize
            summarize_btn = st.button("Summarize")
            if summarize_btn:
                tpl = PromptTemplate.from_template("Provide a concise summary highlighting key points:\n\n{text}")
                msg = HumanMessage(content=tpl.format(text=transcript))
                res = llm.invoke([msg])
                st.subheader("Summary")
                st.write(res.content)

        elif openai_key and not LANGCHAIN_AVAILABLE:
            st.info("LangChain not installed. Run: pip install langchain langchain-openai")

    except Exception as e:
        st.error(f"Recording/transcription failed: {e}")

# Footer tips
st.caption(
    "Windows: enable **Stereo Mix** in Sound Control Panel or install **VB-CABLE** and pick the device. "
    "If you get silence, try a different WASAPI output device. "
    "If Whisper errors with WinError 2, make sure **ffmpeg** is visible to *this* Python process."
)

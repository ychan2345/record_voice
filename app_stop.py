# streamlit run audio_record_stop_anytime.py

import io
import os
import time
import shutil
import platform
import tempfile
from queue import Queue
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write

import whisper
import torch

# =========================
# App setup
# =========================
st.set_page_config(page_title="Audio Recorder (Stop Anytime)", page_icon="🎙️")
st.title("🎙️ Audio Recorder (Select Any Input) — Stop Anytime")
st.write(
    "Select **any input device** (mic, Stereo Mix, VB-CABLE, BlackHole, or a monitor source). "
    "Click **Start** to begin and **Stop** whenever you want. To capture system audio, choose a loopback/virtual device."
)

# =========================
# Helpers
# =========================
def ensure_int16(audio_np: np.ndarray) -> np.ndarray:
    """Convert any incoming dtype to 16-bit PCM."""
    if audio_np.dtype == np.int16:
        return audio_np
    if audio_np.dtype.kind == "f":
        audio_np = np.clip(audio_np, -1.0, 1.0)
        return (audio_np * np.iinfo(np.int16).max).astype(np.int16)
    if np.issubdtype(audio_np.dtype, np.integer):
        info_in = np.iinfo(audio_np.dtype)
        x = audio_np.astype(np.float32) / max(abs(info_in.min), info_in.max)
        x = np.clip(x, -1.0, 1.0)
        return (x * np.iinfo(np.int16).max).astype(np.int16)
    return audio_np.astype(np.int16)

def wav_bytes_from_np(audio_np: np.ndarray, fs: int) -> bytes:
    """Encode a NumPy array to WAV bytes (16-bit PCM)."""
    buf = io.BytesIO()
    write(buf, fs, ensure_int16(audio_np))
    buf.seek(0)
    return buf.getvalue()

def save_wav_bytes_to_temp(wav_bytes: bytes) -> str:
    """Write in-memory WAV bytes to a temp file and return absolute path."""
    p = Path(tempfile.gettempdir()) / f"rec_{int(time.time())}.wav"
    with open(p, "wb") as f:
        f.write(wav_bytes)
    return str(p)

@st.cache_resource
def load_whisper_model(name="small"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(name, device=device)

def transcribe_with_whisper(wav_path: str, model) -> str:
    """Call Whisper on a WAV file. Ensures ffmpeg is on PATH and file exists."""
    if not shutil.which("ffmpeg"):
        # If your IDE hasn't picked up PATH yet, you can prepend here:
        # os.environ["PATH"] = r"C:\Path\To\ffmpeg\bin" + os.pathsep + os.environ["PATH"]
        raise RuntimeError("FFmpeg not found on PATH for this Python process.")
    p = Path(wav_path)
    if not p.exists():
        raise FileNotFoundError(f"WAV not found: {p.resolve()}")
    result = model.transcribe(str(p))
    return result.get("text", "").strip()

# =========================
# Sidebar settings
# =========================
st.sidebar.header("Settings")
fs = st.sidebar.selectbox("Sample rate (Hz)", [16000, 22050, 32000, 44100, 48000], index=4)
channels = st.sidebar.selectbox("Channels", [1, 2], index=1)
dtype = st.sidebar.selectbox("Sample format", ["int16", "float32"], index=0)

# Device list (same style as your working code)
devices = sd.query_devices()
apis = sd.query_hostapis()
api_names = [a["name"] for a in apis]
device_labels = []
for i, d in enumerate(devices):
    api_name = api_names[d["hostapi"]] if 0 <= d["hostapi"] < len(api_names) else "Unknown API"
    device_labels.append(f"{i}: {d['name']} ({api_name})")

default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None
default_index = default_in if isinstance(default_in, int) and 0 <= default_in < len(devices) else 0
device_choice = st.sidebar.selectbox("Input device (pick virtual/monitor for system audio)", device_labels, index=default_index)
device_index = int(device_choice.split(":")[0])

# Whisper model picker
whisper_name = st.sidebar.selectbox("Whisper model", ["tiny","base","small","medium","large"], index=2)
model = load_whisper_model(whisper_name)
st.sidebar.success(f"Whisper loaded: {whisper_name}")

with st.sidebar.expander("Diagnostics"):
    import pprint
    st.write(f"OS: {platform.system()} {platform.release()}")
    st.write(f"sounddevice: {sd.__version__}")
    try:
        st.write(f"PortAudio: {sd.get_portaudio_version()}")
    except Exception:
        pass
    st.write("Host APIs:"); st.write(pprint.pformat(apis))
    st.write(f"ffmpeg visible to Python? {bool(shutil.which('ffmpeg'))}")
    st.write("Tip: For system audio, pick 'Stereo Mix', 'VB-CABLE Output', 'BlackHole', or a 'monitor' source.")

# =========================
# Recording state (session)
# =========================
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_q" not in st.session_state:
    st.session_state.audio_q = None
if "audio_chunks" not in st.session_state:
    st.session_state.audio_chunks = []
if "stream" not in st.session_state:
    st.session_state.stream = None
if "t0" not in st.session_state:
    st.session_state.t0 = None

# =========================
# Start / Stop handlers
# =========================
def start_recording():
    if st.session_state.recording:
        return
    st.session_state.audio_q = Queue()
    st.session_state.audio_chunks = []
    st.session_state.t0 = time.time()

    # Non-blocking InputStream with callback → push chunks to a queue
    def _callback(indata, frames, t, status):
        # indata shape: (frames, channels)
        st.session_state.audio_q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=int(fs),
        channels=int(channels),
        dtype=dtype,
        device=int(device_index),
        callback=_callback,
        blocksize=0,  # let PortAudio choose optimal buffer
    )
    stream.start()
    st.session_state.stream = stream
    st.session_state.recording = True

def stop_recording():
    if not st.session_state.recording:
        return None
    try:
        if st.session_state.stream is not None:
            st.session_state.stream.stop()
            st.session_state.stream.close()
    finally:
        st.session_state.stream = None
        st.session_state.recording = False

    # Drain remaining queue
    q = st.session_state.audio_q
    chunks = st.session_state.audio_chunks
    while q is not None and not q.empty():
        chunks.append(q.get())

    if not chunks:
        return None

    audio = np.concatenate(chunks, axis=0)
    audio = ensure_int16(audio)

    # Reset buffers
    st.session_state.audio_q = None
    st.session_state.audio_chunks = []
    st.session_state.t0 = None

    return audio

# =========================
# UI: Start/Stop + live status
# =========================
col1, col2 = st.columns(2)
with col1:
    st.button("Start", type="primary", on_click=start_recording, disabled=st.session_state.recording)
with col2:
    stop_clicked = st.button("Stop", type="secondary", disabled=not st.session_state.recording)

# While recording, drain queue → keep chunks + show elapsed
if st.session_state.recording:
    q = st.session_state.audio_q
    while q is not None and not q.empty():
        st.session_state.audio_chunks.append(q.get())

    elapsed = time.time() - (st.session_state.t0 or time.time())
    st.info(f"⏺️ Recording… elapsed: {elapsed:.1f}s  •  Device: {devices[device_index]['name']}")
    # Simple pulsing progress bar for feedback (not tied to duration)
    st.progress(min(1.0, (elapsed % 10.0) / 10.0))

# If user clicked Stop on this run, finalize audio
final_audio = None
if stop_clicked:
    final_audio = stop_recording()

# If we have finished audio this run, present it + transcribe
if final_audio is not None:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"recording_{device_index}_{ts}_{fs}Hz_{channels}ch.wav"

    wav_bytes = wav_bytes_from_np(final_audio, int(fs))
    st.success(f"Saved: {filename}")
    st.audio(wav_bytes, format="audio/wav")
    st.download_button("Download WAV", data=wav_bytes, file_name=filename, mime="audio/wav")

    # Transcribe with Whisper
    with st.spinner("Transcribing with Whisper..."):
        wav_path = save_wav_bytes_to_temp(wav_bytes)
        try:
            text = transcribe_with_whisper(wav_path, model)
            st.subheader("Transcript")
            st.write(text if text else "_(empty)_")
        except Exception as e:
            st.error(f"Transcription failed: {e}")

# =========================
# Footer tips
# =========================
st.caption(
    "For system audio: pick **Stereo Mix** (Windows), **VB-CABLE Output** (Windows), "
    "**BlackHole/Loopback** (macOS), or a **monitor** source (Linux). "
    "If you get silence, the chosen device isn’t carrying audio. "
    "If Whisper shows WinError 2, ensure **ffmpeg** is visible to *this* Python process."
)

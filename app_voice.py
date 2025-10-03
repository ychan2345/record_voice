# streamlit run audio_record_transcribe.py

import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import io, time
from datetime import datetime
import platform
from pathlib import Path
import tempfile, shutil, os

import whisper
import torch

st.set_page_config(page_title="Audio Recorder + Whisper", page_icon="🎙️")
st.title("🎙️ Audio Recorder (Select Any Input) → Whisper Transcriber")
st.write(
    "Select **any input device** (mic, virtual cable, Stereo Mix, BlackHole, or a monitor source) "
    "and record to WAV. To capture system audio, choose a loopback/virtual device."
)

# ---------------- utils ----------------
def wav_bytes_from_np(audio_np, fs):
    buf = io.BytesIO()
    write(buf, fs, audio_np)
    buf.seek(0); return buf

def ensure_int16(audio_np):
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

def record_from_device(duration, fs, channels, dtype, device_index):
    st.write("Recording from:", sd.query_devices(device_index)["name"])
    prog = st.progress(0); status = st.empty()
    audio = sd.rec(int(duration*fs), samplerate=fs, channels=channels, dtype=dtype, device=device_index)
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed >= duration:
            break
        pct = int(100 * elapsed / duration)
        prog.progress(min(pct, 100))
        status.write(f"Time remaining: {max(0.0, duration - elapsed):.1f} s")
        time.sleep(0.1)
    sd.wait()
    prog.progress(100); status.write("Recording complete.")
    return audio

def save_wav_bytes_to_temp(wav_bytes: bytes) -> str:
    """Write in-memory WAV to a temp file and return absolute path (str)."""
    tmp = Path(tempfile.gettempdir()) / f"rec_{int(time.time())}.wav"
    with open(tmp, "wb") as f:
        f.write(wav_bytes)
    return str(tmp)

@st.cache_resource
def load_whisper_model(name="small"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(name, device=device)

def transcribe_with_whisper(wav_path: str, model) -> str:
    import shutil as _sh
    if not _sh.which("ffmpeg"):
        # If you installed via winget but the IDE didn't refresh PATH, prepend here:
        # os.environ["PATH"] = r"C:\Path\To\ffmpeg\bin" + os.pathsep + os.environ["PATH"]
        raise RuntimeError("FFmpeg not found on PATH for this Python process.")
    if not Path(wav_path).exists():
        raise FileNotFoundError(f"WAV not found: {Path(wav_path).resolve()}")
    result = model.transcribe(wav_path)
    return result.get("text", "").strip()

# --------------- sidebar ----------------
st.sidebar.header("Settings")
duration = st.sidebar.slider("Duration (seconds)", 5, 1800, 30, 5)
fs = st.sidebar.selectbox("Sample rate (Hz)", [16000, 22050, 32000, 44100, 48000], index=4)
channels = st.sidebar.selectbox("Channels", [1, 2], index=1)
dtype = st.sidebar.selectbox("Sample format", ["int16", "float32"], index=0)

devices = sd.query_devices()
apis = sd.query_hostapis()
api_names = [a["name"] for a in apis]
device_labels = []
for i, d in enumerate(devices):
    api_name = api_names[d["hostapi"]] if 0 <= d["hostapi"] < len(api_names) else "Unknown API"
    device_labels.append(f"{i}: {d['name']} ({api_name})")

# try to preselect default input device
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

# --------------- action ----------------
if st.button("Start Recording", type="primary"):
    try:
        audio = record_from_device(duration=duration, fs=fs, channels=channels, dtype=dtype, device_index=device_index)
        audio = ensure_int16(audio)  # ensure 16-bit PCM for WAV

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"recording_{device_index}_{ts}_{fs}Hz_{channels}ch.wav"

        wav_buf = wav_bytes_from_np(audio, fs)  # in-memory
        st.success(f"Recorded: {filename}")
        st.audio(wav_buf, format="audio/wav")
        st.download_button("Download WAV", data=wav_buf, file_name=filename, mime="audio/wav")

        # --- Transcribe with Whisper ---
        with st.spinner("Transcribing with Whisper..."):
            wav_path = save_wav_bytes_to_temp(wav_buf.getvalue())  # temp file for whisper/ffmpeg
            text = transcribe_with_whisper(wav_path, model)

        st.subheader("Transcript")
        st.write(text if text else "_(empty)_")

    except Exception as e:
        st.error(f"Recording/transcription failed: {e}")

st.caption(
    "Windows: choose **Stereo Mix** (if available) or install **VB-CABLE** and select it. "
    "macOS: install **BlackHole**/Loopback and pick it here. "
    "Linux: pick the output **monitor** source (PulseAudio/PipeWire). "
    "If you get silence, the selected device isn't carrying audio. "
    "If Whisper shows WinError 2, ensure **ffmpeg** is visible to *this* Python process."
)

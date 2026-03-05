"""Audio mixin – recording, playback, TTS, wake-word detection, transcription."""

import os
import random
import re
import select
import subprocess
import sys
import time
import wave

import numpy as np
import scipy.signal
import sounddevice as sd

from .config import (
    CURRENT_CONFIG,
    INPUT_DEVICE_NAME,
    WAKE_WORD_THRESHOLD,
    ack_sounds_dir,
    thinking_sounds_dir,
)
from .states import BotStates


def _find_output_device():
    """Find the best output device — prefer 'pipewire', fall back to None."""
    try:
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d["max_output_channels"] > 0 and "pipewire" in d["name"].lower():
                print(f"[AUDIO] Using output device [{i}] {d['name']}", flush=True)
                return i
    except Exception:
        pass
    print("[AUDIO] No pipewire device found, using system default", flush=True)
    return None


_OUTPUT_DEVICE = _find_output_device()


class AudioMixin:
    """Mixed into BotGUI – provides all audio-related methods."""

    # -- Wake-word / PTT trigger -----------------------------------------------

    def detect_wake_word_or_ptt(self):
        self.set_state(BotStates.IDLE, "Waiting...")
        self.ptt_event.clear()

        if self.oww_model:
            self.oww_model.reset()

        if self.oww_model is None:
            # Wait for either PTT or web text
            while not self.ptt_event.is_set() and not self.web_text_event.is_set():
                self.ptt_event.wait(timeout=0.2)
            if self.web_text_event.is_set():
                self.web_text_event.clear()
                return "WEB"
            self.ptt_event.clear()
            return "PTT"

        chunk_size = 1280
        oww_sample_rate = 16000

        try:
            device_info = sd.query_devices(kind="input")
            native_rate = int(device_info["default_samplerate"])
        except Exception:
            native_rate = 48000

        use_resampling = native_rate != oww_sample_rate
        input_rate = native_rate if use_resampling else oww_sample_rate
        input_chunk_size = int(chunk_size * (input_rate / oww_sample_rate)) if use_resampling else chunk_size

        try:
            with sd.InputStream(
                samplerate=input_rate,
                channels=1,
                dtype="int16",
                blocksize=input_chunk_size,
                device=INPUT_DEVICE_NAME,
            ) as stream:
                while True:
                    if self.web_text_event.is_set():
                        self.web_text_event.clear()
                        return "WEB"

                    if self.ptt_event.is_set():
                        self.ptt_event.clear()
                        return "PTT"

                    rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
                    if rlist:
                        sys.stdin.readline()
                        return "CLI"

                    data, _ = stream.read(input_chunk_size)
                    audio_data = np.frombuffer(data, dtype=np.int16)

                    if use_resampling:
                        audio_data = scipy.signal.resample(audio_data, chunk_size).astype(np.int16)

                    self.oww_model.predict(audio_data)
                    for model_name in self.oww_model.prediction_buffer.keys():
                        if list(self.oww_model.prediction_buffer[model_name])[-1] > WAKE_WORD_THRESHOLD:
                            self.oww_model.reset()
                            return "WAKE"
        except Exception as e:
            print(f"Wake Word Stream Error: {e}")
            while not self.ptt_event.is_set() and not self.web_text_event.is_set():
                self.ptt_event.wait(timeout=0.2)
            if self.web_text_event.is_set():
                self.web_text_event.clear()
                return "WEB"
            return "PTT"

    # -- Recording -------------------------------------------------------------

    def record_voice_adaptive(self, filename="input.wav"):
        print("Recording (Adaptive)...", flush=True)
        time.sleep(0.5)
        try:
            device_info = sd.query_devices(kind="input")
            samplerate = int(device_info["default_samplerate"])
        except Exception:
            samplerate = 44100

        silence_threshold = 0.006
        silence_duration = 1.5
        max_record_time = 30.0
        buffer = []
        silent_chunks = 0
        chunk_duration = 0.05
        chunk_size = int(samplerate * chunk_duration)

        num_silent_chunks = int(silence_duration / chunk_duration)
        max_chunks = int(max_record_time / chunk_duration)
        recorded_chunks = 0
        silence_started = False

        def callback(indata, frames, time_info, status):
            nonlocal silent_chunks, recorded_chunks, silence_started
            volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
            buffer.append(indata.copy())
            recorded_chunks += 1
            if recorded_chunks < 5:
                return
            if volume_norm < silence_threshold:
                silent_chunks += 1
                if silent_chunks >= num_silent_chunks:
                    silence_started = True
            else:
                silent_chunks = 0

        try:
            with sd.InputStream(
                samplerate=samplerate,
                channels=1,
                callback=callback,
                device=INPUT_DEVICE_NAME,
                blocksize=chunk_size,
            ):
                while not silence_started and recorded_chunks < max_chunks:
                    sd.sleep(int(chunk_duration * 1000))
        except Exception:
            return None

        return self.save_audio_buffer(buffer, filename, samplerate)

    def record_voice_ptt(self, filename="input.wav"):
        print("Recording (PTT)...", flush=True)
        time.sleep(0.5)
        try:
            device_info = sd.query_devices(kind="input")
            samplerate = int(device_info["default_samplerate"])
        except Exception:
            samplerate = 44100

        buffer = []

        def callback(indata, frames, time_info, status):
            buffer.append(indata.copy())

        try:
            with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, device=INPUT_DEVICE_NAME):
                while self.recording_active.is_set():
                    sd.sleep(50)
        except Exception:
            return None

        return self.save_audio_buffer(buffer, filename, samplerate)

    def save_audio_buffer(self, buffer, filename, samplerate=16000):
        if not buffer:
            return None
        audio_data = np.concatenate(buffer, axis=0).flatten()
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
        audio_data = (audio_data * 32767).astype(np.int16)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())
        self.play_sound(self.get_random_sound(ack_sounds_dir))
        return filename

    # -- Transcription ---------------------------------------------------------

    def transcribe_audio(self, filename):
        print("Transcribing...", flush=True)
        try:
            result = subprocess.run(
                [
                    "./whisper.cpp/build/bin/whisper-cli",
                    "-m",
                    "./whisper.cpp/models/ggml-base.en.bin",
                    "-l",
                    "en",
                    "-t",
                    "4",
                    "-f",
                    filename,
                ],
                capture_output=True,
                text=True,
            )
            transcription_lines = result.stdout.strip().split("\n")
            if transcription_lines and transcription_lines[-1].strip():
                last_line = transcription_lines[-1].strip()
                if "]" in last_line:
                    transcription = last_line.split("]")[1].strip()
                else:
                    transcription = last_line
            else:
                transcription = ""
            print(f"Heard: '{transcription}'", flush=True)
            return transcription.strip()
        except Exception as e:
            print(f"Transcription Error: {e}")
            return ""

    # -- TTS (Piper) -----------------------------------------------------------

    def speak(self, text):
        clean = re.sub(r"[^\w\s,.!?:;'\"-]", "", text)
        if not clean.strip():
            return

        print(f"[PIPER SPEAKING] '{clean}'", flush=True)
        voice_model = CURRENT_CONFIG.get("voice_model", "piper/en_GB-semaine-medium.onnx")

        try:
            piper_env = os.environ.copy()
            piper_dir = os.path.abspath("./piper")
            piper_env["LD_LIBRARY_PATH"] = piper_dir + ":" + piper_env.get("LD_LIBRARY_PATH", "")

            self.current_audio_process = subprocess.Popen(
                ["./piper/piper", "--model", voice_model, "--output-raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=piper_env,
            )

            self.current_audio_process.stdin.write(clean.encode() + b"\n")
            self.current_audio_process.stdin.close()

            try:
                device_info = sd.query_devices(_OUTPUT_DEVICE or "default", kind="output")
                native_rate = int(device_info["default_samplerate"])
            except Exception:
                native_rate = 48000

            piper_rate = 22050
            use_native_rate = False

            try:
                sd.check_output_settings(device=_OUTPUT_DEVICE, samplerate=piper_rate)
            except Exception:
                use_native_rate = True

            with sd.RawOutputStream(
                samplerate=native_rate if use_native_rate else piper_rate,
                channels=1,
                dtype="int16",
                device=_OUTPUT_DEVICE,
                latency="low",
                blocksize=2048,
            ) as stream:
                while True:
                    if self.interrupted.is_set():
                        break
                    data = self.current_audio_process.stdout.read(4096)
                    if not data:
                        break

                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    if len(audio_chunk) > 0:
                        self.current_volume = np.max(np.abs(audio_chunk))
                        if use_native_rate:
                            num_samples = int(len(audio_chunk) * (native_rate / piper_rate))
                            audio_chunk = scipy.signal.resample(audio_chunk, num_samples).astype(np.int16)
                        stream.write(audio_chunk.tobytes())
                    else:
                        self.current_volume = 0
                time.sleep(0.5)

        except Exception as e:
            print(f"Audio Error: {e}")
        finally:
            self.current_volume = 0
            if self.current_audio_process:
                if self.current_audio_process.stdout:
                    self.current_audio_process.stdout.close()
                if self.current_audio_process.poll() is None:
                    self.current_audio_process.terminate()
                self.current_audio_process = None

    def _tts_worker(self):
        while True:
            text = None
            with self.tts_queue_lock:
                if self.tts_queue:
                    text = self.tts_queue.pop(0)
                    self.tts_active.set()
            if text:
                self.speak(text)
                self.tts_active.clear()
            else:
                time.sleep(0.05)

    def wait_for_tts(self):
        while self.tts_queue or self.tts_active.is_set():
            if self.interrupted.is_set():
                break
            time.sleep(0.1)

    # -- Sound effects ---------------------------------------------------------

    def _run_thinking_sound_loop(self):
        time.sleep(0.5)
        while self.thinking_sound_active.is_set():
            sound = self.get_random_sound(thinking_sounds_dir)
            if sound:
                self.play_sound(sound)
            for _ in range(50):
                if not self.thinking_sound_active.is_set():
                    return
                time.sleep(0.1)

    def get_random_sound(self, directory):
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if f.endswith(".wav")]
            return os.path.join(directory, random.choice(files)) if files else None
        return None

    def play_sound(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return
        try:
            with wave.open(file_path, "rb") as wf:
                file_sr = wf.getframerate()
                data = wf.readframes(wf.getnframes())
                audio = np.frombuffer(data, dtype=np.int16)

            try:
                device_info = sd.query_devices(_OUTPUT_DEVICE or "default", kind="output")
                native_rate = int(device_info["default_samplerate"])
            except Exception:
                native_rate = 48000

            playback_rate = file_sr
            try:
                sd.check_output_settings(device=_OUTPUT_DEVICE, samplerate=file_sr)
            except Exception:
                playback_rate = native_rate
                num_samples = int(len(audio) * (native_rate / file_sr))
                audio = scipy.signal.resample(audio, num_samples).astype(np.int16)

            sd.play(audio, playback_rate, device=_OUTPUT_DEVICE)
            sd.wait()
        except Exception:
            pass

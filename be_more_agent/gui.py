"""GUI module – tkinter UI, animation, state management, and main loop.

All audio logic lives in audio.py (AudioMixin).
All chat / LLM logic lives in chat.py (ChatMixin).
"""

import atexit
import os
import random
import sys
import threading
import time
import traceback
import warnings

import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk

from openwakeword.model import Model

from .audio import AudioMixin
from .chat import ChatMixin
from .config import (
    OLLAMA_CLIENT,
    TEXT_MODEL,
    WAKE_WORD_MODEL,
    greeting_sounds_dir,
)
from .states import BotStates





class BotGUI(AudioMixin, ChatMixin):
    """Main application class – owns the tkinter window and orchestrates mixins."""

    BG_WIDTH, BG_HEIGHT = 800, 480
    OVERLAY_WIDTH, OVERLAY_HEIGHT = 400, 300

    def __init__(self, master):
        self.master = master
        master.title("Pi Assistant")
        master.attributes("-fullscreen", True)
        master.bind("<Escape>", self.exit_fullscreen)
        master.bind("<Return>", self.handle_ptt_toggle)
        master.bind("<space>", self.handle_speaking_interrupt)
        atexit.register(self.safe_exit)

        # State
        self.current_state = BotStates.WARMUP
        self.current_volume = 0
        self.animations = {}
        self.current_frame_index = 0
        self.current_overlay_image = None

        # Memory
        self.permanent_memory = self.load_chat_history()
        self.session_memory = []

        # Threading events
        self.thinking_sound_active = threading.Event()
        self.last_ptt_time = 0
        self.ptt_event = threading.Event()
        self.recording_active = threading.Event()
        self.interrupted = threading.Event()

        # TTS queue
        self.tts_queue = []
        self.tts_queue_lock = threading.Lock()
        self.tts_thread = None
        self.tts_active = threading.Event()
        self.current_audio_process = None

        # Web text queue (filled by the web server)
        self.web_text_queue = []
        self.web_text_lock = threading.Lock()
        self.web_text_event = threading.Event()

        # Wake-word model
        self.oww_model = self._load_wake_word_model()

        # UI widgets
        self._build_ui()

        # Start
        self.load_animations()
        self.update_animation()

        # Start the lightweight web server for text input
        from .web import start_web_server
        start_web_server(self)

        threading.Thread(target=self.safe_main_execution, daemon=True).start()

    # -- Wake-word loader ------------------------------------------------------

    @staticmethod
    def _load_wake_word_model():
        print("[INIT] Loading Wake Word...", flush=True)
        if not os.path.exists(WAKE_WORD_MODEL):
            print(f"[CRITICAL] Model not found: {WAKE_WORD_MODEL}")
            return None
        try:
            model = Model(wakeword_model_paths=[WAKE_WORD_MODEL])
            print("[INIT] Wake Word Loaded.", flush=True)
            return model
        except TypeError:
            try:
                model = Model(wakeword_models=[WAKE_WORD_MODEL])
                print("[INIT] Wake Word Loaded (New API).", flush=True)
                return model
            except Exception as e:
                print(f"[CRITICAL] Failed to load model: {e}")
        except Exception as e:
            print(f"[CRITICAL] Failed to load model: {e}")
        return None

    # -- UI construction -------------------------------------------------------

    def _build_ui(self):
        self.background_label = tk.Label(self.master)
        self.background_label.place(x=0, y=0, width=self.BG_WIDTH, height=self.BG_HEIGHT)
        self.background_label.bind("<Button-1>", self.toggle_hud_visibility)

        self.overlay_label = tk.Label(self.master, bg="black")
        self.overlay_label.bind("<Button-1>", self.toggle_hud_visibility)

        self.response_text = tk.Text(
            self.master,
            height=6,
            width=60,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg="#ffffff",
            fg="#000000",
            font=("Arial", 12),
        )

        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(
            self.master,
            textvariable=self.status_var,
            background="#2e2e2e",
            foreground="white",
        )

        self.exit_button = ttk.Button(self.master, text="Exit & Save", command=self.safe_exit)

    # -- Lifecycle -------------------------------------------------------------

    def safe_exit(self):
        print("\n--- SHUTDOWN SEQUENCE ---", flush=True)
        if self.current_audio_process:
            try:
                self.current_audio_process.terminate()
                self.current_audio_process.wait(timeout=1)
            except Exception:
                pass

        self.recording_active.clear()
        self.thinking_sound_active.clear()
        self.tts_active.clear()
        self.save_chat_history()

        try:
            OLLAMA_CLIENT.generate(model=TEXT_MODEL, prompt="", keep_alive=0)
        except Exception:
            pass

        self.master.quit()
        sys.exit(0)

    def exit_fullscreen(self, event=None):
        self.master.attributes("-fullscreen", False)
        self.safe_exit()

    # -- HUD / input handlers --------------------------------------------------

    def toggle_hud_visibility(self, event=None):
        try:
            if self.response_text.winfo_ismapped():
                self.response_text.place_forget()
                self.status_label.place_forget()
                self.exit_button.place_forget()
            else:
                self.response_text.place(relx=0.5, rely=0.82, anchor=tk.S)
                self.status_label.place(relx=0.5, rely=1.0, anchor=tk.S, relwidth=1)
                self.exit_button.place(x=10, y=10)
        except tk.TclError:
            pass

    def handle_ptt_toggle(self, event=None):
        current_time = time.time()
        if current_time - self.last_ptt_time < 0.5:
            return
        self.last_ptt_time = current_time

        if self.recording_active.is_set():
            print("[PTT] Toggle OFF", flush=True)
            self.recording_active.clear()
        else:
            if self.current_state == BotStates.IDLE or "Wait" in self.status_var.get():
                print("[PTT] Toggle ON", flush=True)
                self.recording_active.set()
                self.ptt_event.set()

    def handle_speaking_interrupt(self, event=None):
        if self.current_state in (BotStates.SPEAKING, BotStates.THINKING):
            self.interrupted.set()
            self.thinking_sound_active.clear()
            with self.tts_queue_lock:
                self.tts_queue.clear()
            if self.current_audio_process:
                try:
                    self.current_audio_process.terminate()
                except Exception:
                    pass
            self.set_state(BotStates.IDLE, "Interrupted.")

    # -- Animations ------------------------------------------------------------

    def load_animations(self):
        base_path = "faces"
        states = ["idle", "listening", "thinking", "speaking", "error", "capturing", "warmup"]
        for state in states:
            folder = os.path.join(base_path, state)
            self.animations[state] = []
            if os.path.exists(folder):
                files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])
                for filename in files:
                    img = Image.open(os.path.join(folder, filename)).resize(
                        (self.BG_WIDTH, self.BG_HEIGHT)
                    )
                    self.animations[state].append(ImageTk.PhotoImage(img))
            if not self.animations[state]:
                if state in self.animations.get("idle", []):
                    self.animations[state] = self.animations["idle"]
                else:
                    blank = Image.new("RGB", (self.BG_WIDTH, self.BG_HEIGHT), color="#0000FF")
                    self.animations[state].append(ImageTk.PhotoImage(blank))

    def update_animation(self):
        frames = self.animations.get(self.current_state, []) or self.animations.get(
            BotStates.IDLE, []
        )
        if not frames:
            self.master.after(500, self.update_animation)
            return

        if self.current_state == BotStates.SPEAKING:
            self.current_frame_index = (
                random.randint(1, len(frames) - 1) if len(frames) > 1 else 0
            )
        else:
            self.current_frame_index = (self.current_frame_index + 1) % len(frames)

        self.background_label.config(image=frames[self.current_frame_index])
        speed = 50 if self.current_state == BotStates.SPEAKING else 500
        self.master.after(speed, self.update_animation)

    # -- State / text helpers --------------------------------------------------

    def set_state(self, state, msg="", cam_path=None):
        def _update():
            if msg:
                print(f"[STATE] {state.upper()}: {msg}", flush=True)
            if self.current_state != state:
                self.current_state = state
                self.current_frame_index = 0
            if msg:
                self.status_var.set(msg)
            if (
                cam_path
                and os.path.exists(cam_path)
                and state in [BotStates.THINKING, BotStates.SPEAKING]
            ):
                try:
                    img = Image.open(cam_path).resize(
                        (self.OVERLAY_WIDTH, self.OVERLAY_HEIGHT)
                    )
                    self.current_overlay_image = ImageTk.PhotoImage(img)
                    self.overlay_label.config(image=self.current_overlay_image)
                    self.overlay_label.place(x=200, y=90)
                except Exception:
                    pass
            else:
                self.overlay_label.place_forget()

        self.master.after(0, _update)

    def append_to_text(self, text, newline=True):
        def _update():
            self.response_text.config(state=tk.NORMAL)
            self.response_text.insert(tk.END, text + ("\n" if newline else ""))
            self.response_text.see(tk.END)
            self.response_text.config(state=tk.DISABLED)

        self.master.after(0, _update)

    def _stream_to_text(self, chunk):
        def _update():
            self.response_text.config(state=tk.NORMAL)
            self.response_text.insert(tk.END, chunk)
            self.response_text.see(tk.END)
            self.response_text.config(state=tk.DISABLED)

        self.master.after(0, _update)

    # -- Main execution loop ---------------------------------------------------

    def safe_main_execution(self):
        try:
            self._warm_up()
            self.tts_active.set()
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()

            while True:
                trigger_source = self.detect_wake_word_or_ptt()
                if self.interrupted.is_set():
                    self.interrupted.clear()
                    self.set_state(BotStates.IDLE, "Resetting...")
                    continue

                # Web text arrived — skip recording entirely
                if trigger_source == "WEB":
                    with self.web_text_lock:
                        web_text = self.web_text_queue.pop(0) if self.web_text_queue else None
                    if web_text:
                        self.append_to_text(f"YOU (web): {web_text}")
                        self.interrupted.clear()
                        self.chat_and_respond(web_text, img_path=None)
                    continue

                self.set_state(BotStates.LISTENING, "I'm listening!")

                # Always use PTT (Enter-toggle) recording
                audio_file = self.record_voice_ptt()

                if not audio_file:
                    self.set_state(BotStates.IDLE, "Heard nothing.")
                    continue

                user_text = self.transcribe_audio(audio_file)
                if not user_text:
                    self.set_state(BotStates.IDLE, "Transcription empty.")
                    continue

                self.append_to_text(f"YOU: {user_text}")
                self.interrupted.clear()
                self.chat_and_respond(user_text, img_path=None)

        except Exception as e:
            traceback.print_exc()
            self.set_state(BotStates.ERROR, f"Fatal Error: {str(e)[:40]}")

    def _warm_up(self):
        self.set_state(BotStates.WARMUP, "Warming up brains...")
        try:
            OLLAMA_CLIENT.generate(model=TEXT_MODEL, prompt="", keep_alive=-1)
        except Exception as e:
            print(f"Failed to load {TEXT_MODEL}: {e}", flush=True)
        # Pre-load FastText classifier (instant, no Ollama needed)
        from .classifier import _get_model as _load_classifier
        _load_classifier()
        self.play_sound(self.get_random_sound(greeting_sounds_dir))
        print("Models loaded.", flush=True)


def run_app():
    print("--- SYSTEM STARTING ---", flush=True)
    root = tk.Tk()
    BotGUI(root)
    root.mainloop()

"""
SmartVisionX Voice Listener
============================

Push-to-talk background listener for the blind-assist system.
When the UI calls `begin_push_to_talk()`, the listener:
  1. Tells the main app to silence TTS (sends "__SILENCE__").
  2. Listens for up to 8 seconds of speech.
  3. When `end_push_to_talk()` is called, returns the parsed command.

Supported commands (same grammar as before):
    "go to <location>"     -> NAVIGATE_TO:<location>
    "navigate to <loc>"    -> NAVIGATE_TO:<location>
    "take me to <loc>"     -> NAVIGATE_TO:<location>
    "lead me to <loc>"     -> NAVIGATE_TO:<location>
    "find <loc>"           -> NAVIGATE_TO:<location>
    "where is the <loc>"   -> NAVIGATE_TO:<location>
    "stop navigation"      -> STOP_NAVIGATION
    "cancel"               -> STOP_NAVIGATION
    "list locations"       -> LIST_LOCATIONS
    "repeat"               -> REPEAT
    "read text"            -> READ_TEXT

Install dependency:  pip install SpeechRecognition pyaudio
                     (On Windows you may also need PyAudio wheel.)
"""

from __future__ import annotations

import re
import threading
import time
from typing import Callable, List, Optional

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False


# Wake word is OPTIONAL — leaving it empty means every command is processed
# directly.  Set to "vision" to require a wake word for privacy.
WAKE_WORD = ""

# Maps raw speech -> action string.  See parse_command() below.
ACTION_REPEAT         = "REPEAT"
ACTION_READ_TEXT      = "READ_TEXT"
ACTION_STOP_NAV       = "STOP_NAVIGATION"
ACTION_LIST_LOCATIONS = "LIST_LOCATIONS"
ACTION_NAVIGATE       = "NAVIGATE_TO"
ACTION_WHERE_AM_I     = "WHERE_AM_I"

# Control messages sent to the main app (not user commands).
CTRL_SILENCE          = "__SILENCE__"
CTRL_UNSILENCE        = "__UNSILENCE__"


class VoiceListener:
    """Push-to-talk background listener."""

    def __init__(self,
                 on_command: Callable[[str], None],
                 known_locations: Optional[List[str]] = None,
                 language: str = "en-US") -> None:
        self.on_command      = on_command
        self.language        = language
        self.known_locations = known_locations or []
        self._lock           = threading.Lock()

        # Push-to-talk state
        self._ptt_active     = False   # True while user is holding the voice button
        self._ptt_event      = threading.Event()
        self._ptt_result     = None    # last recognised phrase during PTT
        self._calibrated     = False

        self.running     = False
        self._thread     = None
        self._recognizer = None
        self._mic        = None
        if SR_AVAILABLE:
            try:
                self._recognizer = sr.Recognizer()
                self._mic        = sr.Microphone()
            except Exception as exc:
                print(f"[Voice] Microphone init failed: {exc}")

    # ---------- public API ----------

    def is_available(self) -> bool:
        return (SR_AVAILABLE
                and self._recognizer is not None
                and self._mic is not None)

    def set_known_locations(self, names: List[str]) -> None:
        with self._lock:
            self.known_locations = list(names)

    def start(self) -> bool:
        if not self.is_available() or self.running:
            return False
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="voice-listener")
        self._thread.start()
        print("[Voice] Listener started (push-to-talk).")
        return True

    def stop(self) -> None:
        self.running = False
        self._ptt_event.set()        # unblock the PTT wait
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[Voice] Listener stopped.")

    # ---------- PUSH-TO-TALK ----------

    def begin_push_to_talk(self) -> None:
        """Call this from the UI (button-down). Stays silent until end_push_to_talk()."""
        if not self.is_available():
            return
        self._ptt_active = True
        self._ptt_result = None
        self._ptt_event.clear()
        # Tell the main loop to suppress TTS while we listen
        try:
            self.on_command(CTRL_SILENCE)
        except Exception:
            pass

    def end_push_to_talk(self, timeout: float = 6.0) -> Optional[str]:
        """Call this from the UI (button-up). Returns the recognised command or None."""
        if not self._ptt_active:
            return None
        self._ptt_active = False
        self._ptt_event.set()         # unblock the listener loop
        # Wait up to `timeout` seconds for the recogniser to finish its current phrase
        deadline = time.time() + timeout
        while self._ptt_result is None and time.time() < deadline:
            time.sleep(0.05)
        result = self._ptt_result
        self._ptt_result = None
        try:
            self.on_command(CTRL_UNSILENCE)
        except Exception:
            pass
        return result

    # ---------- listener loop ----------

    def _run(self) -> None:
        try:
            with self._mic as src:
                self._recognizer.adjust_for_ambient_noise(src, duration=0.8)
                self._calibrated = True
        except Exception as exc:
            print(f"[Voice] Calibration failed: {exc}")
            self.running = False
            return

        while self.running:
            # Sleep until PTT is activated
            if not self._ptt_active:
                time.sleep(0.05)
                continue

            try:
                with self._mic as src:
                    audio = self._recognizer.listen(
                        src, timeout=1.0,
                        phrase_time_limit=8   # generous, user is dictating
                    )
                text = self._recognizer.recognize_google(
                    audio, language=self.language).lower().strip()
                print(f"[Voice] Heard: '{text}'")   # DEBUG — remove when stable
                with self._lock:
                    locations = list(self.known_locations)
                action = self.parse_command(text, locations) if text else None
                if action:
                    self._ptt_result = action
            except sr.WaitTimeoutError:
                pass
            except Exception:
                # No speech detected in this window — keep listening
                pass

            # If PTT is still active, loop again; otherwise exit PTT
            if not self._ptt_active:
                if self._ptt_result is None:
                    self._ptt_result = ""
                break

    @staticmethod
    def parse_command(text: str,
                      known_locations: Optional[List[str]] = None) -> Optional[str]:
        text = re.sub(r"[^\w\s]", "", (text or "").lower())
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return None

        # Optional wake word
        if WAKE_WORD and WAKE_WORD in text:
            text = text.replace(WAKE_WORD, "").strip()
        if not text:
            return None

        # Stop / cancel
        if re.match(r"^(stop|cancel|abort|halt)(?: navigation| navigating| the navigation)?$", text):
            return ACTION_STOP_NAV
        # Repeat
        if re.match(r"^(repeat|say again|again)$", text):
            return ACTION_REPEAT
        # Read text
        if re.match(r"^(read|read text|read this|what does it say)$", text):
            return ACTION_READ_TEXT
        # List locations
        if re.match(r"^(list|show|what are)(?: the| my| all)? ?(locations|places|rooms|destinations)$", text):
            return ACTION_LIST_LOCATIONS
        # Get my location — primary trigger + natural variants
        _location_exact = {
            # Primary phrase and close variants
            "get my location", "get location", "get current location",
            "get my current location",
            # "tell me" variants
            "tell me my location", "tell me my current location",
            # "what is" variants
            "what is my location", "what is my current location",
            "whats my location", "whats my current location",
            # "show" / "find" my location
            "show my location", "show me my location",
            "find my location", "find my current location",
            # simple shorthands
            "my location", "current location", "location",
            # fallback: user may still say "where am I"
            "where am i", "where am i now",
        }
        if text in _location_exact:
            return ACTION_WHERE_AM_I
        # Keyword catch: any phrase containing "my location" or "get location"
        _nav_verbs = {"go", "navigate", "take", "lead", "find", "bring", "direct"}
        words = text.split()
        if any(kw in text for kw in ("my location", "current location",
                                     "get location", "get my")):
            if not any(w in words for w in _nav_verbs):
                return ACTION_WHERE_AM_I
        # Go home
        if re.match(r"^(go home|go to home|return home|take me home|home)$", text):
            return f"{ACTION_NAVIGATE}:home"

        # Navigate patterns
        nav_patterns = [
            r"^(?:go to|navigate to|take me to|lead me to|find|where is|where's) (?:the |a |an )?(.+?)(?: please)?$",
        ]
        for pat in nav_patterns:
            m = re.match(pat, text)
            if m:
                target_raw = m.group(1).strip()
                # Match against known locations (fuzzy)
                target_norm = target_raw.replace(" ", "_")
                if known_locations:
                    for loc in known_locations:
                        loc_l = loc.lower()
                        if target_norm == loc_l:
                            return f"{ACTION_NAVIGATE}:{loc}"
                    for loc in known_locations:
                        loc_l = loc.lower()
                        if target_norm in loc_l or loc_l in target_norm:
                            return f"{ACTION_NAVIGATE}:{loc}"
                return f"{ACTION_NAVIGATE}:{target_norm}"

        # Catch-all: any unmatched phrase still referencing "get" + "location"
        if any(kw in text for kw in ("get my", "get location", "my location",
                                     "current location", "where am i")):
            _nav_verbs = {"go", "navigate", "take", "lead", "find", "bring", "direct"}
            if not any(w in text.split() for w in _nav_verbs):
                print(f"[Voice] Catch-all WHERE_AM_I for: '{text}'")
                return ACTION_WHERE_AM_I

        return None
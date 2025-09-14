from email.mime import text
import os, sys, traceback, io, time, struct, tempfile, threading, queue
import re
import threading


INPUT_DEVICE_INDEX = 1          
SELECTED_INPUT_DEVICE = None    # will be set after detection
USE_CLOUD_TTS = True  # set True to use OpenAI TTS instead of pyttsx3
# Follow-up / session settings
SESSION_IDLE_TIMEOUT = 15       # seconds since last speech to auto-exit
FOLLOWUP_SECONDS = 6            # how long to listen per turn
STOP_PHRASES = {"no", "nope", "that's all", "thats all", "nothing", "we're good", "were good", "stop", "go to sleep", "shut up", "be quiet", "stop talking", "enough", "that's enough"}
# Half-duplex guard: true while Jarvis is speaking
# Global flag to allow stopping Jarvis
STOP_REQUESTED = False

SPEAKING = threading.Event()

print(">>> Starting Jarvis (diagnostic+fallback mode)...", flush=True)

# unbuffered logs
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
print(f">>> Has OPENAI_API_KEY: {bool(OPENAI_API_KEY)}", flush=True)
print(f">>> Has PICOVOICE_ACCESS_KEY: {bool(PICOVOICE_ACCESS_KEY)}", flush=True)

def hard_fail(msg): print(f"!!! {msg}", flush=True); sys.exit(1)

# imports
try:
    import sounddevice as sd
    import pvporcupine
    import pyttsx3
    import numpy as np
    from openai import OpenAI, RateLimitError
    print(">>> Imported core deps OK", flush=True)
except Exception as e:
    print("!!! Import error:", repr(e), flush=True); traceback.print_exc(); sys.exit(1)

# device list
try:
    devs = sd.query_devices()
    print(f">>> Found {len(devs)} audio devices. sd.default.device = {sd.default.device}", flush=True)
except Exception as e:
    print("!!! PortAudio error:", repr(e), flush=True); traceback.print_exc(); sys.exit(1)

if not OPENAI_API_KEY: hard_fail("Missing OPENAI_API_KEY in .env (next to jarvis.py)")
if not PICOVOICE_ACCESS_KEY: hard_fail("Missing PICOVOICE_ACCESS_KEY in .env")

# ---- runtime config ----
TTS_VOICE = (os.getenv("TTS_VOICE") or "").strip()
WAKEWORD = "jarvis"
RECORD_SECONDS = 7
RECORD_SAMPLERATE = 16000
RECORD_CHANNELS = 1

client = OpenAI(api_key=OPENAI_API_KEY)

def capture_and_transcribe(seconds=FOLLOWUP_SECONDS) -> str:
    while SPEAKING.is_set():
        time.sleep(0.05)
    
    wav = record_wav_bytes(seconds=seconds)
    try:
        text = transcribe_wav_bytes_cloud_then_local(wav)
    except Exception as e:
        log("Transcription fatal:", repr(e)); traceback.print_exc()
        return ""
    return (text or "").strip()

def run_conversation_session(initial_user_text: str):
    """
    Wake-word is paused. We speak answers (blocking), then listen for follow-ups.
    Ends on STOP_PHRASES or idle timeout.
    """
    last_turn_time = time.time()

    messages = [
        {"role": "system", "content": "You are Jarvis, a concise, helpful voice assistant. Keep answers short unless asked for detail. If you need clarification, ask a brief question."},
        {"role": "user", "content": initial_user_text}
    ]

    # First answer (blocking speak)
    reply = ask_gpt_chat(messages)
    tts_say(reply)
    messages.append({"role": "assistant", "content": reply})
    last_turn_time = time.time()

    while True:
        # (Optional) ask once if the last line wasn't a question
        if not reply.strip().endswith("?"):
            tts_say("Anything else?")

        # Now listen (SPEAKING is cleared because tts_say blocked)
        user_text = capture_and_transcribe(FOLLOWUP_SECONDS)
        now = time.time()

        # Exit if silence long enough
        if not user_text:
            if (now - last_turn_time) >= SESSION_IDLE_TIMEOUT:
                tts_say("Okay, I’m going back to standby.")
                break
            # brief silence -> try another listening window
            continue

        # Stop phrases
        ltxt = user_text.lower().replace("’","'").strip(" .!?")
        if ltxt in STOP_PHRASES:
            tts_say("Okay. Say Jarvis if you need me.")
            break

        # Continue chat
        messages.append({"role": "user", "content": user_text})
        reply = ask_gpt_chat(messages)
        messages.append({"role": "assistant", "content": reply})
        tts_say(reply)   # blocks; no listening during speech
        last_turn_time = time.time()


# ---- Keyboard listener for stopping Jarvis ----
def keyboard_listener():
    """Listen for Ctrl+C or other stop keys"""
    global STOP_REQUESTED
    try:
        import msvcrt
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'\x03':  # Ctrl+C
                    STOP_REQUESTED = True
                    break
                elif key == b'q':  # Press 'q' to quit
                    STOP_REQUESTED = True
                    break
    except:
        pass

# ---- Stop word checking during speech ----
def check_for_stop_words() -> bool:
    """Quick check for stop words while Jarvis is speaking."""
    global STOP_REQUESTED
    
    # Check global stop flag first
    if STOP_REQUESTED:
        return True
        
    try:
        # Quick recording to check for stop words
        import sounddevice as sd
        import numpy as np
        
        # Record a short sample
        duration = 0.5  # 500ms
        sample_rate = 16000
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        
        # Check if there's significant audio (someone speaking)
        if np.max(np.abs(audio)) > 1000:  # Threshold for speech detection
            # Quick transcription check
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    import wave
                    with wave.open(tmp.name, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sample_rate)
                        wf.writeframes(audio.tobytes())
                    
                    # Quick local transcription
                    from faster_whisper import WhisperModel
                    model = WhisperModel("tiny", device="cpu", compute_type="int8")
                    segments, _ = model.transcribe(tmp.name, language="en")
                    text = "".join(seg.text for seg in segments).strip().lower()
                    
                    # Check for stop words
                    for phrase in STOP_PHRASES:
                        if phrase in text:
                            return True
                    
                    os.remove(tmp.name)
            except:
                pass
    except:
        pass
    return False

# ---- OpenAI TTS (cloud -> WAV -> play) ----
def tts_say_cloud(text: str):
    """Speak via OpenAI TTS and block until playback completes."""
    try:
        model = os.getenv("OPENAI_TTS_MODEL", "tts-1")   # or gpt-4o-mini-tts
        voice = os.getenv("OPENAI_TTS_VOICE", "echo")

        # Mark speaking
        SPEAKING.set()

        # Try response_format first, fall back to format
        try:
            speech = client.audio.speech.create(
                model=model,
                voice=voice,
                input=str(text),
                response_format="wav",
            )
        except TypeError:
            speech = client.audio.speech.create(
                model=model,
                voice=voice,
                input=str(text),
                format="wav",
            )

        data = getattr(speech, "content", None)
        if data is None and hasattr(speech, "read"):
            data = speech.read()
        if data is None:
            data = bytes(speech)

        import simpleaudio as sa, tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(data)
            wav_path = tf.name

        wave_obj = sa.WaveObject.from_wave_file(wav_path)
        play_obj = wave_obj.play()
        
        # Check for stop words while playing
        while play_obj.is_playing():
            time.sleep(0.1)
            if check_for_stop_words():
                play_obj.stop()
                log("🛑 Speech interrupted by stop word")
                break

    except Exception as e:
        print("Cloud TTS failed:", repr(e), flush=True)
        # best effort local fallback (still blocking)
        try:
            _engine.stop()
            _engine.say(str(text))
            _engine.runAndWait()
        except:
            pass
    finally:
        # Clear speaking + cleanup file
        SPEAKING.clear()
        try:
            os.remove(wav_path)
        except:
            pass



# ---- TTS: serialize with a queue so pyttsx3 never overlaps ----
import pyttsx3, re, queue, threading, time, os
def _pick_voice(engine, preferred=None):
    voices = engine.getProperty("voices")
    if preferred:
        for v in voices:
            if preferred.lower() in (v.id or "").lower() or preferred.lower() in (v.name or "").lower():
                return v.id
    for v in voices:
        label = f"{v.id} {v.name}".lower()
        if "en-" in label or "english" in label:
            return v.id
    return voices[0].id if voices else None

try:
    _engine = pyttsx3.init(driverName="sapi5")  
except Exception:
    _engine = pyttsx3.init()                    

# voice, rate, volume
PREFERRED_VOICE = os.getenv("TTS_VOICE", "David")  
vid = _pick_voice(_engine, PREFERRED_VOICE)
if vid: _engine.setProperty("voice", vid)
_engine.setProperty("rate", 185)
_engine.setProperty("volume", 1.0)

_tts_q = queue.Queue()

def _tts_worker():
    while True:
        phrase = _tts_q.get()
        try:
            print(f"[TTS] speaking: {phrase[:80]}...", flush=True)
            _engine.stop()
            _engine.say(phrase)
            _engine.runAndWait()
        except Exception as e:
            print("TTS error:", repr(e), flush=True)
        finally:
            _tts_q.task_done()

threading.Thread(target=_tts_worker, daemon=True).start()

def tts_say(text: str):
    msg = str(text)
    print(f"[TTS] {msg[:80]}...", flush=True)
    if USE_CLOUD_TTS:
        tts_say_cloud(msg) 
    else:
       
        try:
            SPEAKING.set()
            _engine.stop()
            _engine.say(msg)
            
           
            while _engine.isBusy():
                time.sleep(0.1)
                if check_for_stop_words():
                    _engine.stop()
                    log("🛑 Local speech interrupted by stop word")
                    break
                    
        finally:
            SPEAKING.clear()


def log(*a): print(*a, flush=True)

# ---- mic selection ----
def select_input_device():
    global INPUT_DEVICE_INDEX, SELECTED_INPUT_DEVICE
    devices = sd.query_devices()
    if INPUT_DEVICE_INDEX is not None:
        sd.check_input_settings(device=INPUT_DEVICE_INDEX)
        dev = sd.query_devices(INPUT_DEVICE_INDEX)
        SELECTED_INPUT_DEVICE = INPUT_DEVICE_INDEX
        sd.default.device = (SELECTED_INPUT_DEVICE, None)
        log(f"🎤 Using forced input device #{SELECTED_INPUT_DEVICE}: {dev['name']}")
        return SELECTED_INPUT_DEVICE
    default = sd.default.device
    try: default_in = default[0]
    except Exception: default_in = getattr(default, "input", None)
    if isinstance(default_in, int) and default_in >= 0:
        sd.check_input_settings(device=default_in)
        dev = sd.query_devices(default_in)
        SELECTED_INPUT_DEVICE = default_in
        sd.default.device = (SELECTED_INPUT_DEVICE, None)
        log(f"🎤 Using default input device #{SELECTED_INPUT_DEVICE}: {dev['name']}")
        return SELECTED_INPUT_DEVICE
    for i, d in enumerate(devices):
        if d.get('max_input_channels', 0) > 0:
            sd.check_input_settings(device=i)
            SELECTED_INPUT_DEVICE = i
            sd.default.device = (SELECTED_INPUT_DEVICE, None)
            log(f"🎤 Using discovered input device #{SELECTED_INPUT_DEVICE}: {d['name']}")
            return SELECTED_INPUT_DEVICE
    raise RuntimeError("No input-capable devices found.")

# ---- record helper ----
def record_wav_bytes(seconds=RECORD_SECONDS, samplerate=RECORD_SAMPLERATE, channels=RECORD_CHANNELS):
    import wave
    sd.default.samplerate = samplerate
    sd.default.channels = channels
    log("stage=record")
    log(f"🎙️  Recording {seconds}s @ {samplerate}Hz using device #{SELECTED_INPUT_DEVICE} ...")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=channels, dtype='int16',
                   device=SELECTED_INPUT_DEVICE)
    sd.wait()
    peak = int(audio.max()) if audio.size else 0
    log(f"   → samples: {audio.size}, peak: {peak}")
    buf = io.BytesIO()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        # write a simple WAV header + PCM to tmp to avoid py wave module in memory
        import wave as _wave
        with _wave.open(tmp, 'wb') as wf:
            wf.setnchannels(channels); wf.setsampwidth(2); wf.setframerate(samplerate)
            wf.writeframes(audio.tobytes())
        tmp.flush()
        with open(tmp.name, "rb") as f:
            buf.write(f.read())
    buf.seek(0)
    return buf

# ---- STT: cloud first, local fallback ----
def transcribe_wav_bytes_cloud_then_local(wav_bytes: io.BytesIO) -> str:
    # try OpenAI Whisper
    wav_bytes.seek(0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes.read()); tmp_path = tmp.name
    try:
        log("stage=transcribe (cloud)")
        log("📝 Whisper: sending audio ...")
        with open(tmp_path, "rb") as f:
            resp = client.audio.transcriptions.create(model="whisper-1", file=f)
        text = (getattr(resp, "text", None) or "").strip()
        log("📝 Whisper text:", text)
        return text
    except RateLimitError as e:
        log("Whisper API quota/rate limit:", repr(e))
        tts_say("Speech service is at its limit. Falling back to local transcription.")
        return transcribe_wav_bytes_local(tmp_path)
    except Exception as e:
        log("Whisper API error:", repr(e))
        return transcribe_wav_bytes_local(tmp_path)
    finally:
        try: os.remove(tmp_path)
        except: pass

def transcribe_wav_bytes_local(wav_path: str) -> str:
    log("stage=transcribe (local faster-whisper)")
    try:
        from faster_whisper import WhisperModel
    except Exception:
        tts_say("Local speech engine not installed. Run: pip install faster-whisper")
        return ""
    
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(wav_path, language="en", vad_filter=True)
    text = "".join(seg.text for seg in segments).strip()
    log("📝 Local Whisper text:", text)
    return text

def ask_gpt_chat(messages) -> str:
    """
    messages = [{"role":"system"/"user"/"assistant","content": "..."}]
    """
    log("stage=think")
    model = os.getenv("OPENAI_MODEL", "gpt-5")
    try:
        resp = client.chat.completions.create(model=model, messages=messages)
    except Exception as e:
        log("GPT error (first attempt):", repr(e))
        fallback_model = "gpt-5-thinking"
        if model != fallback_model:
            log(f"Retrying with fallback model: {fallback_model}")
            resp = client.chat.completions.create(model=fallback_model, messages=messages)
        else:
            raise
    ans = resp.choices[0].message.content.strip()
    log("🤖 Jarvis:", ans)
    return ans

# ---- LLM ----
def ask_gpt5(txt: str) -> str:
    log("stage=think")
    model = os.getenv("OPENAI_MODEL", "gpt-5")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are Jarvis, concise and helpful. Keep answers short unless asked for detail."},
                {"role": "user", "content": txt}
            ]
        )
    except Exception as e:
        log("GPT error (first attempt):", repr(e))
        fallback_model = "gpt-5-thinking"
        if model != fallback_model:
            log(f"Retrying with fallback model: {fallback_model}")
            resp = client.chat.completions.create(
                model=fallback_model,
                messages=[
                    {"role": "system", "content": "You are Jarvis, concise and helpful. Keep answers short unless asked for detail."},
                    {"role": "user", "content": txt}
                ]
            )
        else:
            raise
    ans = resp.choices[0].message.content.strip()
    log("🤖 Jarvis:", ans)
    return ans



# ---- wake-word loop ----
is_listening = True
is_busy = False
input_stream = None

def wakeword_loop():
    global input_stream
    device_index = select_input_device()
    porcupine = pvporcupine.create(access_key=PICOVOICE_ACCESS_KEY, keywords=[WAKEWORD])
    try:
        input_stream = sd.RawInputStream(
            samplerate=porcupine.sample_rate,
            blocksize=porcupine.frame_length,
            dtype='int16',
            channels=1,
            device=device_index,
            callback=lambda indata, frames, time_info, status: wakeword_callback(indata, porcupine)
        )
        input_stream.start()
        log("🔊 Say 'Jarvis' to start. Running in background.")
        while True:
            time.sleep(0.1)
    finally:
        if input_stream:
            try: input_stream.stop(); input_stream.close()
            except: pass
        porcupine.delete()

def wakeword_callback(indata, porcupine):
    global is_listening, is_busy
    if not is_listening or is_busy: return
    pcm = struct.unpack_from("h" * porcupine.frame_length, indata)
    if porcupine.process(pcm) >= 0:
        is_busy = True
        threading.Thread(target=handle_interaction, daemon=True).start()

def handle_interaction():
    global is_busy, input_stream
    try:
        # Pause wake-word so recorder owns the mic for the whole session
        if input_stream:
            try: input_stream.stop(); log("⏸️  Wake-word paused")
            except Exception as e: log("Pause stream error:", repr(e))

        # Brief cue
        tts_say("Yes?")
        time.sleep(0.2)

        # First user utterance
        first_text = capture_and_transcribe(seconds=RECORD_SECONDS)
        if not first_text:
            tts_say("I didn't catch that.")
            return

        # Check for immediate stop commands
        first_text_lower = first_text.lower().strip()
        if any(phrase in first_text_lower for phrase in ["stop", "shut up", "be quiet", "enough", "go away"]):
            tts_say("Okay, I'm going back to standby.")
            return

        # Run interactive session (handles follow-ups)
        run_conversation_session(first_text)

    except Exception as e:
        log("Interaction error:", repr(e)); traceback.print_exc()
        tts_say("I hit an error.")
    finally:
        # Resume wake-word
        if input_stream:
            try: input_stream.start(); log("▶️  Wake-word resumed")
            except Exception as e: log("Failed to restart input stream:", repr(e))
        is_busy = False


# ---- main ----
if __name__ == "__main__":
    try:
        print("🔊 Jarvis is starting... Press Ctrl+C or 'q' to stop at any time.")
        print("💡 You can also say 'stop', 'shut up', 'be quiet', or 'enough' to interrupt Jarvis while speaking.")
        
        # Start keyboard listener in background
        keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
        keyboard_thread.start()
        
        wakeword_loop()
    except KeyboardInterrupt:
        print("\n🛑 Jarvis stopped by user.")
        if input_stream:
            try: input_stream.stop(); input_stream.close()
            except: pass
    except SystemExit:
        pass
    except Exception as e:
        print("!!! Fatal error:", repr(e), flush=True); traceback.print_exc(); sys.exit(1)

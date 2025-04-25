import pyaudio
import wave
import numpy as np
import tempfile
import os
from faster_whisper import WhisperModel
import keyboard
import threading
import queue
import time
from collections import deque
from deep_translator import GoogleTranslator

def get_input_device():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    print("\nAvailable Audio Input Devices:")
    print("-" * 30)
    for i in range(numdevices):
        device_info = p.get_device_info_by_index(i)
        if device_info.get('maxInputChannels') > 0:
            print(f"Device {i}: {device_info.get('name')}")
    
    default_device = p.get_default_input_device_info()
    print(f"\nUsing default input device: {default_device.get('name')}")
    return default_device.get('index')

class AudioTranscriber:
    def __init__(self):
        self.CHUNK = 1024 * 2
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.BUFFER_TIME = 2  # Buffer 2 seconds of audio
        self.running = False
        self.audio_queue = queue.Queue()
        self.buffer = deque(maxlen=int(self.RATE * self.BUFFER_TIME / self.CHUNK))
        
        print("Loading Whisper model...")
        # Using medium model for better Arabic recognition
        self.model = WhisperModel("medium", device="cpu", compute_type="int8")
        
        # Initialize translator
        self.translator = GoogleTranslator(source='ar', target='en')
        
        self.p = pyaudio.PyAudio()
        self.input_device_index = get_input_device()

    def audio_callback(self, in_data, frame_count, time_info, status):
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)

    def process_audio(self):
        temp_dir = tempfile.mkdtemp()
        temp_count = 0

        while self.running:
            # Collect audio chunks until we have enough for our buffer
            chunks = []
            while len(chunks) < int(self.RATE * 2 / self.CHUNK):  # 2 seconds of audio
                try:
                    chunk = self.audio_queue.get(timeout=1)
                    chunks.append(chunk)
                except queue.Empty:
                    continue

            if not chunks:
                continue

            # Save audio chunks to temporary WAV file
            temp_filename = os.path.join(temp_dir, f'temp_{temp_count}.wav')
            temp_count = (temp_count + 1) % 10  # Rotate through 10 temp files

            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(chunks))

            try:
                # Transcribe the audio with Arabic language hint
                segments, info = self.model.transcribe(
                    temp_filename,
                    beam_size=5,
                    language='ar',
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=300,
                        speech_pad_ms=100
                    )
                )

                # Print transcription and translation if speech was detected
                segments = list(segments)
                if segments:
                    for segment in segments:
                        arabic_text = segment.text.strip()
                        if arabic_text:
                            try:
                                english_text = self.translator.translate(arabic_text)
                                print(f"\n🎤 Arabic: {arabic_text}")
                                print(f"🔄 English: {english_text}")
                                print("-" * 50)
                            except Exception as e:
                                print(f"\nTranslation error: {e}")

            except Exception as e:
                print(f"\nTranscription error: {e}")

            try:
                os.remove(temp_filename)
            except:
                pass

    def start(self):
        self.running = True
        
        # Start audio stream
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.audio_callback
        )

        print("\nStarted continuous Arabic transcription and translation.")
        print("Speak in Arabic, and you'll see both Arabic text and English translation.")
        print("Press 'q' to quit...")
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.start()

        # Wait for 'q' key to stop
        while self.running and not keyboard.is_pressed('q'):
            time.sleep(0.1)

        self.stop()

    def stop(self):
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        if hasattr(self, 'process_thread'):
            self.process_thread.join()

if __name__ == "__main__":
    # Set environment variable to handle OpenMP runtime error
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    transcriber = AudioTranscriber()
    try:
        transcriber.start()
    except KeyboardInterrupt:
        transcriber.stop()
    print("\nTranscription and translation ended.")
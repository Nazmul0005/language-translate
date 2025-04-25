import logging
from faster_whisper import WhisperModel

# Setup logging for better visibility
logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

def transcribe_audio(audio_path: str):
    """
    Transcribe audio using Faster Whisper with optimized settings
    """
    # Initialize model - using small model for faster processing
    print("Loading model...")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    
    print(f"\nTranscribing file: {audio_path}")
    # Run transcription with VAD filter
    segments, info = model.transcribe(
        audio_path,
        beam_size=1,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=100
        )
    )
    
    # Process results
    segments = list(segments)
    
    print(f"\nTranscription Results:")
    print(f"Detected language: {info.language} ({info.language_probability:.2f})")
    print("-" * 50)
    
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

if __name__ == "__main__":
    # Using the JFK speech sample from the test data
    AUDIO_FILE = "tests/data/jfk.flac"
    transcribe_audio(AUDIO_FILE)
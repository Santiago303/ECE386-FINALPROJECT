import sounddevice as sd
import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline
import sys
import time
import wave
import os
from scipy import signal

def record_audio(duration_seconds: int = 5) -> npt.NDArray:
    """Record duration_seconds of audio from default microphone.
    Return a single channel numpy array."""
    sample_rate = 16000  # Hz
    samples = int(duration_seconds * sample_rate)
    print(f"Recording for {duration_seconds} seconds...")
    
    # Will use default microphone; on Jetson this is likely a USB WebCam
    audio = sd.rec(samples, samplerate=sample_rate, channels=1, dtype=np.float32)
    # Blocks until recording complete
    sd.wait()
    # Model expects single axis
    return np.squeeze(audio)

def build_pipeline(model_id: str) -> Pipeline:
    """Creates a Hugging Face automatic-speech-recognition pipeline on GPU."""
    # Force GPU usage if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"Loading model {model_id} on {device} with {torch_dtype}...")
    
    if device == "cpu":
        print("WARNING: GPU not available! Falling back to CPU.")
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    print(f"Pipeline built successfully on {device}")
    return pipe

# Global pipeline to avoid reloading the model
_pipeline = None

def get_pipeline():
    """Get or create the speech recognition pipeline"""
    global _pipeline
    if _pipeline is None:
        model_id = "distil-whisper/distil-medium.en"
        _pipeline = build_pipeline(model_id)
    return _pipeline

def transcribe_audio(audio_file_path):
    """Transcribe audio from a file path"""
    print(f"Transcribing audio from file: {audio_file_path}")
    
    # Get the pipeline
    pipe = get_pipeline()
    
    # Load audio from file
    with wave.open(audio_file_path, 'rb') as wf:
        # Get audio parameters
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        
        # Convert to numpy array
        if sample_width == 2:  # 16-bit audio
            dtype = np.int16
        elif sample_width == 4:  # 32-bit audio
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        audio_np = np.frombuffer(frames, dtype=dtype)
        
        # Convert to float32 in range [-1, 1]
        audio_np = audio_np.astype(np.float32) / (2**(8 * sample_width - 1))
        
        # If stereo, convert to mono
        if channels == 2:
            audio_np = audio_np.reshape(-1, 2).mean(axis=1)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        audio_np = signal.resample(audio_np, int(len(audio_np) * 16000 / sample_rate))
    
    # Transcribe
    print("Processing audio with Whisper model...")
    start_time = time.time()
    result = pipe(audio_np)
    end_time = time.time()
    
    print(f"Transcription completed in {end_time - start_time:.2f} seconds")
    print(f"Transcription: {result['text']}")
    
    # Return the transcription text
    return result["text"]

def capture_and_transcribe(duration=5):
    """Record audio and transcribe it"""
    print(f"Starting audio capture for {duration} seconds...")
    
    # Record audio
    audio = record_audio(duration)
    
    # Get the pipeline
    pipe = get_pipeline()
    
    # Transcribe
    print("Processing audio with Whisper model...")
    start_time = time.time()
    result = pipe(audio)
    end_time = time.time()
    
    print(f"Transcription completed in {end_time - start_time:.2f} seconds")
    print(f"Transcription: {result['text']}")
    
    # Return the transcription text
    return result["text"]

def save_audio_to_file(audio_data, filename="recorded_audio.wav", sample_rate=16000):
    """Save audio data to a WAV file"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        
        # Convert float32 to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wf.writeframes(audio_int16.tobytes())
    
    print(f"Audio saved to {filename}")
    return filename

if __name__ == "__main__":
    # Check for GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: GPU not available! Using CPU instead.")
    
    # Get model as argument, default to "distil-whisper/distil-medium.en" if not given
    model_id = sys.argv[1] if len(sys.argv) > 1 else "distil-whisper/distil-medium.en"
    print(f"Using model_id {model_id}")
    
    # Build the pipeline
    pipe = build_pipeline(model_id)
    
    # Test recording and transcription
    print("Recording...")
    audio = record_audio(5)  # 5 seconds of audio
    print("Recording complete")
    
    # Save audio to file
    audio_file = save_audio_to_file(audio)
    
    # Transcribe from memory
    print("Transcribing from memory...")
    start_time = time.time()
    result_memory = pipe(audio)
    end_time = time.time()
    print(f"Transcription from memory: {result_memory['text']}")
    print(f"Transcription took {end_time - start_time:.2f} seconds")
    
    # Transcribe from file
    print("Transcribing from file...")
    result_file = transcribe_audio(audio_file)
    print(f"Transcription from file: {result_file}")
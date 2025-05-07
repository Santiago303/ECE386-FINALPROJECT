from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import tempfile
import sounddevice as sd
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

app = FastAPI()

# Load the model once at startup
def load_model():
    model_id = "distil-whisper/distil-medium.en"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"Loading model {model_id} on {device}...")
    
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
    
    return pipe

# Load the model at startup
transcription_pipeline = load_model()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(await file.read())
        
        # Process the audio file
        import wave
        with wave.open(temp_file_path, 'rb') as wf:
            # Get audio parameters
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            
            # Read all frames
            frames = wf.readframes(n_frames)
        
        # Convert to numpy array
        import numpy as np
        if sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.float32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        audio = np.frombuffer(frames, dtype=dtype)
        
        # If stereo, convert to mono
        if channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        
        # Normalize if int16
        if dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        
        # Transcribe
        result = transcription_pipeline(audio)
        transcription = result["text"]
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        return JSONResponse(content={"text": transcription})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
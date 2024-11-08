from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import librosa
import torch
import os

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://127.0.0.1:3000",  # Local testing
        "https://genuine-halva-146908.netlify.app",  # Netlify frontend
        "https://f5ee-110-174-242-7.ngrok-free.app"  # Replace with actual ngrok URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and processor
model_path = "/Users/sabinghimire/Downloads/wav2vec2_api/fine_yolngu_model"
processor_path = "/Users/sabinghimire/Downloads/wav2vec2_api/fine_processor"
model = Wav2Vec2ForCTC.from_pretrained(model_path)
processor = Wav2Vec2Processor.from_pretrained(processor_path)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        if file.content_type not in ["audio/wav", "audio/mp3"]:
            raise HTTPException(status_code=400, detail="Invalid file format. Only WAV and MP3 are supported.")
        
        # Load and resample audio to 16kHz
        audio_data, sr = librosa.load(file.file, sr=16000)

        # Process the audio data with the model's processor
        input_values = processor(audio_data, return_tensors="pt", sampling_rate=16000).input_values

        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits
        
        # Decode the model output
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        
        return {"transcription": transcription[0]}
    
    except Exception as e:
        print("Error during transcription:", e)
        raise HTTPException(status_code=500, detail="Error during transcription")

# Serve the React build's static files
app.mount("/static", StaticFiles(directory="build/static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("build/index.html")

# Catch-all route to serve `index.html` for all other routes (for React Router compatibility)
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    return FileResponse("build/index.html")

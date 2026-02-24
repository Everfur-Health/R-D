#!/usr/bin/env python3
"""
Canine Cough Detection - Web API Server

This FastAPI server provides:
1. Audio file upload and analysis
2. Frame-level predictions (where in the audio events occur)
3. Visualization data (waveform, spectrogram, heatmap)
4. Static file serving for the frontend

USAGE:
======
    # Start the server (from the python/ directory)
    cd audio_health_detection/python
    python -m web.server
    
    # Or run directly
    cd audio_health_detection/python/web
    python server.py
    
    # Then open http://localhost:8000 in your browser
    # API documentation at http://localhost:8000/docs
"""

import io
import sys
import base64
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================
# Fix Python Path for Imports
# ============================================
# Add the parent directory (python/) to the path so we can import our modules
SCRIPT_DIR = Path(__file__).parent.resolve()
PYTHON_DIR = SCRIPT_DIR.parent.resolve()
PROJECT_DIR = PYTHON_DIR.parent.resolve()

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

# Now we can import our modules
from preprocessing import AudioPreprocessor
from model import CoughClassifier

# Optional: matplotlib for spectrogram images
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not found - spectrogram images disabled")


# ============================================
# Configuration
# ============================================

class Config:
    """Server configuration."""
    HOST = "0.0.0.0"
    PORT = 8000
    
    # Model path (relative to project root)
    MODEL_PATH = PROJECT_DIR / "models" / "checkpoints" / "latest" / "best_model.pth"
    
    # Audio settings (must match training)
    SAMPLE_RATE = 16000
    N_MELS = 64
    N_FFT = 1024
    HOP_LENGTH = 320
    TARGET_DURATION = 5.0  # seconds
    
    # Analysis settings
    WINDOW_SIZE = 1.0      # Analysis window in seconds
    WINDOW_STRIDE = 0.25   # Stride between windows in seconds
    
    # Class info
    CLASSES = ['healthy', 'kennel_cough', 'heart_failure_cough', 'other_respiratory']
    CLASS_COLORS = {
        'healthy': '#4CAF50',           # Green
        'kennel_cough': '#F44336',      # Red
        'heart_failure_cough': '#9C27B0',  # Purple
        'other_respiratory': '#FF9800',  # Orange
    }

config = Config()


# ============================================
# Response Models (Pydantic)
# ============================================

class SegmentPrediction(BaseModel):
    """A prediction for a segment of audio."""
    start_time: float
    end_time: float
    label: str
    confidence: float
    color: str
    all_probabilities: Dict[str, float]


class ClassificationResult(BaseModel):
    """Classification result for the entire audio."""
    label: str
    confidence: float
    all_probabilities: Dict[str, float]


class AnalysisResponse(BaseModel):
    """Complete analysis response."""
    success: bool
    filename: str
    duration: float
    sample_rate: int
    
    # Overall classification (entire clip)
    classification: ClassificationResult
    
    # Timeline of predictions (for heatmap)
    segments: List[SegmentPrediction]
    
    # Visualization data
    waveform: List[float]
    waveform_times: List[float]
    spectrogram_base64: Optional[str] = None
    
    # Processing info
    processing_time_ms: float


class ModelInfo(BaseModel):
    """Model metadata."""
    model_loaded: bool
    model_path: str
    classes: List[str]
    device: str
    sample_rate: int


# ============================================
# Global State (loaded on startup)
# ============================================

model: Optional[CoughClassifier] = None
preprocessor: Optional[AudioPreprocessor] = None
device: torch.device = torch.device('cpu')


def load_model():
    """Load the trained model."""
    global model, preprocessor, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=config.SAMPLE_RATE,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    
    # Load model
    print(f"Looking for model at: {config.MODEL_PATH}")
    if config.MODEL_PATH.exists():
        print(f"Loading model from {config.MODEL_PATH}")
        model = CoughClassifier.load_checkpoint(str(config.MODEL_PATH), device=str(device))
        model.eval()
        print("✓ Model loaded successfully")
    else:
        print(f"⚠ Model not found at {config.MODEL_PATH}")
        print("  Run training first: cd python && python train.py")
        model = None


# ============================================
# Analysis Functions
# ============================================

def analyze_audio_file(audio_data: bytes, filename: str) -> AnalysisResponse:
    """
    Analyze an uploaded audio file.
    
    This function:
    1. Loads and preprocesses the audio
    2. Runs overall classification on the entire clip
    3. Runs sliding window analysis to get time-localized predictions
    4. Generates visualization data
    """
    import time
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Save to temp file (librosa needs a file path)
    suffix = Path(filename).suffix or '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name
    
    try:
        # Load audio
        audio = preprocessor.load_audio(tmp_path)
        duration = len(audio) / config.SAMPLE_RATE
        
        # ====================================
        # Overall Classification
        # ====================================
        target_samples = int(config.TARGET_DURATION * config.SAMPLE_RATE)
        audio_padded = preprocessor.pad_or_trim(audio, target_samples)
        mel_spec = preprocessor.extract_mel_spectrogram(audio_padded)
        
        # Run inference
        input_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)
        overall_prediction = model.predict(input_tensor)
        
        # ====================================
        # Sliding Window Analysis (for heatmap)
        # ====================================
        segments = []
        window_samples = int(config.WINDOW_SIZE * config.SAMPLE_RATE)
        stride_samples = int(config.WINDOW_STRIDE * config.SAMPLE_RATE)
        
        position = 0
        while position + window_samples <= len(audio):
            # Extract window
            window_audio = audio[position:position + window_samples]
            
            # Pad to target duration for model
            window_padded = preprocessor.pad_or_trim(window_audio, target_samples)
            window_mel = preprocessor.extract_mel_spectrogram(window_padded)
            
            # Run inference
            window_tensor = torch.FloatTensor(window_mel).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(window_tensor)
                probs = output['probabilities'][0].cpu().numpy()
            
            # Get prediction for this window
            pred_idx = int(probs.argmax())
            pred_label = config.CLASSES[pred_idx]
            pred_conf = float(probs[pred_idx])
            
            start_sec = position / config.SAMPLE_RATE
            end_sec = (position + window_samples) / config.SAMPLE_RATE
            
            segments.append(SegmentPrediction(
                start_time=start_sec,
                end_time=end_sec,
                label=pred_label,
                confidence=pred_conf,
                color=config.CLASS_COLORS.get(pred_label, '#888888'),
                all_probabilities={
                    config.CLASSES[i]: float(probs[i])
                    for i in range(len(config.CLASSES))
                }
            ))
            
            position += stride_samples
        
        # Handle final partial window
        if position < len(audio) and len(audio) - position > stride_samples:
            window_audio = audio[position:]
            window_padded = preprocessor.pad_or_trim(window_audio, target_samples)
            window_mel = preprocessor.extract_mel_spectrogram(window_padded)
            
            window_tensor = torch.FloatTensor(window_mel).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(window_tensor)
                probs = output['probabilities'][0].cpu().numpy()
            
            pred_idx = int(probs.argmax())
            pred_label = config.CLASSES[pred_idx]
            
            segments.append(SegmentPrediction(
                start_time=position / config.SAMPLE_RATE,
                end_time=len(audio) / config.SAMPLE_RATE,
                label=pred_label,
                confidence=float(probs[pred_idx]),
                color=config.CLASS_COLORS.get(pred_label, '#888888'),
                all_probabilities={
                    config.CLASSES[i]: float(probs[i])
                    for i in range(len(config.CLASSES))
                }
            ))
        
        # ====================================
        # Generate Visualization Data
        # ====================================
        
        # Downsample waveform for efficient transfer
        target_points = 1000
        step = max(1, len(audio) // target_points)
        waveform_downsampled = audio[::step].tolist()
        waveform_times = [i * step / config.SAMPLE_RATE for i in range(len(waveform_downsampled))]
        
        # Generate spectrogram image
        spectrogram_base64 = None
        if HAS_MATPLOTLIB:
            spectrogram_base64 = generate_spectrogram_image(audio)
        
        processing_time = (time.time() - start_time) * 1000
        
        return AnalysisResponse(
            success=True,
            filename=filename,
            duration=duration,
            sample_rate=config.SAMPLE_RATE,
            classification=ClassificationResult(
                label=overall_prediction['label'],
                confidence=overall_prediction['confidence'],
                all_probabilities=overall_prediction['all_probabilities'],
            ),
            segments=segments,
            waveform=waveform_downsampled,
            waveform_times=waveform_times,
            spectrogram_base64=spectrogram_base64,
            processing_time_ms=processing_time,
        )
        
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


def generate_spectrogram_image(audio: np.ndarray) -> str:
    """Generate a spectrogram image as base64 PNG."""
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Compute mel spectrogram
    mel_spec = preprocessor.extract_mel_spectrogram(audio, normalize=False)
    
    # Plot
    img = ax.imshow(
        mel_spec,
        aspect='auto',
        origin='lower',
        cmap='magma',
        extent=[0, len(audio) / config.SAMPLE_RATE, 0, config.N_MELS]
    )
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mel Bin')
    ax.set_title('Mel Spectrogram')
    plt.colorbar(img, ax=ax, label='Magnitude')
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


# ============================================
# FastAPI Application
# ============================================

app = FastAPI(
    title="Canine Cough Detection API",
    description="AI-powered analysis of dog respiratory sounds",
    version="1.0.0",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model on server startup."""
    load_model()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main page."""
    index_path = SCRIPT_DIR / "static" / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    else:
        return HTMLResponse(content="""
        <html>
        <head><title>Canine Cough Detection</title></head>
        <body style="font-family: sans-serif; max-width: 600px; margin: 50px auto; padding: 20px;">
            <h1>🐕 Canine Cough Detection API</h1>
            <p>Frontend not found at static/index.html</p>
            <p>API documentation: <a href="/docs">/docs</a></p>
        </body>
        </html>
        """)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/model-info", response_model=ModelInfo)
async def model_info():
    """Get model information."""
    return ModelInfo(
        model_loaded=model is not None,
        model_path=str(config.MODEL_PATH),
        classes=config.CLASSES,
        device=str(device),
        sample_rate=config.SAMPLE_RATE,
    )


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze an uploaded audio file.
    
    Accepts WAV, MP3, OGG, FLAC, M4A formats.
    
    Returns:
    - Overall classification of the audio
    - Time-segmented predictions (for visualization)
    - Waveform data
    - Spectrogram image (base64)
    """
    # Validate file type
    allowed_extensions = {'.wav', '.mp3', '.ogg', '.flac', '.m4a'}
    suffix = Path(file.filename).suffix.lower()
    
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {allowed_extensions}"
        )
    
    # Read file
    audio_data = await file.read()
    
    if len(audio_data) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    
    if len(audio_data) > 50 * 1024 * 1024:  # 50 MB limit
        raise HTTPException(status_code=400, detail="File too large (max 50 MB)")
    
    # Analyze
    try:
        result = analyze_audio_file(audio_data, file.filename)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Mount static files
static_dir = SCRIPT_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ============================================
# Entry Point
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🐕 Canine Cough Detection - Web Server")
    print("=" * 60)
    print(f"\nProject directory: {PROJECT_DIR}")
    print(f"Python directory: {PYTHON_DIR}")
    print(f"Model path: {config.MODEL_PATH}")
    print(f"\nStarting server at http://{config.HOST}:{config.PORT}")
    print(f"API docs at http://{config.HOST}:{config.PORT}/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        app,  # Pass the app object directly instead of string
        host=config.HOST,
        port=config.PORT,
    )

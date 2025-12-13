"""
CoquiTTS Configuration Example

This file shows various configuration options for CoquiTTS.
Copy this to config.py and modify as needed.
"""

# Model Configuration
MODEL_CONFIG = {
    # Default model (fast, good quality)
    "default": None,  # Uses TTS default
    
    # High-quality models (slower but better)
    "high_quality": "tts_models/en/ljspeech/tacotron2-DDC",
    
    # Fast models (faster but lower quality)
    "fast": "tts_models/en/ljspeech/glow-tts",
    
    # Multi-speaker models
    "multi_speaker": "tts_models/en/vctk/vits",
}

# Output Configuration
OUTPUT_CONFIG = {
    "default_format": "wav",
    "sample_rate": 22050,  # Standard sample rate
    "output_directory": "outputs",
}

# Synthesis Settings
SYNTHESIS_CONFIG = {
    "progress_bar": True,
    "gpu": False,  # Set to True if you have CUDA GPU
}

# Example usage in your code:
"""
from TTS.api import TTS
import config

# Use configured model
tts = TTS(
    model_name=config.MODEL_CONFIG["high_quality"],
    progress_bar=config.SYNTHESIS_CONFIG["progress_bar"],
    gpu=config.SYNTHESIS_CONFIG["gpu"]
)

# Synthesize with configured settings
tts.tts_to_file(
    text="Your text here",
    file_path=f"{config.OUTPUT_CONFIG['output_directory']}/output.wav"
)
"""

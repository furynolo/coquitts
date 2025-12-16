# CoquiTTS Setup

This project contains a setup for CoquiTTS, a text-to-speech synthesis system.

## Project Structure

```
coquitts/
├── book-to-audiobook_coquitts.py  # Main conversion script
├── requirements.txt               # Python dependencies
├── README.md                     # This file
├── docs/                         # Documentation
│   ├── ESPEAK_INSTALL.md
│   ├── PYTORCH_FIX.md
│   └── WINDOWS_SETUP.md
├── scripts/                      # Utility scripts
│   ├── list_models.py
│   ├── setup.py
│   ├── config_example.py
│   └── enable_long_paths.ps1
├── input-text/                   # Input text files
└── output-audio/                 # Generated audio files
```

## Installation

### ⚠️ Windows Users: Long Path Support Required

**If you encounter path length errors during installation**, you need to enable Windows Long Path Support first. See `docs/WINDOWS_SETUP.md` for detailed instructions, or run:

```powershell
# Run PowerShell as Administrator
.\scripts\enable_long_paths.ps1
```

After enabling and restarting, continue with the installation below.

### 1. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** The first time you run CoquiTTS, it will download model files automatically. This may take a few minutes depending on your internet connection.

**Troubleshooting:** If you still encounter path issues, try:
```bash
pip install --no-cache-dir -r requirements.txt
```

### 3. Verify Installation

Run the setup verification script:

```bash
python scripts/setup.py
```

Or manually verify:

```bash
tts --help
```

## Usage

### Command Line Interface

Generate speech from text:

```bash
tts --text "Hello, this is a test." --out_path output.wav
```

### Python API

Convert a text file to audiobook:

```bash
python book-to-audiobook_coquitts.py input-text/mybook.txt
```

The output file will be saved as `output-audio/mybook.wav` (matching the input filename).

**Optional:** Specify a different model:

```bash
python book-to-audiobook_coquitts.py input-text/mybook.txt tts_models/en/ljspeech/glow-tts
```

**Note:** The script uses `tts_models/en/ljspeech/tacotron2-DDC` by default. If you encounter model errors, you can:

1. List all available models:
   ```bash
   python scripts/list_models.py
   ```

2. Use a different model via command line (see above)

See `scripts/config_example.py` for configuration options.

### Multi-Speaker Models

Some models support multiple speakers. For example, the `tts_models/en/vctk/vits` model includes 109 different speakers.

**Using a specific speaker:**

```bash
python book-to-audiobook_coquitts.py input-text/mybook.txt -m tts_models/en/vctk/vits --speaker p225
```

**Available speakers for `tts_models/en/vctk/vits`:**

The VCTK model includes 109 speakers. The complete list:

```
ED
p225, p226, p227, p228, p229, p230, p231, p232, p233, p234
p236, p237, p238, p239, p240, p241, p243, p244, p245, p246
p247, p248, p249, p250, p251, p252, p253, p254, p255, p256
p257, p258, p259, p260, p261, p262, p263, p264, p265, p266
p267, p268, p269, p270, p271, p272, p273, p274, p275, p276
p277, p278, p279, p280, p281, p282, p283, p284, p285, p286
p287, p288, p292, p293, p294, p295, p297, p298, p299, p300
p301, p302, p303, p304, p305, p306, p307, p308, p310, p311
p312, p313, p314, p316, p317, p318, p323, p326, p329, p330
p333, p334, p335, p336, p339, p340, p341, p343, p345, p347
p351, p360, p361, p362, p363, p364, p374, p376
```

**Using reference audio (voice cloning):**

Instead of specifying a speaker name, you can use a reference audio file:

```bash
python book-to-audiobook_coquitts.py input-text/mybook.txt -m tts_models/en/vctk/vits --speaker-wav reference.wav
```

## Optional Dependencies

- **Server mode:** `pip install coqui-tts[server]`
- **Language-specific:** `pip install coqui-tts[ja]` (for Japanese, replace `ja` with other language codes as needed)

## Additional Requirements

### eSpeak-NG

Some models (particularly "capacitron" models) require **eSpeak-NG** to be installed. See `docs/ESPEAK_INSTALL.md` for installation instructions.

### PyTorch Compatibility

**Note:** PyTorch 2.6+ may have compatibility issues with some older CoquiTTS models (like `capacitron-t2-c150_v2`). 

- **Recommended models** work fine with PyTorch 2.6+:
  - `tts_models/en/ljspeech/tacotron2-DDC`
  - `tts_models/en/ljspeech/glow-tts`
  - `tts_models/en/vctk/vits`
  
- If you encounter `weights_only` errors, see `docs/PYTORCH_FIX.md` for solutions.

**Quick install on Windows:**
1. Download from: https://github.com/espeak-ng/espeak-ng/releases
2. Run the Windows installer (.msi or .exe)
3. Add to PATH: `C:\Program Files\eSpeak NG\`
4. Restart your terminal

After installation, restart your terminal and try again.

## Resources

- [CoquiTTS Documentation](https://coqui-tts.readthedocs.io/)
- [GitHub Repository](https://github.com/coqui-ai/TTS)
- [eSpeak-NG Installation Guide](docs/ESPEAK_INSTALL.md)

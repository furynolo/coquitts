# Installing eSpeak-NG for CoquiTTS

Some CoquiTTS models (particularly those with "capacitron" or phoneme-based models) require eSpeak-NG or eSpeak to be installed on your system.

## Windows Installation

### Manual Installation (Recommended)

1. **Download eSpeak-NG:**
   - Visit: https://github.com/espeak-ng/espeak-ng/releases
   - Download the latest Windows installer:
     - For 64-bit Windows: `espeak-ng-x64.msi` (or `espeak-ng-1.51.1-x64.exe` if available)
     - For 32-bit Windows: `espeak-ng-x86.msi`

2. **Install:**
   - Run the downloaded installer (`.msi` or `.exe`)
   - Follow the installation wizard
   - Note the installation directory (typically `C:\Program Files\eSpeak NG\` or `C:\Program Files (x86)\eSpeak NG\`)

3. **Add to PATH:**
   - Press `Win + R`, type `sysdm.cpl`, press Enter
   - Go to "Advanced" tab → "Environment Variables"
   - Under "System variables", find "Path" and click "Edit"
   - Click "New" and add the eSpeak NG directory:
     - `C:\Program Files\eSpeak NG\` (64-bit)
     - OR `C:\Program Files (x86)\eSpeak NG\` (32-bit)
   - Click OK on all dialogs

4. **Restart your terminal/IDE** for PATH changes to take effect

5. **Verify installation:**
   ```powershell
   espeak-ng --version
   ```
   
   If this doesn't work, try:
   ```powershell
   espeak --version
   ```

**Note:** eSpeak-NG is not available via `winget`. You must download and install it manually from GitHub.

## Linux Installation

### Debian/Ubuntu:
```bash
sudo apt-get update
sudo apt-get install espeak-ng
```

### Verify:
```bash
espeak-ng --version
```

## macOS Installation

### Using Homebrew:
```bash
brew install espeak-ng
```

### Verify:
```bash
espeak-ng --version
```

## Models That Require eSpeak

The following model types typically require eSpeak:
- Models with "capacitron" in the name (e.g., `tts_models/en/blizzard2013/capacitron-t2-c150_v2`)
- Models with "tacotron2-DDC_ph" (phoneme-based models)

## Troubleshooting

If you still get "No espeak backend found" errors after installation:

1. **Verify eSpeak is in PATH:**
   ```powershell
   # Windows PowerShell
   where.exe espeak-ng
   
   # Linux/Mac
   which espeak-ng
   ```

2. **Restart your terminal/IDE** - PATH changes require a restart

3. **Check Python can find it:**
   ```python
   import shutil
   print(shutil.which("espeak-ng"))
   ```

4. **Try using espeak instead of espeak-ng:**
   - Some systems may have `espeak` instead of `espeak-ng`
   - Both should work with CoquiTTS

## Alternative: Use Models That Don't Require eSpeak

If you prefer not to install eSpeak, you can use models that don't require it:
- `tts_models/en/ljspeech/tacotron2-DDC`
- `tts_models/en/ljspeech/glow-tts`
- `tts_models/en/vctk/vits`
- `tts_models/en/ljspeech/vits`

These models work without eSpeak and provide good quality output.

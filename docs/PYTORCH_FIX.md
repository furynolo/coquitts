# PyTorch 2.6+ Compatibility Fix

## Problem

PyTorch 2.6+ changed the default `weights_only=True` for security reasons. Some CoquiTTS models (particularly older ones like `capacitron-t2-c150_v2`) were saved with numpy objects that aren't allowed by default, causing this error:

```
WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray.scalar
```

## Solutions

### Solution 1: Downgrade PyTorch (Recommended for now)

Downgrade to PyTorch 2.5.x which doesn't have this restriction:

```powershell
pip install torch==2.5.1 torchaudio==2.5.1
```

**Note:** This is the simplest solution but means you'll be using an older PyTorch version.

### Solution 2: Use Models That Work with PyTorch 2.6+

The following models work fine with PyTorch 2.6+:
- `tts_models/en/ljspeech/tacotron2-DDC` ✓
- `tts_models/en/ljspeech/glow-tts` ✓
- `tts_models/en/vctk/vits` ✓
- `tts_models/en/ljspeech/vits` ✓
- `tts_models/en/ek1/tacotron2` ✓

Models that may have issues:
- `tts_models/en/blizzard2013/capacitron-t2-c150_v2` ✗
- `tts_models/en/blizzard2013/capacitron-t2-c50` ✗
- `tts_models/en/ljspeech/tacotron2-DDC_ph` ✗ (if it has the same issue)

### Solution 3: Wait for CoquiTTS Update

CoquiTTS may release an update that handles PyTorch 2.6+ compatibility. Check for updates:

```powershell
pip install --upgrade coqui-tts
```

## Current Status

- **Your PyTorch version:** 2.8.0+cpu
- **Your CoquiTTS version:** 0.27.2

## Recommendation

For now, either:
1. Use the recommended models listed above (they work fine)
2. Or downgrade PyTorch to 2.5.1 if you specifically need the capacitron models

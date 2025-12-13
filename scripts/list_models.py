"""
List available CoquiTTS models and test model compatibility.
"""

try:
    from TTS.api import TTS
except ImportError:
    print("Error: CoquiTTS is not installed. Please run: pip install -r requirements.txt")
    exit(1)

# Language code to name mapping
LANGUAGE_NAMES = {
    "en": "English",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "es": "Spanish",
    "et": "Estonian",
    "ewe": "Ewe",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "ga": "Irish",
    "hau": "Hausa",
    "hr": "Croatian",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "lin": "Lingala",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mt": "Maltese",
    "multilingual": "Multilingual",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sv": "Swedish",
    "tr": "Turkish",
    "tw_akuapem": "Twi (Akuapem)",
    "tw_asante": "Twi (Asante)",
    "uk": "Ukrainian",
    "yor": "Yoruba",
    "zh-CN": "Chinese (Simplified)",
}

def extract_language_code(model_name):
    """Extract language code from model name."""
    parts = model_name.split("/")
    if len(parts) >= 2 and parts[0] == "tts_models":
        lang_code = parts[1]
        return lang_code
    return None

def get_language_name(lang_code):
    """Get human-readable language name from code."""
    return LANGUAGE_NAMES.get(lang_code, lang_code)

def is_english_model(model_name):
    """Check if a model is English or multilingual (supports English)."""
    if not model_name.startswith("tts_models"):
        return True  # Include non-TTS models (vocoders, etc.)
    
    # Include English models
    if model_name.startswith("tts_models/en/"):
        return True
    
    # Include multilingual models (they support English)
    if "/multilingual/" in model_name:
        return True
    
    return False

def list_all_models(group_by_language=False, english_only=True):
    """List all available TTS models."""
    print("Fetching available models...")
    tts = TTS()
    all_models = tts.list_models()
    
    # Filter to English-only if requested
    if english_only:
        models = [m for m in all_models if is_english_model(m)]
        print(f"\nFound {len(models)} English/multilingual models (filtered from {len(all_models)} total):\n")
    else:
        models = all_models
        print(f"\nFound {len(models)} available models:\n")
    
    # Group models by type
    tts_models = [m for m in models if m.startswith("tts_models")]
    vocoder_models = [m for m in models if m.startswith("vocoder_models")]
    other_models = [m for m in models if m not in tts_models and m not in vocoder_models]
    
    if tts_models:
        if group_by_language:
            # Group TTS models by language
            models_by_lang = {}
            for model in tts_models:
                lang_code = extract_language_code(model)
                if lang_code:
                    lang_name = get_language_name(lang_code)
                    if lang_name not in models_by_lang:
                        models_by_lang[lang_name] = []
                    models_by_lang[lang_name].append(model)
                else:
                    if "Other" not in models_by_lang:
                        models_by_lang["Other"] = []
                    models_by_lang["Other"].append(model)
            
            print("TTS Models (grouped by language):")
            print("-" * 60)
            for lang_name in sorted(models_by_lang.keys()):
                print(f"\n  {lang_name}:")
                for model in sorted(models_by_lang[lang_name]):
                    requires_espeak = any(pattern in model for pattern in models_requiring_espeak())
                    espeak_note = " [requires espeak]" if requires_espeak else ""
                    print(f"    • {model}{espeak_note}")
            print()
        else:
            if english_only:
                print("TTS Models (English and Multilingual only):")
            else:
                print("TTS Models (for text-to-speech synthesis):")
            print("-" * 60)
            for model in sorted(tts_models):
                lang_code = extract_language_code(model)
                requires_espeak = any(pattern in model for pattern in models_requiring_espeak())
                
                lang_info = ""
                if lang_code:
                    lang_name = get_language_name(lang_code)
                    lang_info = f" ({lang_name})"
                
                espeak_note = " [requires espeak]" if requires_espeak else ""
                print(f"  • {model}{lang_info}{espeak_note}")
            print()
    
    if vocoder_models:
        print("Vocoder Models (for voice conversion):")
        print("-" * 60)
        for model in sorted(vocoder_models):
            print(f"  • {model}")
        print()
    
    if other_models:
        print("Other Models:")
        print("-" * 60)
        for model in sorted(other_models):
            print(f"  • {model}")
        print()
    
    return models

def detect_text_language(text):
    """Simple heuristic to detect if text is likely English."""
    # Very basic check - English text typically contains common English letters
    # and doesn't contain many non-ASCII characters
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text) if text else 0
    # Check for common non-English characters
    non_english_chars = any(ord(c) > 127 for c in text)
    
    if ascii_ratio > 0.9 and not non_english_chars:
        return "en"
    return "unknown"

def check_espeak_available():
    """Check if espeak or espeak-ng is available on the system."""
    import subprocess
    import shutil
    
    # Check if espeak or espeak-ng is in PATH
    espeak_path = shutil.which("espeak")
    espeak_ng_path = shutil.which("espeak-ng")
    
    if espeak_path or espeak_ng_path:
        return True, espeak_ng_path or espeak_path
    return False, None

def models_requiring_espeak():
    """Return list of model patterns that typically require espeak."""
    # Models with "capacitron" or "tacotron2-DDC_ph" typically require espeak
    return ["capacitron", "tacotron2-DDC_ph"]

def test_model(model_name, test_text="Thrall, son of Durotan, orc of Draenor but now denizen of Azeroth, stared in quiet contemplation at one of his guards, trying to make a decision. The guard attacked an unarmed civilian, though they did not kill them. Thrall can see how they may have truly perceived them to be a threat, the toy did look like a real weapon. Suddenly a memory if his past, one of or perhaps the earliest he can remember, played through is mind. He was just a babe, but there was loud comotion around him. Those he cared about were being attacked with weapons. Everyone was screaming, in terror, in pain, or simply a cry of battle. \"Come and face me, you cowards who hide in the shadows!\" a familiar low, tusked voice shouted. \"Please, just leave us be,\" another familiar but calm, matronly, and also tusked voice spoke. \"You cannot stop us all. You were foolish to be so far from your guards,\" the third, slightly higher male voice rasped before chuckling maniacly. Thrall made his decision."):
    """Test if a model can synthesize speech."""
    try:
        print(f"\nTesting model: {model_name}")
        
        # Check if model requires espeak
        requires_espeak = any(pattern in model_name for pattern in models_requiring_espeak())
        if requires_espeak:
            espeak_available, espeak_path = check_espeak_available()
            if not espeak_available:
                print("\n⚠ WARNING: This model requires espeak-ng or espeak to be installed.")
                print("  The model uses phoneme-based synthesis which requires espeak.")
                print("\n  To install espeak-ng on Windows:")
                print("    1. Download from: https://github.com/espeak-ng/espeak-ng/releases")
                print("    2. Download the Windows installer (.msi or .exe)")
                print("    3. Run the installer")
                print("    4. Add to PATH: C:\\Program Files\\eSpeak NG\\")
                print("    5. Restart your terminal")
                print("\n  See docs/ESPEAK_INSTALL.md for detailed instructions.")
                return False
        
        # Check language compatibility
        model_lang_code = extract_language_code(model_name)
        text_lang = detect_text_language(test_text)
        
        if model_lang_code and model_lang_code != "multilingual":
            model_lang_name = get_language_name(model_lang_code)
            if model_lang_code != text_lang and text_lang == "en":
                print(f"⚠ WARNING: This model is for {model_lang_name}, but you're testing with English text.")
                print(f"  The model may not support English characters, which could cause errors or poor output.")
                print(f"  Consider using text in {model_lang_name} for better results.")
                response = input("  Continue anyway? (y/n): ").strip().lower()
                if response != 'y':
                    print("Test cancelled.")
                    return False
        
        tts = TTS(model_name=model_name, progress_bar=True)
        output_path = f"test_{model_name.replace('/', '_').replace(':', '_')}.wav"
        tts.tts_to_file(text=test_text, file_path=output_path)
        print(f"✓ SUCCESS: Model works! Output saved to {output_path}")
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"✗ FAILED: {error_msg}")
        
        # Check for PyTorch 2.6+ compatibility issues
        if "weights_only" in error_msg or "WeightsUnpickler" in error_msg or "numpy.core.multiarray" in error_msg:
            print("\n⚠ PyTorch 2.6+ compatibility issue detected.")
            print("  This model may not be compatible with PyTorch 2.6+.")
            print("\n  Solutions:")
            print("    1. Use a different model (see recommended models)")
            print("    2. Downgrade PyTorch: pip install torch==2.5.1 torchaudio==2.5.1")
            print("    3. See docs/PYTORCH_FIX.md for more details")
        
        # Check for espeak-related errors
        if "espeak" in error_msg.lower() or "No espeak backend" in error_msg:
            print("\n⚠ This model requires espeak-ng or espeak to be installed.")
            print("\n  To install espeak-ng on Windows:")
            print("    1. Download from: https://github.com/espeak-ng/espeak-ng/releases")
            print("    2. Download the Windows installer (.msi or .exe)")
            print("    3. Run the installer")
            print("    4. Add to PATH: C:\\Program Files\\eSpeak NG\\")
            print("    5. Restart your terminal")
            print("\n  For Linux/Mac:")
            print("    sudo apt-get install espeak-ng  # Debian/Ubuntu")
            print("    brew install espeak-ng          # macOS")
            print("\n  See docs/ESPEAK_INSTALL.md for detailed instructions.")
        
        return False

def main():
    """Main function."""
    print("CoquiTTS Model Lister\n" + "="*60)
    
    # Automatically filter to English-only and group by language
    english_only = True
    group_by_lang = True
    
    models = list_all_models(group_by_language=group_by_lang, english_only=english_only)
    
    # Show recommended models
    print("\nRecommended Models for English TTS:")
    print("-" * 60)
    recommended = [
        "tts_models/en/ljspeech/tacotron2-DDC",
        "tts_models/en/ljspeech/glow-tts",
        "tts_models/en/vctk/vits",
        "tts_models/en/ek1/tacotron2",
    ]
    
    for model in recommended:
        if model in models:
            print(f"  ✓ {model}")
        else:
            print(f"  ✗ {model} (not available)")
    
    # Option to test a model
    print("\n" + "="*60)
    test = input("\nWould you like to test a model? (y/n): ").strip().lower()
    if test == 'y':
        model_name = input("Enter model name to test: ").strip()
        if model_name in models:
            # Allow custom test text
            custom_text = input("Enter test text (press Enter for default): ").strip()
            test_text = custom_text if custom_text else "Thrall, son of Durotan, orc of Draenor but now denizen of Azeroth, stared in quiet contemplation at one of his guards, trying to make a decision. The guard attacked an unarmed civilian, though they did not kill them. Thrall can see how they may have truly perceived them to be a threat, the toy did look like a real weapon. Suddenly a memory if his past, one of or perhaps the earliest he can remember, played through is mind. He was just a babe, but there was loud comotion around him. Those he cared about were being attacked with weapons. Everyone was screaming, in terror, in pain, or simply a cry of battle. \"Come and face me, you cowards who hide in the shadows!\" a familiar low, tusked voice shouted. \"Please, just leave us be,\" another familiar but calm, matronly, and also tusked voice spoke. \"You cannot stop us all. You were foolish to be so far from your guards,\" the third, slightly higher male voice rasped before chuckling maniacly. Thrall made his decision."
            test_model(model_name, test_text)
        else:
            print(f"Error: Model '{model_name}' not found in available models.")

if __name__ == "__main__":
    main()

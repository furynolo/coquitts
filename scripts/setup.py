"""
Setup script for CoquiTTS installation verification.
Run this after installing dependencies to verify everything works.
"""

import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7 or higher is required.")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_installation():
    """Check if CoquiTTS is installed."""
    try:
        import TTS
        print("✓ CoquiTTS is installed")
        return True
    except ImportError:
        print("❌ CoquiTTS is not installed. Run: pip install -r requirements.txt")
        return False

def check_cli():
    """Check if TTS CLI is available."""
    try:
        result = subprocess.run(
            ["tts", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✓ TTS CLI is available")
            return True
        else:
            print("⚠ TTS CLI returned an error")
            return False
    except FileNotFoundError:
        print("⚠ TTS CLI not found in PATH (this is optional)")
        return False
    except Exception as e:
        print(f"⚠ Error checking TTS CLI: {e}")
        return False

def main():
    """Run all checks."""
    print("CoquiTTS Setup Verification\n" + "="*50)
    
    all_checks = [
        check_python_version(),
        check_installation(),
        check_cli()
    ]
    
    print("\n" + "="*50)
    if all(all_checks[:2]):  # Python and TTS installation are required
        print("✓ Setup complete! You can now use CoquiTTS.")
        print("\nTry running: python example.py")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

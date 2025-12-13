# Windows Long Path Support Setup

## Problem
CoquiTTS installation fails on Windows due to path length limitations (260 character limit). The error message indicates that Windows Long Path support needs to be enabled.

## Solution 1: Enable Long Path Support via Registry (Recommended)

### Method A: Using Registry Editor (Manual)

1. **Open Registry Editor:**
   - Press `Win + R`
   - Type `regedit` and press Enter
   - Click "Yes" if prompted by UAC

2. **Navigate to:**
   ```
   HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
   ```

3. **Modify LongPathsEnabled:**
   - Find `LongPathsEnabled` in the right pane
   - Double-click it
   - Change value from `0` to `1`
   - Click OK

4. **Restart your computer** for changes to take effect

### Method B: Using PowerShell (Automated)

Run PowerShell as Administrator and execute:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Then restart your computer.

**Note:** You can use the `enable_long_paths.ps1` script in this directory (run as Administrator).

## Solution 2: Install in a Shorter Path

If you can't enable long paths, install in a directory with a shorter path:

1. Create a directory like `C:\tts` or `C:\coquitts`
2. Move your project there or create a new virtual environment there
3. Try installing again

## Solution 3: Use Alternative Installation Method

You can try installing with the `--no-cache-dir` flag:

```bash
pip install --no-cache-dir -r requirements.txt
```

Or install directly:

```bash
pip install --no-cache-dir coqui-tts
```

## After Enabling Long Paths

1. **Restart your computer**
2. **Update pip:**
   ```bash
   python -m pip install --upgrade pip
   ```
3. **Try installation again:**
   ```bash
   pip install -r requirements.txt
   ```

## Verify Long Path Support is Enabled

Run this in PowerShell to check:

```powershell
Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled"
```

If it returns `LongPathsEnabled : 1`, it's enabled. If it returns `0` or the property doesn't exist, it's not enabled.

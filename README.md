# BatchTranscribe

Whisper-powered batch audio/video transcription. Drag a folder of files in,
pick a model, hit Start. Outputs `.txt` and `.srt` next to your originals.

## Run from source

```bash
pip install -r requirements.txt
python main.py
```

## Speaker diarization

Enable **Speaker labels** to export anonymous labels such as `Speaker 1:` and `Speaker 2:` in TXT and SRT files. Labels reset for each recording. Attribution is based on overlapping speech timestamps and is approximate when people speak simultaneously.

From source, install the optional dependencies with `pip install -r requirements-speakers.txt`. Request access to the [Community-1 model](https://huggingface.co/pyannote/speaker-diarization-community-1), then follow its offline setup instructions to download the model into `~/.batchtranscribe/community-1` (or set `BATCHTRANSCRIBE_DIARIZATION_MODEL` to its local directory). Audio processing is local. The base executable does not bundle PyTorch or the gated model; use source mode or build with those optional dependencies installed and included in the package.

For faster processing, select **Fast decoding** (beam size 1 instead of 5) or use the `turbo` model. Fast decoding may reduce accuracy. Cancellation is checked between generated segments.

## Build a Windows .exe

```bat
build_windows.bat
```

Output: `build\windows\batchtranscribe.exe`. Keep the entire `build\windows` folder together: the executable requires the accompanying DLLs and `data` directory. The script verifies fresh output and opens the folder after a successful build.

The first build downloads a Flutter toolchain and the Flet runtime, so it
takes 5-10 minutes. Subsequent builds are faster.

### Windows build prerequisites (one-time setup)

The Flet/Flutter build toolchain needs two things that aren't on a default
Windows machine. Both are required.

1. **Enable Developer Mode** - required because Flutter's package format
   uses symlinks, and Windows blocks `os.symlink()` unless Developer Mode
   is on. Open the Settings app, search "Developer settings" (or run
   `ms-settings:developers`), and flip "Developer Mode" to On. No restart
   required.

2. **Complete Visual Studio 2022 install with the C++ desktop workload** -
   the Flutter Windows build needs MSVC + Windows 10/11 SDK to compile
   the C++ engine that hosts the Flet window. If you have VS 2022
   installed, open Visual Studio Installer  Modify  check "Desktop
   development with C++"  Modify. ~3 GB extra.

If `flutter doctor` (run from `C:\Users\<you>\flutter\3.41.7\bin\`) reports
the green check on "Visual Studio" and the symlink warning is gone, the
build is ready to run.

## Build a macOS .dmg

The build **must** run on a Mac - Windows can't cross-compile macOS apps.

```bash
chmod +x build_macos.sh
./build_macos.sh
```

Output:
- `build/macos/BatchTranscribe.app` - the application bundle
- `dist/BatchTranscribe-2.0.0.dmg` - drag-and-drop installer image

## Supported formats

MP3, MP4, WAV, FLAC, OGG, M4A, MOV, AVI, MKV, WEBM

## Settings

- **Model**: tiny (fastest)  large-v3 (most accurate). Default: medium.
- **Device**: CUDA if available, falls back to CPU.

### Windows GPU and publisher trust

CUDA requires a working CUDA 12 cuBLAS and cuDNN 9 runtime, not just an NVIDIA driver. The app checks the required DLLs at startup and retries a failed GPU inference on CPU. Install the matching runtime from NVIDIA, or set `BATCHTRANSCRIBE_CUDA_DIR` to a directory containing the matching NVIDIA runtime DLLs before running `build_windows.bat`. The build copies them into `build/windows/cuda/bin`. Keep the entire Windows output directory together.

Windows Smart App Control may reject unsigned bundled Python DLLs and extension modules. A production release should use a trusted code-signing certificate to sign the unsigned executable and native dependencies, or use a verified, vendor-signed runtime distribution. Do not disable Windows Security to run this app.
- **Language**: auto-detect, or pin to one of 20 supported languages.
- **Output**: tick `.txt` and/or `.srt`.
- **Output folder**: optional. If not selected, Start creates a `Transcripts` folder next to the first input file. Existing transcript files are preserved with numbered output filenames.

## Files

- `main.py` - Flet UI
- `transcriber.py` - faster-whisper backend with frozen-exe CUDA fix
- `build_icon.py` - generates `assets/app.ico` and `assets/app.iconset`
- `assets/` - bundled fonts (Inter) and icons
- `pyproject.toml` - Flet project metadata
- `build_windows.bat`, `build_macos.sh` - one-shot build scripts
- `plan.md` - V1 design notes (obsolete; see commit history)

## Configuration

Persisted in `~/.batchtranscribe.json`. Currently stores the theme preference.



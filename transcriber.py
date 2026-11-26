import os
import sys
import time
import threading
from queue import Queue

_DLL_DIRECTORY_HANDLES = []
_CUDA_LIBRARIES = []


def _candidate_nvidia_bases() -> list[str]:
    """Return every directory that *might* contain bundled nvidia/*/bin folders.

    Order matters - frozen-exe paths first because those are what Flet/PyInstaller
    actually use at runtime. Regular site-packages are still checked as a fallback.
    """
    bases: list[str] = [os.path.dirname(__file__), os.path.dirname(sys.executable), sys.prefix, *sys.path]
    meipass = getattr(sys, "_MEIPASS", None)  # set by PyInstaller
    if meipass:
        bases.append(meipass)
        bases.append(os.path.join(meipass, "_internal"))  # PyInstaller 6+
        # PyInstaller sometimes flattens into a "nvidia" subdir at the root
        bases.append(os.path.join(meipass, "nvidia"))
    if sys.platform == "win32":
        import site
        for sp in list(site.getsitepackages()):
            bases.append(sp)
            bases.append(os.path.join(sp, "nvidia"))
        user_site = site.getusersitepackages()
        if user_site:
            bases.append(user_site)
            bases.append(os.path.join(user_site, "nvidia"))
    return bases


def _add_nvidia_dll_dirs() -> int:
    """Walk candidate bases for nvidia/<lib>/bin and add them to the loader.

    Returns the number of directories successfully registered. On non-Windows
    this is a no-op (the loader doesn't use these).
    """
    if sys.platform != "win32":
        return 0
    nvidia_libs = ("cublas", "cudnn", "cufft", "curand", "cuda_runtime", "cuda_nvrtc", "nvjitlink")
    added = 0
    for base in _candidate_nvidia_bases():
        nvidia_root = os.path.join(base, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for lib in nvidia_libs:
            for d in (os.path.join(nvidia_root, lib, "bin"), os.path.join(nvidia_root, lib)):
                if not os.path.isdir(d):
                    continue
                try:
                    _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(d))
                    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                    added += 1
                except (OSError, AttributeError):
                    pass
    extra = [os.path.join(base, "cuda", "bin") for base in
             (os.path.dirname(sys.executable), os.path.dirname(__file__), sys.prefix)]
    for key, value in os.environ.items():
        if key == "CUDA_PATH" or key.startswith("CUDA_PATH_V") or key == "CUDNN_PATH":
            bin_dir = os.path.join(value, "bin")
            extra.append(bin_dir)
            if os.path.isdir(bin_dir):
                extra.extend(os.path.join(bin_dir, name) for name in os.listdir(bin_dir) if name.startswith("12."))
    for folder in dict.fromkeys(extra):
        if os.path.isdir(folder):
            try:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(folder))
                os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
                added += 1
            except OSError:
                pass
    return added


# Make sure the loader can find bundled CUDA deps *before* importing faster_whisper,
# which transitively loads cublas/cudnn via ctranslate2.
WhisperModel = None
_IMPORT_ERROR: Exception | None = None
_IMPORT_CHECKED = False
_IMPORT_LOCK = threading.Lock()


def get_import_error() -> Exception | None:
    """Load native dependencies lazily, after the UI has rendered."""
    global WhisperModel, _IMPORT_ERROR, _IMPORT_CHECKED
    with _IMPORT_LOCK:
        if not _IMPORT_CHECKED:
            try:
                _add_nvidia_dll_dirs()
                if sys.platform == "win32":
                    from windows_diarization import load_speaker_runtime, runtime_directory
                    runtime = runtime_directory()
                    if runtime is not None:
                        try:
                            load_speaker_runtime(runtime)
                        except OSError:
                            # Report an optional speaker-runtime failure when labels are requested.
                            pass
                from faster_whisper import WhisperModel as model_class
                WhisperModel = model_class
            except Exception as exc:
                _IMPORT_ERROR = exc
            _IMPORT_CHECKED = True
    return _IMPORT_ERROR


def check_cuda() -> tuple[bool, str]:
    """A detected GPU is not enough: verify the runtime libraries can load."""
    try:
        import ctypes
        error = get_import_error()
        if error is not None:
            return False, str(error)
        import ctranslate2
        if ctranslate2.get_cuda_device_count() < 1:
            return False, "No CUDA GPU detected."
        if sys.platform == "win32":
            for library in ("cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll"):
                _CUDA_LIBRARIES.append(ctypes.WinDLL(library))
        return True, "CUDA runtime detected."
    except Exception as exc:
        return False, f"CUDA unavailable: {exc}"


def format_srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_txt(filepath: str, segments: list) -> None:
    blocks = []
    previous = None
    for seg in segments:
        speaker = getattr(seg, "speaker", "")
        if speaker and speaker != previous:
            blocks.append(f"\n\n{speaker}: ")
        elif blocks:
            blocks.append(" ")
        blocks.append(seg.text.strip())
        previous = speaker
    text = "".join(blocks).strip()
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


def write_srt(filepath: str, segments: list) -> None:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_srt_timestamp(seg.start)
        end = format_srt_timestamp(seg.end)
        text = seg.text.strip()
        if getattr(seg, "speaker", ""):
            text = f"{seg.speaker}: {text}"
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


SUPPORTED_EXTS = {".mp3", ".mp4", ".wav", ".flac", ".ogg", ".m4a", ".mov", ".avi", ".mkv", ".webm"}
OUTPUT_FORMATS = {"txt": write_txt, "srt": write_srt}


class BatchTranscriber:
    def __init__(self, model_size: str = "medium", device: str = "cuda", compute_type: str = "float16",
                 language: str | None = None, formats: tuple[str, ...] = ("txt", "srt"),
                 queue: Queue | None = None, diarize: bool = False, fast: bool = False):
        self.model = None
        self.model_size = model_size
        self.device = device
        self.diarization_device = device if sys.platform == "win32" else "cpu"
        self.compute_type = compute_type
        self.language = language
        self.formats = formats
        self.queue = queue
        self.diarize = diarize
        self.fast = fast
        self.diarizer = None
        self._cancel = threading.Event()

    def _emit(self, kind: str, msg: str, file_index: int = -1, total_files: int = 0, pct: float = 0.0):
        if self.queue:
            self.queue.put({
                "kind": kind,
                "msg": msg,
                "file_index": file_index,
                "total_files": total_files,
                "pct": pct,
                "cancel": self._cancel.is_set(),
            })

    def cancel(self):
        self._cancel.set()

    def _load_model(self):
        get_import_error()
        if WhisperModel is None:
            err = _IMPORT_ERROR or RuntimeError("faster_whisper failed to import")
            self._emit("error", f"Cannot load model - faster_whisper not available: {err}")
            raise RuntimeError(str(err))
        self._emit("log", f"Loading model '{self.model_size}' on {self.device}...")
        try:
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            self._emit("log", "Model loaded.")
        except Exception as e:
            if self.device == "cuda":
                self._fallback_cpu(e)
            else:
                self._emit("error", f"Model load failed: {e}")
                raise

    def _fallback_cpu(self, error):
        self._emit("log", f"CUDA failed: {error}")
        self._emit("log", "Loading model on CPU (int8)...")
        self.model = None
        self.device = "cpu"
        self.compute_type = "int8"
        self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        self._emit("log", "Model loaded on CPU.")

    def _transcribe_file(self, audio):
        segments, _ = self.model.transcribe(audio, beam_size=1 if self.fast else 5, language=self.language, vad_filter=True)
        result = []
        for segment in segments:
            if self._cancel.is_set():
                break
            result.append(segment)
        return result

    def _collect_files(self, paths: list[str]) -> list[str]:
        files = []
        for p in paths:
            if os.path.isfile(p):
                if os.path.splitext(p)[1].lower() in SUPPORTED_EXTS:
                    files.append(p)
            elif os.path.isdir(p):
                for root, _, filenames in os.walk(p):
                    for fn in filenames:
                        fp = os.path.join(root, fn)
                        if os.path.splitext(fp)[1].lower() in SUPPORTED_EXTS:
                            files.append(fp)
        return files

    def transcribe_batch(self, paths: list[str], output_dir: str):
        if self.diarize and self.diarizer is None:
            from diarization import LocalDiarizer
            self._emit("log", f"Loading speaker diarization on {self.diarization_device.upper()}...")
            error = get_import_error()
            if error is not None:
                raise RuntimeError(f"Could not load transcription dependencies: {error}") from error
            self.diarizer = LocalDiarizer(device=self.diarization_device)
        if self.model is None:
            self._load_model()

        files = self._collect_files(paths)
        total = len(files)
        if total == 0:
            self._emit("error", "No supported audio/video files found.")
            return

        self._emit("log", f"Found {total} files to transcribe.")

        for idx, filepath in enumerate(files):
            if self._cancel.is_set():
                self._emit("log", "Transcription cancelled by user.")
                return

            basename = os.path.splitext(os.path.basename(filepath))[0]
            self._emit("progress", f"Transcribing {basename} ({idx+1}/{total})",
                       file_index=idx + 1, total_files=total, pct=0.0)

            try:
                start_time = time.time()
                from media_audio import decode_media
                audio = decode_media(filepath)
                if self._cancel.is_set():
                    return
                try:
                    segments = self._transcribe_file(audio)
                except Exception as exc:
                    if self.device != "cuda" or self._cancel.is_set():
                        raise
                    self._fallback_cpu(exc)
                    if self._cancel.is_set():
                        return
                    segments = self._transcribe_file(audio)
                if self._cancel.is_set():
                    return
                if self.diarizer is not None:
                    self._emit("log", f"Detecting speakers on {self.diarizer.device.upper()}: {basename}")
                    segments = self.diarizer(audio, segments)
                if self._cancel.is_set():
                    return
                elapsed = time.time() - start_time

                # Keep existing transcripts intact, including duplicate input basenames.
                stem = basename
                suffix = 2
                while any(os.path.exists(os.path.join(output_dir, f"{stem}.{fmt}")) for fmt in self.formats):
                    stem = f"{basename}_{suffix}"
                    suffix += 1
                for fmt in self.formats:
                    writer = OUTPUT_FORMATS.get(fmt)
                    if writer:
                        ext = fmt
                        out_path = os.path.join(output_dir, f"{stem}.{ext}")
                        writer(out_path, segments)

                self._emit("progress", f"Done: {basename} ({elapsed:.1f}s)",
                           file_index=idx + 1, total_files=total, pct=100.0)
            except Exception as e:
                self._emit("file_error", f"Failed: {filepath} - {e}", file_index=idx+1, total_files=total)

        self._emit("done", f"All {total} files processed.")


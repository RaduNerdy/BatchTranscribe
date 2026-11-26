"""Prepare/verify the one-click Windows distribution runtime."""
import sys
import filecmp
from pathlib import Path
import shutil

from diarization import prepare_models
from windows_diarization import RUNTIME_DLLS, runtime_directory

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".runtime-cache" / "speakers"
REQUIRED_CUDA = ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll",
                 "cudart64_12.dll", "cufft64_11.dll", "curand64_10.dll")


def prepare():
    from transcriber import get_import_error
    error = get_import_error()
    if error is not None:
        raise RuntimeError("Could not load the transcription backend") from error
    print("Downloading speaker models (cached for subsequent builds)...", flush=True)
    prepare_models(CACHE)
    from diarization import LocalDiarizer
    LocalDiarizer(CACHE)
    import unittest
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_media_audio.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("Audio decoder smoke test failed or was skipped.")


def verify(output):
    output = Path(output)
    packages = output / "site-packages"
    if not (packages / "flet-0.85.2.dist-info").is_dir() or any(p.name != "flet-0.85.2.dist-info" for p in packages.glob("flet-*.dist-info")):
        raise RuntimeError("Flet Python/native version mismatch: expected packaged Flet 0.85.2.")
    if not (packages / "sherpa_onnx").is_dir():
        raise RuntimeError("Flet did not package sherpa-onnx; check the build log.")
    cuda = packages / "nvidia"
    missing = [name for name in REQUIRED_CUDA if not any(cuda.rglob(name))]
    if missing:
        raise RuntimeError("Flet did not package CUDA runtime: " + ", ".join(missing))
    runtime = runtime_directory()
    if runtime is None:
        raise RuntimeError("The build Python is missing the Sherpa-ONNX CUDA runtime.")
    speaker_runtime = output / "speaker-runtime"
    speaker_runtime.mkdir(parents=True, exist_ok=True)
    for name in RUNTIME_DLLS:
        source = runtime / name
        target = speaker_runtime / name
        if not target.is_file() or not filecmp.cmp(source, target, shallow=False):
            shutil.copy2(source, target)
    models = output / "models" / "speakers"
    models.mkdir(parents=True, exist_ok=True)
    for source in prepare_models(CACHE):
        shutil.copy2(source, models / source.name)
    print("Verified bundled CUDA, speaker engine and model files.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        prepare()
    else:
        verify(sys.argv[1])


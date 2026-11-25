"""Prepare/verify the one-click Windows distribution runtime."""
import sys
from pathlib import Path
import shutil

from diarization import prepare_models

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".runtime-cache" / "speakers"
REQUIRED_CUDA = ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll")


def prepare():
    import sherpa_onnx
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


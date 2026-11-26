"""Bindings for Sherpa-ONNX 1.13.4's Windows CUDA diarization runtime."""
import ctypes as c
import os
import sys
from pathlib import Path


RUNTIME_DLLS = (
    "sherpa-onnx-c-api.dll",
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
    "onnxruntime_providers_cuda.dll",
)
_loaded_runtimes = {}


class PyannoteConfig(c.Structure):
    _fields_ = [("model", c.c_char_p)]


class SegmentationConfig(c.Structure):
    _fields_ = [("pyannote", PyannoteConfig), ("num_threads", c.c_int32),
                ("debug", c.c_int32), ("provider", c.c_char_p)]


class EmbeddingConfig(c.Structure):
    _fields_ = [("model", c.c_char_p), ("num_threads", c.c_int32),
                ("debug", c.c_int32), ("provider", c.c_char_p)]


class ClusteringConfig(c.Structure):
    _fields_ = [("num_clusters", c.c_int32), ("threshold", c.c_float)]


class Config(c.Structure):
    _fields_ = [("segmentation", SegmentationConfig), ("embedding", EmbeddingConfig),
                ("clustering", ClusteringConfig), ("min_duration_on", c.c_float),
                ("min_duration_off", c.c_float)]


class Segment(c.Structure):
    _fields_ = [("start", c.c_float), ("end", c.c_float), ("speaker", c.c_int32)]


def runtime_directory():
    for base in (Path(sys.executable).parent, Path(sys.prefix), Path(__file__).parent):
        directory = base / "speaker-runtime"
        if all((directory / name).is_file() for name in RUNTIME_DLLS):
            return directory
    from importlib.util import find_spec
    spec = find_spec("sherpa_onnx")
    if spec and spec.origin:
        directory = Path(spec.origin).parent / "lib"
        if all((directory / name).is_file() for name in RUNTIME_DLLS):
            return directory
    return None


def load_speaker_runtime(directory):
    # Keep these loaded across batches. ONNX Runtime's CPU and CUDA builds ship
    # different copies of the provider bridge under the same DLL filename.
    if directory not in _loaded_runtimes:
        dll_path = os.add_dll_directory(str(directory))
        core = c.WinDLL(str(directory / "onnxruntime.dll"))
        bridge = c.WinDLL(str(directory / "onnxruntime_providers_shared.dll"))
        _loaded_runtimes[directory] = (dll_path, core, bridge)


class WindowsDiarizer:
    def __init__(self, directory, segmentation, embedding, provider):
        self.handle = None
        load_speaker_runtime(directory)
        library = c.CDLL(str(directory / RUNTIME_DLLS[0]))

        def bind(name, result, *args):
            function = getattr(library, "SherpaOnnx" + name)
            function.restype = result
            function.argtypes = args
            return function

        version = bind("GetVersionStr", c.c_char_p)().decode()
        if version != "1.13.4":
            raise RuntimeError(f"Expected Sherpa-ONNX native 1.13.4, found {version}")
        pointer = c.c_void_p
        create = bind("CreateOfflineSpeakerDiarization", pointer, c.POINTER(Config))
        self.destroy = bind("DestroyOfflineSpeakerDiarization", None, pointer)
        self.process = bind("OfflineSpeakerDiarizationProcess", pointer, pointer, c.POINTER(c.c_float), c.c_int32)
        self.count = bind("OfflineSpeakerDiarizationResultGetNumSegments", c.c_int32, pointer)
        self.sort = bind("OfflineSpeakerDiarizationResultSortByStartTime", c.POINTER(Segment), pointer)
        self.free_segments = bind("OfflineSpeakerDiarizationDestroySegment", None, c.POINTER(Segment))
        self.free_result = bind("OfflineSpeakerDiarizationDestroyResult", None, pointer)
        device = provider.encode("ascii")
        config = Config(
            SegmentationConfig(PyannoteConfig(str(segmentation).encode("utf-8")), 1, 0, device),
            EmbeddingConfig(str(embedding).encode("utf-8"), 1, 0, device),
            ClusteringConfig(-1, 0.5), 0.3, 0.5,
        )
        self.handle = create(c.byref(config))
        if not self.handle:
            raise RuntimeError(f"Could not initialize speaker diarization on {provider.upper()}.")
        if provider == "cuda":
            get_module = c.windll.kernel32.GetModuleHandleW
            get_module.argtypes = [c.c_wchar_p]
            get_module.restype = c.c_void_p
            if not get_module(str(directory / "onnxruntime_providers_cuda.dll")):
                raise RuntimeError("The speaker engine could not load the ONNX Runtime CUDA provider.")
        self.sample_rate = bind("OfflineSpeakerDiarizationGetSampleRate", c.c_int32, pointer)(self.handle)

    def turns(self, audio):
        import numpy as np
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.size > 2**31 - 1:
            raise ValueError("Recording exceeds the speaker engine's sample limit.")
        result = self.process(self.handle, audio.ctypes.data_as(c.POINTER(c.c_float)), audio.size)
        if not result:
            raise RuntimeError("Speaker diarization failed.")
        try:
            count = self.count(result)
            if not count:
                return []
            segments = self.sort(result)
            if not segments:
                raise RuntimeError("The speaker engine returned no segment data.")
            try:
                return [(segments[i].start, segments[i].end, segments[i].speaker) for i in range(count)]
            finally:
                self.free_segments(segments)
        finally:
            self.free_result(result)

    def __del__(self):
        if self.handle:
            self.destroy(self.handle)


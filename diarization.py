"""Local, anonymous speaker diarization and timestamp alignment."""
from dataclasses import dataclass
from pathlib import Path
import os
import sys
import shutil
import tempfile
import urllib.request

MODEL_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
SEGMENTATION_ARCHIVE = "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMBEDDING_MODEL = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"


def prepare_models(directory):
    """Download public model weights once; never upload recordings."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    segmentation = directory / "segmentation.onnx"
    embedding = directory / "embedding.onnx"
    with tempfile.TemporaryDirectory(dir=directory) as temp:
        temporary = Path(temp)
        if not segmentation.is_file():
            import tarfile
            archive = temporary / SEGMENTATION_ARCHIVE
            with urllib.request.urlopen(MODEL_RELEASE + "speaker-segmentation-models/" + SEGMENTATION_ARCHIVE, timeout=60) as response, archive.open("wb") as out:
                shutil.copyfileobj(response, out)
            with tarfile.open(archive, "r:bz2") as tar:
                member = tar.getmember("sherpa-onnx-pyannote-segmentation-3-0/model.onnx")
                if not member.isfile():
                    raise RuntimeError("Invalid segmentation model archive")
                with tar.extractfile(member) as source, (temporary / "segmentation.onnx").open("wb") as out:
                    shutil.copyfileobj(source, out)
            (temporary / "segmentation.onnx").replace(segmentation)
        if not embedding.is_file():
            with urllib.request.urlopen(MODEL_RELEASE + "speaker-recongition-models/" + EMBEDDING_MODEL, timeout=60) as response, (temporary / "embedding.onnx").open("wb") as out:
                shutil.copyfileobj(response, out)
            (temporary / "embedding.onnx").replace(embedding)
    return segmentation, embedding


def model_directory():
    override = os.environ.get("BATCHTRANSCRIBE_DIARIZATION_MODEL")
    if override:
        return Path(override)
    for base in (Path(sys.executable).parent, Path(sys.prefix), Path(__file__).parent):
        candidate = base / "models" / "speakers"
        if (candidate / "segmentation.onnx").is_file() and (candidate / "embedding.onnx").is_file():
            return candidate
    return Path.home() / ".batchtranscribe" / "speakers"


@dataclass(frozen=True)
class SpeakerSegment:
    start: float
    end: float
    text: str
    speaker: str = ""


def align_speakers(segments, turns):
    """Assign the greatest overlapping speaker, with labels reset per file."""
    turns = sorted(turns, key=lambda t: t[0])
    labels = {}
    for _, _, label in turns:
        labels.setdefault(label, f"Speaker {len(labels) + 1}")
    result = []
    cursor = 0
    for seg in segments:
        while cursor < len(turns) and turns[cursor][1] <= seg.start:
            cursor += 1
        scores = {}
        for index in range(cursor, len(turns)):
            start, end, label = turns[index]
            if start >= seg.end:
                break
            overlap = max(0, min(end, seg.end) - max(start, seg.start))
            if overlap:
                scores[label] = scores.get(label, 0) + overlap
        label = labels[max(scores, key=scores.get)] if scores else "Unknown speaker"
        result.append(SpeakerSegment(seg.start, seg.end, seg.text, label))
    return result


class LocalDiarizer:
    def __init__(self, directory=None, device="cpu"):
        self.device = device
        segmentation, embedding = prepare_models(directory if directory is not None else model_directory())
        if sys.platform == "win32":
            from windows_diarization import WindowsDiarizer, runtime_directory
            runtime = runtime_directory()
            if runtime is not None:
                self.pipeline = WindowsDiarizer(runtime, segmentation, embedding, device)
                self.get_turns = self.pipeline.turns
                return
        if device == "cuda":
            raise RuntimeError("CUDA speaker diarization requires the Windows CUDA speaker runtime.")
        try:
            import sherpa_onnx as sherpa
        except ImportError as exc:
            raise RuntimeError("Speaker engine missing or blocked. Use the updated Windows build with the bundled speaker engine.") from exc
        config = sherpa.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(segmentation))),
            embedding=sherpa.SpeakerEmbeddingExtractorConfig(model=str(embedding)),
            clustering=sherpa.FastClusteringConfig(num_clusters=-1, threshold=0.5),
            min_duration_on=0.3, min_duration_off=0.5,
        )
        if not config.validate():
            raise RuntimeError("Invalid speaker model files; rebuild to restore them.")
        self.pipeline = sherpa.OfflineSpeakerDiarization(config)
        self.get_turns = self._cpu_turns

    def _cpu_turns(self, audio):
        output = self.pipeline.process(audio).sort_by_start_time()
        return [(turn.start, turn.end, turn.speaker) for turn in output]

    def __call__(self, audio, segments):
        from media_audio import SAMPLE_RATE
        if self.pipeline.sample_rate != SAMPLE_RATE:
            raise RuntimeError("Speaker model sample rate does not match transcription audio.")
        return align_speakers(segments, self.get_turns(audio))


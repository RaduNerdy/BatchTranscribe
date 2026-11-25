"""Local, anonymous speaker diarization and timestamp alignment."""
from dataclasses import dataclass
from pathlib import Path
import os


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
    def __init__(self):
        model = Path(os.environ.get("BATCHTRANSCRIBE_DIARIZATION_MODEL", str(Path.home() / ".batchtranscribe" / "community-1")))
        if not model.is_dir():
            raise RuntimeError("Speaker model not installed. See README: Speaker diarization setup.")
        from pyannote.audio import Pipeline
        self.pipeline = Pipeline.from_pretrained(str(model))

    def __call__(self, filepath, segments):
        import torch
        from faster_whisper.audio import decode_audio
        audio = decode_audio(filepath, sampling_rate=16000)
        output = self.pipeline({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000})
        turns = [(turn.start, turn.end, label) for turn, label in output.exclusive_speaker_diarization]
        return align_speakers(segments, turns)


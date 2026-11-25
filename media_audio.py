"""Decode media to the mono 16 kHz waveform shared by both inference engines."""
import io

SAMPLE_RATE = 16000


def decode_media(path):
    import av
    import numpy as np

    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    data = io.BytesIO()
    # PyAV 19 removed metadata_errors; the standard open signature works on
    # both older and newer releases without monkey-patching dependencies.
    with av.open(str(path), mode="r") as container:
        if not container.streams.audio:
            raise ValueError("The selected media file has no audio track.")
        frames = iter(container.decode(audio=0))
        while True:
            try:
                frame = next(frames)
            except StopIteration:
                break
            except av.error.InvalidDataError:
                continue
            frame.pts = None
            for converted in resampler.resample(frame):
                data.write(converted.to_ndarray().tobytes())
        for converted in resampler.resample(None):
            data.write(converted.to_ndarray().tobytes())
    if not data.tell():
        raise ValueError("No decodable audio found in the selected file.")
    return np.frombuffer(data.getbuffer(), dtype=np.int16).astype(np.float32) / 32768.0


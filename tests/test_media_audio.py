import importlib.util
import tempfile
import unittest
import wave
from pathlib import Path

from media_audio import decode_media, SAMPLE_RATE


@unittest.skipUnless(importlib.util.find_spec("av") and importlib.util.find_spec("numpy"), "audio dependencies not installed")
class MediaAudioTests(unittest.TestCase):
    def test_stereo_resamples_to_mono_float32(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
                wav.writeframes(b"\x00\x10\x00\x10" * 48000)
            audio = decode_media(path)
            self.assertEqual(audio.ndim, 1)
            self.assertEqual(audio.dtype.name, "float32")
            self.assertEqual(len(audio), SAMPLE_RATE)
            self.assertGreater(float(audio.max()), 0)


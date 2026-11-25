import subprocess
import sys
import unittest


class StartupTests(unittest.TestCase):
    def test_backend_import_is_lazy(self):
        subprocess.run([sys.executable, "-c", "import sys, transcriber; assert 'faster_whisper' not in sys.modules; assert 'ctranslate2' not in sys.modules"], check=True)


if __name__ == "__main__":
    unittest.main()


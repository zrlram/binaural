import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from add_binaural import generate_binaural_wave
from detect_beat import BinauralBeatDetector


SAMPLE_RATE = 44100


def centered_tone(frequency, duration, volume=0.5):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    tone = np.sin(2 * np.pi * frequency * t) * volume
    return np.stack((tone, tone), axis=-1)


class BinauralBeatDetectorTests(unittest.TestCase):
    def detector(self, min_duration=20.0):
        return BinauralBeatDetector(
            min_duration=min_duration,
            min_beat_freq=0.5,
            max_beat_freq=40,
            max_carrier_freq=600,
            window_size=4.0,
            hop_size=1.0,
            relative_floor_db=70,
            max_peaks=32,
        )

    def detect_array(self, audio, min_duration=20.0):
        detector = self.detector(min_duration=min_duration)
        return detector.detect_binaural_beats(audio[:, 0], audio[:, 1], SAMPLE_RATE)

    def test_detects_multiple_generated_sections_and_labels_bands(self):
        sections = [
            (25, 200, 4, "theta"),
            (25, 250, 10, "alpha"),
            (25, 300, 20, "beta"),
        ]
        audio = np.concatenate(
            [
                generate_binaural_wave(base, beat, duration, SAMPLE_RATE) * 0.5
                for duration, base, beat, _band in sections
            ]
        )

        beats = self.detect_array(audio)

        self.assertEqual(len(beats), 3)
        for detected, expected in zip(beats, sections):
            duration, base, beat, band = expected
            self.assertAlmostEqual(detected.duration, duration, delta=1.5)
            self.assertAlmostEqual(detected.base_freq, base, delta=0.5)
            self.assertAlmostEqual(detected.beat_freq, beat, delta=0.35)
            self.assertEqual(detected.band, band)

    def test_rejects_identical_stereo_tone_as_zero_hz_beat(self):
        audio = centered_tone(440, 25)

        beats = self.detect_array(audio)

        self.assertEqual(beats, [])

    def test_finds_quiet_binaural_beat_under_louder_centered_tone(self):
        duration = 25
        binaural = generate_binaural_wave(200, 4, duration, SAMPLE_RATE) * 0.12
        masker = centered_tone(440, duration, volume=0.5)

        beats = self.detect_array(binaural + masker)

        self.assertEqual(len(beats), 1)
        self.assertAlmostEqual(beats[0].base_freq, 200, delta=0.5)
        self.assertAlmostEqual(beats[0].beat_freq, 4, delta=0.35)
        self.assertEqual(beats[0].band, "theta")

    def test_ignores_sections_below_duration_threshold(self):
        audio = generate_binaural_wave(200, 4, 10, SAMPLE_RATE) * 0.5

        beats = self.detect_array(audio, min_duration=20)

        self.assertEqual(beats, [])

    def test_reports_original_timeline_when_start_offset_is_used(self):
        audio = generate_binaural_wave(200, 4, 35, SAMPLE_RATE) * 0.5

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "offset.wav"
            sf.write(path, audio, SAMPLE_RATE)
            beats = self.detector().find_binaural_beats(str(path), start_time=5, end_time=30)

        self.assertEqual(len(beats), 1)
        self.assertAlmostEqual(beats[0].start_time, 5, delta=0.25)
        self.assertAlmostEqual(beats[0].duration, 25, delta=1.0)

    def test_estimates_non_integer_gateway_style_beat(self):
        audio = generate_binaural_wave(100, 3.78, 25, SAMPLE_RATE) * 0.5

        beats = self.detect_array(audio)

        self.assertEqual(len(beats), 1)
        self.assertAlmostEqual(beats[0].beat_freq, 3.78, delta=0.25)
        self.assertEqual(beats[0].band, "delta")


if __name__ == "__main__":
    unittest.main()

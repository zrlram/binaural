#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.io import wavfile


@dataclass
class BinauralBeat:
    """Detected sustained stereo frequency offset."""

    start_time: float
    duration: float
    left_freq: float
    right_freq: float
    base_freq: float
    beat_freq: float
    amplitude: float
    prominence_db: float
    intensity_score: float
    intensity: str
    band: str
    confidence: str


@dataclass
class Peak:
    freq: float
    amp: float
    prominence: float


def load_pyplot():
    """Import Matplotlib only when image output is requested."""
    config_dir = os.path.join(tempfile.gettempdir(), "binaural-matplotlib")
    os.makedirs(config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", config_dir)
    import matplotlib.pyplot as plt

    return plt


class BinauralBeatDetector:
    def __init__(
        self,
        min_duration: float = 20.0,
        min_beat_freq: float = 0.5,
        max_beat_freq: float = 100.0,
        min_carrier_freq: float = 40.0,
        max_carrier_freq: float = 1000.0,
        freq_tolerance: float = 1.0,
        beat_tolerance: float = 0.75,
        amp_tolerance_db: float = 18.0,
        relative_floor_db: float = 60.0,
        max_peaks: int = 24,
        max_pairs_per_frame: int = 80,
        window_size: float = 4.0,
        hop_size: float = 1.0,
        analysis_sample_rate: int = 4096,
        debug: bool = False,
    ):
        self.min_duration = min_duration
        self.min_beat_freq = min_beat_freq
        self.max_beat_freq = max_beat_freq
        self.min_carrier_freq = min_carrier_freq
        self.max_carrier_freq = max_carrier_freq
        self.freq_tolerance = freq_tolerance
        self.beat_tolerance = beat_tolerance
        self.amp_tolerance_db = amp_tolerance_db
        self.relative_floor_db = relative_floor_db
        self.max_peaks = max_peaks
        self.max_pairs_per_frame = max_pairs_per_frame
        self.window_size = window_size
        self.hop_size = hop_size
        self.analysis_sample_rate = analysis_sample_rate
        self.debug = debug
        if self.max_carrier_freq >= self.analysis_sample_rate / 2:
            raise ValueError("analysis_sample_rate must be greater than twice max_carrier_freq")

    def load_audio(
        self,
        file_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Decode an audio file to stereo float channels."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name

        cmd = ["ffmpeg", "-v", "error"]
        if start_time is not None:
            cmd.extend(["-ss", str(start_time)])
        cmd.extend(["-i", file_path])
        if end_time is not None:
            duration = end_time - (start_time or 0)
            if duration <= 0:
                raise ValueError("end_time must be greater than start_time")
            cmd.extend(["-t", str(duration)])
        cmd.extend([
            "-ac", "2",
            "-ar", str(self.analysis_sample_rate),
            "-acodec", "pcm_s16le",
            "-y", temp_wav_path,
        ])

        try:
            subprocess.run(cmd, check=True)
            sample_rate, data = wavfile.read(temp_wav_path)
        finally:
            os.unlink(temp_wav_path)

        if len(data.shape) == 1 or data.shape[1] < 2:
            raise ValueError("Input audio must be stereo")

        data = data[:, :2].astype(np.float32) / np.iinfo(data.dtype).max
        return data[:, 0], data[:, 1], sample_rate

    def get_spectrogram(
        self,
        channel: np.ndarray,
        sample_rate: float,
        window_size: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute a dB spectrogram for one channel."""
        window_size = window_size or self.window_size
        nperseg = int(window_size * sample_rate)
        hop_samples = max(1, int(self.hop_size * sample_rate))
        noverlap = max(0, nperseg - hop_samples)

        frequencies, times, spectrum = signal.spectrogram(
            channel,
            sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hann",
            scaling="spectrum",
        )
        spectrum_db = 10 * np.log10(spectrum + np.finfo(float).eps)
        return frequencies, times, spectrum_db

    def frame_peak_candidates(
        self,
        channel: np.ndarray,
        sample_rate: float,
    ) -> Tuple[np.ndarray, List[List[Peak]], np.ndarray, np.ndarray]:
        """Find multiple possible carrier tones per analysis frame."""
        frequencies, times, spectrum_db = self.get_spectrogram(channel, sample_rate)
        freq_mask = (
            (frequencies >= self.min_carrier_freq)
            & (frequencies <= self.max_carrier_freq)
        )
        freq_indices = np.flatnonzero(freq_mask)
        peaks_by_frame: List[List[Peak]] = []

        for frame in range(spectrum_db.shape[1]):
            frame_db = spectrum_db[:, frame]
            search_db = frame_db[freq_mask]
            if search_db.size == 0:
                peaks_by_frame.append([])
                continue

            floor = float(np.max(search_db) - self.relative_floor_db)
            local_peaks, _ = signal.find_peaks(search_db, height=floor)
            if local_peaks.size == 0:
                local_peaks = np.array([int(np.argmax(search_db))])
                prominences = np.array([float(np.max(search_db) - np.median(search_db))])
            else:
                prominences = signal.peak_prominences(search_db, local_peaks)[0]

            absolute_indices = freq_indices[local_peaks]
            candidates = [
                self.interpolate_peak(frequencies, frame_db, int(index), float(prominence))
                for index, prominence in zip(absolute_indices, prominences)
            ]
            candidates.sort(key=lambda peak: peak.amp, reverse=True)
            peaks_by_frame.append(candidates[: self.max_peaks])

            if self.debug:
                preview = ", ".join(
                    f"{peak.freq:.2f}Hz/{peak.amp:.1f}dB"
                    for peak in peaks_by_frame[-1][:6]
                )
                print(f"Frame {frame}: {preview}")

        return times, peaks_by_frame, frequencies, spectrum_db

    @staticmethod
    def interpolate_peak(
        frequencies: np.ndarray,
        frame_db: np.ndarray,
        peak_index: int,
        prominence: float,
    ) -> Peak:
        """Refine a spectral peak using quadratic interpolation."""
        freq = float(frequencies[peak_index])
        amp = float(frame_db[peak_index])
        if 0 < peak_index < len(frame_db) - 1:
            left = frame_db[peak_index - 1]
            center = frame_db[peak_index]
            right = frame_db[peak_index + 1]
            denom = left - 2 * center + right
            if denom != 0:
                offset = 0.5 * (left - right) / denom
                if abs(offset) <= 1:
                    freq += float(offset * (frequencies[1] - frequencies[0]))
                    amp = float(center - 0.25 * (left - right) * offset)
        return Peak(freq=freq, amp=amp, prominence=prominence)

    def frame_pair_candidates(
        self,
        left_peaks: List[Peak],
        right_peaks: List[Peak],
    ) -> List[Dict[str, float]]:
        """Build candidate binaural pairs from one frame of left/right peaks."""
        candidates: List[Dict[str, float]] = []
        right_peaks = sorted(right_peaks, key=lambda peak: peak.freq)
        for left in left_peaks:
            for right in right_peaks:
                beat_freq = abs(right.freq - left.freq)
                if right.freq > left.freq + self.max_beat_freq:
                    break
                if right.freq < left.freq - self.max_beat_freq:
                    continue
                if beat_freq < self.min_beat_freq or beat_freq > self.max_beat_freq:
                    continue
                if abs(left.amp - right.amp) > self.amp_tolerance_db:
                    continue
                candidates.append(
                    {
                        "left_freq": left.freq,
                        "right_freq": right.freq,
                        "base_freq": min(left.freq, right.freq),
                        "beat_freq": beat_freq,
                        "amplitude": (left.amp + right.amp) / 2,
                        "prominence_db": (left.prominence + right.prominence) / 2,
                    }
                )
        candidates.sort(key=lambda item: (-item["amplitude"], item["beat_freq"]))
        return candidates[: self.max_pairs_per_frame]

    def detect_binaural_beats(
        self,
        left: np.ndarray,
        right: np.ndarray,
        sample_rate: float,
    ) -> List[BinauralBeat]:
        """Detect prolonged, stable left/right carrier offsets."""
        times, left_peaks, _, _ = self.frame_peak_candidates(left, sample_rate)
        right_times, right_peaks, _, _ = self.frame_peak_candidates(right, sample_rate)
        if not np.allclose(times, right_times):
            raise ValueError("Left and right channel analysis windows do not align")

        active_tracks: List[Dict[str, object]] = []
        finished_tracks: List[Dict[str, object]] = []

        for frame, (left_frame, right_frame) in enumerate(zip(left_peaks, right_peaks)):
            pairs = self.frame_pair_candidates(left_frame, right_frame)
            assigned_tracks = set()
            assigned_pairs = set()

            for pair_index, pair in enumerate(pairs):
                best_track_index = None
                best_score = None
                for track_index, track in enumerate(active_tracks):
                    if track_index in assigned_tracks:
                        continue
                    if int(track["last_frame"]) != frame - 1:
                        continue

                    left_delta = abs(
                        pair["left_freq"] - float(np.mean(track["left_freqs"]))
                    )
                    right_delta = abs(
                        pair["right_freq"] - float(np.mean(track["right_freqs"]))
                    )
                    beat_delta = abs(
                        pair["beat_freq"] - float(np.mean(track["beat_freqs"]))
                    )
                    if (
                        left_delta <= self.freq_tolerance
                        and right_delta <= self.freq_tolerance
                        and beat_delta <= self.beat_tolerance
                    ):
                        score = left_delta + right_delta + beat_delta
                        if best_score is None or score < best_score:
                            best_score = score
                            best_track_index = track_index

                if best_track_index is None:
                    continue

                track = active_tracks[best_track_index]
                self.add_pair_to_track(track, frame, pair)
                assigned_tracks.add(best_track_index)
                assigned_pairs.add(pair_index)

            still_active: List[Dict[str, object]] = []
            for track_index, track in enumerate(active_tracks):
                if int(track["last_frame"]) == frame:
                    still_active.append(track)
                else:
                    finished_tracks.append(track)
            active_tracks = still_active

            for pair_index, pair in enumerate(pairs):
                if pair_index in assigned_pairs:
                    continue
                active_tracks.append(self.new_track(frame, pair))

        finished_tracks.extend(active_tracks)
        audio_duration = len(left) / sample_rate
        beats = [
            beat
            for track in finished_tracks
            if (beat := self.finalize_track(track, times, audio_duration)) is not None
        ]
        beats.sort(key=lambda beat: (beat.start_time, beat.beat_freq, beat.base_freq))
        return self.resolve_overlaps(self.dedupe_beats(beats))

    @staticmethod
    def new_track(frame: int, pair: Dict[str, float]) -> Dict[str, object]:
        return {
            "start_frame": frame,
            "last_frame": frame,
            "left_freqs": [pair["left_freq"]],
            "right_freqs": [pair["right_freq"]],
            "beat_freqs": [pair["beat_freq"]],
            "amplitudes": [pair["amplitude"]],
            "prominences": [pair["prominence_db"]],
        }

    @staticmethod
    def add_pair_to_track(
        track: Dict[str, object],
        frame: int,
        pair: Dict[str, float],
    ) -> None:
        track["last_frame"] = frame
        track["left_freqs"].append(pair["left_freq"])
        track["right_freqs"].append(pair["right_freq"])
        track["beat_freqs"].append(pair["beat_freq"])
        track["amplitudes"].append(pair["amplitude"])
        track["prominences"].append(pair["prominence_db"])

    def finalize_track(
        self,
        track: Dict[str, object],
        times: np.ndarray,
        audio_duration: float,
    ) -> Optional[BinauralBeat]:
        start_frame = int(track["start_frame"])
        last_frame = int(track["last_frame"])
        start_time = max(0.0, float(times[start_frame] - self.window_size / 2))
        end_time = min(audio_duration, float(times[last_frame] + self.window_size / 2))
        duration = end_time - start_time
        if duration < self.min_duration:
            return None

        left_freq = float(np.mean(track["left_freqs"]))
        right_freq = float(np.mean(track["right_freqs"]))
        beat_freq = abs(right_freq - left_freq)
        if beat_freq < self.min_beat_freq or beat_freq > self.max_beat_freq:
            return None

        beat_std = float(np.std(track["beat_freqs"]))
        frame_count = len(track["beat_freqs"])
        expected_frames = max(1, int(round(duration / self.hop_size)))
        coverage = min(1.0, frame_count / expected_frames)
        if coverage >= 0.8 and beat_std <= self.beat_tolerance:
            confidence = "high"
        elif coverage >= 0.6 and beat_std <= self.beat_tolerance * 2:
            confidence = "medium"
        else:
            confidence = "low"

        prominence_db = float(np.mean(track["prominences"]))
        intensity_score = self.intensity_score(prominence_db)
        return BinauralBeat(
            start_time=start_time,
            duration=duration,
            left_freq=left_freq,
            right_freq=right_freq,
            base_freq=min(left_freq, right_freq),
            beat_freq=beat_freq,
            amplitude=float(np.mean(track["amplitudes"])),
            prominence_db=prominence_db,
            intensity_score=intensity_score,
            intensity=self.classify_intensity(intensity_score),
            band=self.classify_band(beat_freq),
            confidence=confidence,
        )

    def dedupe_beats(self, beats: List[BinauralBeat]) -> List[BinauralBeat]:
        deduped: List[BinauralBeat] = []
        for beat in beats:
            duplicate_index = None
            for index, existing in enumerate(deduped):
                if (
                    abs(existing.start_time - beat.start_time) <= self.hop_size
                    and abs(existing.duration - beat.duration) <= self.hop_size * 2
                    and abs(existing.left_freq - beat.left_freq) <= self.freq_tolerance
                    and abs(existing.right_freq - beat.right_freq) <= self.freq_tolerance
                ):
                    duplicate_index = index
                    break
            if duplicate_index is None:
                deduped.append(beat)
            elif self.confidence_rank(beat.confidence) > self.confidence_rank(
                deduped[duplicate_index].confidence
            ):
                deduped[duplicate_index] = beat

        deduped.sort(key=lambda beat: (beat.start_time, beat.beat_freq, beat.base_freq))
        return deduped

    def resolve_overlaps(self, beats: List[BinauralBeat]) -> List[BinauralBeat]:
        """Split adjacent smeared detections caused by long STFT windows."""
        if len(beats) < 2:
            return beats

        beats.sort(key=lambda beat: (beat.start_time, beat.beat_freq, beat.base_freq))
        for current, following in zip(beats, beats[1:]):
            current_end = current.start_time + current.duration
            following_end = following.start_time + following.duration
            if following.start_time >= current_end:
                continue
            if following.start_time <= current.start_time:
                continue

            transition = (current_end + following.start_time) / 2
            current.duration = max(0.0, transition - current.start_time)
            following.start_time = transition
            following.duration = max(0.0, following_end - transition)

        return [beat for beat in beats if beat.duration >= self.min_duration]

    @staticmethod
    def confidence_rank(confidence: str) -> int:
        return {"low": 0, "medium": 1, "high": 2}.get(confidence, 0)

    @staticmethod
    def classify_band(beat_freq: float) -> str:
        # Keep exact boundary targets like 4.0 Hz stable despite FFT interpolation noise.
        if 0.5 <= beat_freq < 3.95:
            return "delta"
        if 3.95 <= beat_freq < 8.0:
            return "theta"
        if 8.0 <= beat_freq < 14.0:
            return "alpha"
        if 14.0 <= beat_freq < 30.0:
            return "beta"
        if beat_freq >= 30.0:
            return "gamma"
        return "sub-delta"

    @staticmethod
    def intensity_score(prominence_db: float) -> float:
        """Map spectral prominence to a 0-100 signal-intensity score."""
        return float(np.clip((prominence_db / 40.0) * 100.0, 0.0, 100.0))

    @staticmethod
    def classify_intensity(intensity_score: float) -> str:
        if intensity_score >= 75.0:
            return "high"
        if intensity_score >= 40.0:
            return "medium"
        return "low"

    def visualize_analysis(
        self,
        left: np.ndarray,
        right: np.ndarray,
        sample_rate: float,
        beats: List[BinauralBeat],
        save_path: Optional[str] = None,
    ) -> None:
        plt = load_pyplot()
        f_left, t_left, s_left = self.get_spectrogram(left, sample_rate)
        f_right, t_right, s_right = self.get_spectrogram(right, sample_rate)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        max_freq = self.max_carrier_freq
        freq_mask = f_left <= max_freq

        im1 = ax1.pcolormesh(t_left, f_left[freq_mask], s_left[freq_mask], shading="gouraud")
        im2 = ax2.pcolormesh(t_right, f_right[freq_mask], s_right[freq_mask], shading="gouraud")
        plt.colorbar(im1, ax=ax1, label="Power (dB)")
        plt.colorbar(im2, ax=ax2, label="Power (dB)")

        for beat in beats:
            rect1 = plt.Rectangle(
                (beat.start_time, beat.left_freq - 5),
                beat.duration,
                10,
                facecolor="none",
                edgecolor="red",
                linewidth=2,
            )
            rect2 = plt.Rectangle(
                (beat.start_time, beat.right_freq - 5),
                beat.duration,
                10,
                facecolor="none",
                edgecolor="red",
                linewidth=2,
            )
            ax1.add_patch(rect1)
            ax2.add_patch(rect2)

        ax1.set_ylabel("Frequency (Hz)")
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_xlabel("Time (s)")
        ax1.set_title("Left Channel Spectrogram")
        ax2.set_title("Right Channel Spectrogram")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close(fig)

    def visualize_summary(
        self,
        beats: List[BinauralBeat],
        save_path: str,
        title: str = "Detected Binaural Beats",
    ) -> None:
        """Create a compact timeline summary of detected beats."""
        plt = load_pyplot()
        band_colors = {
            "delta": "#4c78a8",
            "theta": "#72b7b2",
            "alpha": "#54a24b",
            "beta": "#f58518",
            "gamma": "#b279a2",
            "sub-delta": "#9d755d",
        }
        fig, (timeline_ax, beat_ax, carrier_ax) = plt.subplots(
            3,
            1,
            figsize=(14, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [1.3, 1.0, 1.0]},
        )

        if not beats:
            timeline_ax.text(
                0.5,
                0.5,
                "No sustained binaural beats detected",
                ha="center",
                va="center",
                transform=timeline_ax.transAxes,
                fontsize=14,
            )
            timeline_ax.set_axis_off()
            beat_ax.set_axis_off()
            carrier_ax.set_axis_off()
            fig.suptitle(title)
            plt.tight_layout()
            plt.savefig(save_path, dpi=160)
            plt.close(fig)
            return

        for index, beat in enumerate(beats):
            color = band_colors.get(beat.band, "#bab0ac")
            timeline_ax.barh(
                y=index,
                width=beat.duration,
                left=beat.start_time,
                height=0.72,
                color=color,
                edgecolor="black",
                linewidth=0.6,
            )
            timeline_ax.text(
                beat.start_time + beat.duration / 2,
                index,
                f"{beat.band} {beat.beat_freq:.2f} Hz",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )

        y_labels = [
            f"{beat.start_time:.0f}-{beat.start_time + beat.duration:.0f}s"
            for beat in beats
        ]
        timeline_ax.set_yticks(range(len(beats)))
        timeline_ax.set_yticklabels(y_labels)
        timeline_ax.invert_yaxis()
        timeline_ax.set_ylabel("Sections")
        timeline_ax.grid(axis="x", alpha=0.25)

        starts = [beat.start_time for beat in beats]
        mids = [beat.start_time + beat.duration / 2 for beat in beats]
        beat_freqs = [beat.beat_freq for beat in beats]
        base_freqs = [beat.base_freq for beat in beats]
        shifted_freqs = [max(beat.left_freq, beat.right_freq) for beat in beats]
        colors = [band_colors.get(beat.band, "#bab0ac") for beat in beats]

        beat_ax.scatter(mids, beat_freqs, c=colors, s=60, edgecolors="black", linewidths=0.5)
        beat_ax.plot(mids, beat_freqs, color="#555555", alpha=0.5, linewidth=1)
        beat_ax.set_ylabel("Beat Hz")
        beat_ax.grid(alpha=0.25)

        carrier_ax.scatter(starts, base_freqs, label="Base", color="#4c78a8", s=45)
        carrier_ax.scatter(starts, shifted_freqs, label="Shifted", color="#f58518", s=45)
        for beat in beats:
            carrier_ax.plot(
                [beat.start_time, beat.start_time],
                [beat.base_freq, max(beat.left_freq, beat.right_freq)],
                color="#777777",
                alpha=0.45,
                linewidth=1,
            )
        carrier_ax.set_ylabel("Carrier Hz")
        carrier_ax.set_xlabel("Time (s)")
        carrier_ax.grid(alpha=0.25)
        carrier_ax.legend(loc="best")

        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        plt.close(fig)

    def find_binaural_beats(
        self,
        file_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        visualization_path: Optional[str] = None,
        summary_path: Optional[str] = None,
    ) -> List[BinauralBeat]:
        time_offset = start_time or 0.0
        left, right, sample_rate = self.load_audio(file_path, start_time, end_time)
        beats = self.detect_binaural_beats(left, right, sample_rate)

        if visualization_path:
            self.visualize_analysis(left, right, sample_rate, beats, visualization_path)

        for beat in beats:
            beat.start_time += time_offset
        if summary_path:
            self.visualize_summary(beats, summary_path)
        return beats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect binaural beats in stereo audio files")
    parser.add_argument("file_path", help="Path to the stereo audio file")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--visualize", help="Save visualization to specified path")
    parser.add_argument("--summary-image", help="Save compact timeline summary to specified path")
    parser.add_argument("--json", action="store_true", help="Output detected beats as JSON")
    parser.add_argument("--verbose", action="store_true", help="Print detailed multi-line output")
    parser.add_argument("--start", type=float, help="Start time in seconds")
    parser.add_argument("--end", type=float, help="End time in seconds")
    parser.add_argument("--min-duration", type=float, default=20.0, help="Minimum duration for binaural beats")
    parser.add_argument("--min-beat-freq", type=float, default=0.5, help="Minimum allowed beat frequency")
    parser.add_argument("--max-beat-freq", type=float, default=100.0, help="Maximum allowed beat frequency")
    parser.add_argument("--min-carrier-freq", type=float, default=40.0, help="Minimum carrier frequency")
    parser.add_argument("--max-carrier-freq", type=float, default=1000.0, help="Maximum carrier frequency")
    parser.add_argument("--window-size", type=float, default=4.0, help="Analysis window size in seconds")
    parser.add_argument("--hop-size", type=float, default=1.0, help="Analysis hop size in seconds")
    parser.add_argument("--max-peaks", type=int, default=24, help="Candidate peaks per frame per channel")
    parser.add_argument("--max-pairs-per-frame", type=int, default=80, help="Candidate left/right pairs per frame")
    parser.add_argument("--analysis-sample-rate", type=int, default=4096, help="Temporary decode sample rate for analysis")
    parser.add_argument("--amp-tolerance-db", type=float, default=18.0, help="Maximum left/right amplitude mismatch")
    parser.add_argument("--relative-floor-db", type=float, default=60.0, help="Peak floor below frame maximum")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    detector = BinauralBeatDetector(
        min_duration=args.min_duration,
        min_beat_freq=args.min_beat_freq,
        max_beat_freq=args.max_beat_freq,
        min_carrier_freq=args.min_carrier_freq,
        max_carrier_freq=args.max_carrier_freq,
        window_size=args.window_size,
        hop_size=args.hop_size,
        max_peaks=args.max_peaks,
        max_pairs_per_frame=args.max_pairs_per_frame,
        analysis_sample_rate=args.analysis_sample_rate,
        amp_tolerance_db=args.amp_tolerance_db,
        relative_floor_db=args.relative_floor_db,
        debug=args.debug,
    )

    beats = detector.find_binaural_beats(
        args.file_path,
        start_time=args.start,
        end_time=args.end,
        visualization_path=args.visualize,
        summary_path=args.summary_image,
    )

    if args.json:
        print(json.dumps([asdict(beat) for beat in beats], indent=2))
        return

    if not beats:
        print("No sustained binaural beats detected.")
        return

    if not args.verbose:
        print("start   end     dur    base Hz   left Hz   right Hz  beat Hz  band   conf  int   score")
        print("------  ------  -----  --------  --------  --------  -------  -----  ----  ----  -----")
        for beat in beats:
            end_time = beat.start_time + beat.duration
            print(
                f"{beat.start_time:6.1f}  "
                f"{end_time:6.1f}  "
                f"{beat.duration:5.1f}  "
                f"{beat.base_freq:8.2f}  "
                f"{beat.left_freq:8.2f}  "
                f"{beat.right_freq:8.2f}  "
                f"{beat.beat_freq:7.2f}  "
                f"{beat.band:5s}  "
                f"{beat.confidence:4s}  "
                f"{beat.intensity:4s}  "
                f"{beat.intensity_score:5.1f}"
            )
        return

    for beat in beats:
        print("Found binaural beat:")
        print(f"  Start time: {beat.start_time:.2f}s")
        print(f"  Duration: {beat.duration:.2f}s")
        print(f"  Left frequency: {beat.left_freq:.2f}Hz")
        print(f"  Right frequency: {beat.right_freq:.2f}Hz")
        print(f"  Base frequency: {beat.base_freq:.2f}Hz")
        print(f"  Beat frequency: {beat.beat_freq:.2f}Hz")
        print(f"  Band: {beat.band}")
        print(f"  Confidence: {beat.confidence}")
        print(f"  Intensity: {beat.intensity}")
        print(f"  Intensity score: {beat.intensity_score:.1f}/100")
        print(f"  Prominence: {beat.prominence_db:.1f}dB")
        print(f"  Amplitude: {beat.amplitude:.1f}dB")
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import numpy as np
from scipy import signal
from scipy.io import wavfile
from pydub import AudioSegment
import tempfile
import os
from typing import Tuple, List, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

@dataclass
class BinauralBeat:
    """Class to store information about detected binaural beats"""
    start_time: float  # Start time in seconds
    duration: float    # Duration in seconds
    base_freq: float   # Base frequency (in left channel)
    beat_freq: float   # Beat frequency (difference between channels)
    amplitude: float   # Average amplitude of the beat

class BinauralBeatDetector:
    def __init__(
        self,
        min_duration: float = 20.0,
        max_beat_freq: float = 200.0,
        freq_tolerance: float = 2.0,  # Increased tolerance
        amp_tolerance: float = 0.3,   # Increased tolerance
        debug: bool = False
    ):
        """
        Initialize the binaural beat detector with configuration parameters.
        
        Args:
            min_duration: Minimum duration (in seconds) for a valid binaural beat
            max_beat_freq: Maximum allowed frequency difference between channels
            freq_tolerance: Tolerance for frequency matching (Hz)
            amp_tolerance: Tolerance for amplitude matching (relative)
        """
        self.min_duration = min_duration
        self.max_beat_freq = max_beat_freq
        self.freq_tolerance = freq_tolerance
        self.amp_tolerance = amp_tolerance
        self.debug = debug

    def load_audio(
        self,
        file_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Load and prepare audio file for analysis.
        
        Args:
            file_path: Path to the MP3 file
            start_time: Start time in seconds for analysis
            end_time: End time in seconds for analysis
            
        Returns:
            Tuple of (left_channel, right_channel, sample_rate)
        """
        # Load MP3 using pydub
        audio = AudioSegment.from_mp3(file_path)
        
        # Extract segment if specified
        if start_time is not None or end_time is not None:
            start_ms = int(start_time * 1000) if start_time else 0
            end_ms = int(end_time * 1000) if end_time else None
            audio = audio[start_ms:end_ms]
        
        # Export to temporary WAV file for scipy to read
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            audio.export(temp_wav.name, format='wav')
            sample_rate, data = wavfile.read(temp_wav.name)
        
        # Clean up temporary file
        os.unlink(temp_wav.name)
        
        # Ensure stereo
        if len(data.shape) == 1:
            raise ValueError("Input audio must be stereo")
        
        # Convert to float32 and normalize
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
        
        return data[:, 0], data[:, 1], sample_rate

    def get_spectrogram(
        self,
        channel: np.ndarray,
        sr: float,
        window_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the spectrogram of an audio channel.
        
        Args:
            channel: Audio channel data
            sr: Sample rate
            window_size: Size of analysis window in seconds
            
        Returns:
            Tuple of (frequencies, times, spectrogram in dB)
        """
        # Calculate window parameters
        nperseg = int(window_size * sr)
        noverlap = nperseg // 2
        
        # Compute spectrogram
        frequencies, times, Sxx = signal.spectrogram(
            channel,
            sr,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann',
            scaling='spectrum'
        )
        
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + np.finfo(float).eps)
        
        return frequencies, times, Sxx_db

    def detect_continuous_frequency(
        self,
        channel: np.ndarray,
        sr: float,
        window_size: float = 0.2,  # Increased window size
    ) -> List[Tuple[float, float, float, float]]:
        """
        Detect continuous frequencies in a single channel.
        
        Args:
            channel: Audio channel data
            sr: Sample rate
            window_size: Size of analysis window in seconds
            
        Returns:
            List of tuples (start_time, duration, frequency, amplitude)
        """
        # Calculate window parameters
        nperseg = int(window_size * sr)
        noverlap = nperseg // 2
        
        # Compute spectrogram
        frequencies, times, Sxx = signal.spectrogram(
            channel,
            sr,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann',
            scaling='spectrum'
        )
        
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + np.finfo(float).eps)
        
        # Find peaks in each time frame
        continuous_freqs = []
        current_freq = None
        start_frame = 0
        
        for frame in range(Sxx_db.shape[1]):
            # Find the top N strongest frequencies in current frame
            n_peaks = 3  # Consider top 3 peaks
            peak_indices = np.argpartition(Sxx_db[:, frame], -n_peaks)[-n_peaks:]
            peak_indices = peak_indices[np.argsort(-Sxx_db[peak_indices, frame])]
            
            # Get the strongest peak that's not too close to 0 Hz
            peak_idx = peak_indices[0]
            freq = frequencies[peak_idx]
            amp = Sxx_db[peak_idx, frame]
            
            # Skip very low frequencies (usually noise)
            min_freq = 20  # Hz
            for idx in peak_indices:
                if frequencies[idx] >= min_freq:
                    peak_idx = idx
                    freq = frequencies[idx]
                    amp = Sxx_db[idx, frame]
                    break
            
            if self.debug:
                print(f"Frame {frame}: Found frequency {freq:.1f} Hz at amplitude {amp:.1f} dB")
            
            if current_freq is None:
                current_freq = freq
                start_frame = frame
            elif abs(freq - current_freq) <= self.freq_tolerance:
                continue
            else:
                # Pattern ended, check if it meets minimum duration
                duration = (frame - start_frame) * (nperseg - noverlap) / sr
                if duration >= self.min_duration:
                    continuous_freqs.append((
                        start_frame * (nperseg - noverlap) / sr,
                        duration,
                        current_freq,
                        np.mean(Sxx_db[peak_idx, start_frame:frame])
                    ))
                current_freq = freq
                start_frame = frame
        
        return continuous_freqs

    def visualize_analysis(
        self,
        left: np.ndarray,
        right: np.ndarray,
        sr: float,
        beats: List[BinauralBeat],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create visualization of the analysis results.
        
        Args:
            left: Left channel data
            right: Right channel data
            sr: Sample rate
            beats: Detected binaural beats
            save_path: Optional path to save the plot
        """
        # Create spectrograms
        f_left, t_left, Sxx_left = self.get_spectrogram(left, sr)
        f_right, t_right, Sxx_right = self.get_spectrogram(right, sr)
        
        # Create the visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Plot spectrograms
        max_freq = 500  # Only show up to 200 Hz for better visibility
        freq_mask = f_left <= max_freq
        
        im1 = ax1.pcolormesh(t_left, f_left[freq_mask], Sxx_left[freq_mask], shading='gouraud')
        im2 = ax2.pcolormesh(t_right, f_right[freq_mask], Sxx_right[freq_mask], shading='gouraud')
        
        # Add colorbars
        plt.colorbar(im1, ax=ax1, label='Power (dB)')
        plt.colorbar(im2, ax=ax2, label='Power (dB)')
        
        # Highlight binaural beats
        for beat in beats:
            # Draw rectangles for the duration of each beat
            rect1 = plt.Rectangle(
                (beat.start_time, beat.base_freq - 5),
                beat.duration,
                10,
                facecolor='none',
                edgecolor='red',
                linewidth=2
            )
            rect2 = plt.Rectangle(
                (beat.start_time, beat.base_freq + beat.beat_freq - 5),
                beat.duration,
                10,
                facecolor='none',
                edgecolor='red',
                linewidth=2
            )
            ax1.add_patch(rect1)
            ax2.add_patch(rect2)
        
        # Labels and titles
        ax1.set_ylabel('Frequency (Hz)')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_xlabel('Time (s)')
        ax1.set_title('Left Channel Spectrogram')
        ax2.set_title('Right Channel Spectrogram')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()

    def find_binaural_beats(
        self,
        file_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        visualization_path: Optional[str] = None
    ) -> List[BinauralBeat]:
        """
        Find binaural beats in the audio file.
        
        Args:
            file_path: Path to the MP3 file
            start_time: Start time in seconds for analysis
            end_time: End time in seconds for analysis
            
        Returns:
            List of detected BinauralBeat objects
        """
        # Load and prepare audio
        left, right, sr = self.load_audio(file_path, start_time, end_time)
        
        # Detect continuous frequencies in both channels
        left_patterns = self.detect_continuous_frequency(left, sr)
        right_patterns = self.detect_continuous_frequency(right, sr)
        
        # Match patterns between channels to find binaural beats
        binaural_beats = []
        
        for l_start, l_duration, l_freq, l_amp in left_patterns:
            for r_start, r_duration, r_freq, r_amp in right_patterns:
                # Check if patterns align in time
                if abs(l_start - r_start) > self.freq_tolerance:
                    continue
                if abs(l_duration - r_duration) > self.freq_tolerance:
                    continue
                    
                # Check frequency difference
                beat_freq = abs(l_freq - r_freq)
                if beat_freq > self.max_beat_freq:
                    continue
                    
                # Check amplitude matching
                if abs(l_amp - r_amp) / max(abs(l_amp), abs(r_amp)) > self.amp_tolerance:
                    continue
                    
                # We found a binaural beat
                binaural_beats.append(BinauralBeat(
                    start_time=l_start,
                    duration=l_duration,
                    base_freq=min(l_freq, r_freq),
                    beat_freq=beat_freq,
                    amplitude=(l_amp + r_amp) / 2
                ))
        # Create visualization if requested
        if visualization_path:
            self.visualize_analysis(left, right, sr, binaural_beats, visualization_path)
                
        return binaural_beats

def main():
    parser = argparse.ArgumentParser(description='Detect binaural beats in MP3 files')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--visualize', help='Save visualization to specified path')
    parser.add_argument('file_path', help='Path to the MP3 file')
    parser.add_argument('--start', type=float, help='Start time in seconds')
    parser.add_argument('--end', type=float, help='End time in seconds')
    parser.add_argument('--min-duration', type=float, default=20.0,
                        help='Minimum duration for binaural beats (seconds)')
    parser.add_argument('--max-beat-freq', type=float, default=100.0,
                        help='Maximum allowed beat frequency (Hz)')
    args = parser.parse_args()
    
    detector = BinauralBeatDetector(
        min_duration=args.min_duration,
        max_beat_freq=args.max_beat_freq,
        debug=args.debug
    )
    
    beats = detector.find_binaural_beats(
        args.file_path,
        start_time=args.start,
        end_time=args.end,
        visualization_path=args.visualize
    )
    
    # Print results
    for beat in beats:
        print(f"Found binaural beat:")
        print(f"  Start time: {beat.start_time:.2f}s")
        print(f"  Duration: {beat.duration:.2f}s")
        print(f"  Base frequency: {beat.base_freq:.1f}Hz")
        print(f"  Beat frequency: {beat.beat_freq:.1f}Hz")
        print(f"  Amplitude: {beat.amplitude:.1f}dB")
        print()

if __name__ == '__main__':
    main()

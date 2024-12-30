#!/opt/homebrew/opt/python@3/libexec/bin/python

import argparse
import numpy as np
import sounddevice as sd

def generate_binaural_beats_stream(base_frequency, beat_frequency, sample_rate=44100):
    """
    Generate and play binaural beats continuously using a stream.

    Parameters:
        base_frequency (float): Base frequency in Hz (e.g., 200 Hz).
        beat_frequency (float): Beat frequency in Hz (e.g., 10 Hz).
        sample_rate (int): Sample rate in Hz (default is 44100 Hz).
    """
    def callback(outdata, frames, time, status):
        t = np.arange(frames) / sample_rate + callback.t
        left_channel = np.sin(2 * np.pi * base_frequency * t)
        right_channel = np.sin(2 * np.pi * (base_frequency + beat_frequency) * t)
        outdata[:, 0] = left_channel
        outdata[:, 1] = right_channel
        callback.t = t[-1] + 1 / sample_rate

    callback.t = 0

    print("Playing binaural beats indefinitely. Press Ctrl+C to stop.")
    try:
        with sd.OutputStream(channels=2, samplerate=sample_rate, callback=callback):
            input()  # Keep the stream open until interrupted
    except KeyboardInterrupt:
        print("\nPlayback stopped.")

def main():
    parser = argparse.ArgumentParser(description="Binaural Beats Generator")
    parser.add_argument("--base-frequency", type=float, help="Base frequency in Hz (e.g., 200)")
    parser.add_argument("--beat-frequency", type=float, help="Beat frequency in Hz (e.g., 10)")
    parser.add_argument("--duration", type=float, help="Duration in seconds (e.g., 30). If omitted, plays indefinitely.")
    parser.add_argument("--hemi-sync", action="store_true", help="Play a Hemi-Sync compatible beat (defaults to 100 Hz base and 4 Hz beat)")

    args = parser.parse_args()

    if args.hemi_sync:
        print("Hemi-Sync mode activated: Base frequency = 100 Hz, Beat frequency = 4 Hz")
        base_frequency = 100.0
        beat_frequency = 4.0
    else:
        if args.base_frequency is None or args.beat_frequency is None:
            parser.error("--base-frequency and --beat-frequency are required unless --hemi-sync is specified.")
        base_frequency = args.base_frequency
        beat_frequency = args.beat_frequency

    if args.duration:
        # Generate a fixed duration signal
        t = np.linspace(0, args.duration, int(44100 * args.duration), endpoint=False)
        left_channel = np.sin(2 * np.pi * base_frequency * t)
        right_channel = np.sin(2 * np.pi * (base_frequency + beat_frequency) * t)
        stereo_signal = np.stack((left_channel, right_channel), axis=-1)
        print("Playing binaural beats...")
        sd.play(stereo_signal, samplerate=44100)
        sd.wait()
        print("Done.")
    else:
        # Play indefinitely using a stream
        generate_binaural_beats_stream(base_frequency, beat_frequency)

if __name__ == "__main__":
    main()


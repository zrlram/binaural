#!/opt/homebrew/opt/python@3/libexec/bin/python

import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf

def generate_binaural_wave(base_frequency, beat_frequency, duration, sample_rate=44100):
    """
    Generate binaural beats as a stereo waveform.

    Parameters:
        base_frequency (float): Base frequency in Hz.
        beat_frequency (float): Beat frequency in Hz.
        duration (float): Duration in seconds.
        sample_rate (int): Sample rate in Hz (default is 44100 Hz).

    Returns:
        np.ndarray: Stereo waveform containing the binaural beats.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    left_channel = np.sin(2 * np.pi * base_frequency * t)
    right_channel = np.sin(2 * np.pi * (base_frequency + beat_frequency) * t)
    return np.stack((left_channel, right_channel), axis=-1)

def normalize_audio(audio):
    """
    Normalize audio to prevent clipping.

    Parameters:
        audio (np.ndarray): Audio waveform to normalize.

    Returns:
        np.ndarray: Normalized audio waveform.
    """
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio /= max_val
    return audio

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

def overlay_binaural_beats_on_audio(base_frequency, beat_frequency, input_audio, sample_rate=44100, volume=0.5):
    """
    Overlay binaural beats on existing audio data.

    Parameters:
        base_frequency (float): Base frequency in Hz.
        beat_frequency (float): Beat frequency in Hz.
        input_audio (np.ndarray): Input audio waveform.
        sample_rate (int): Sample rate in Hz (default is 44100 Hz).
        volume (float): Volume of the binaural beats (0.0 to 1.0).

    Returns:
        np.ndarray: Combined audio with binaural beats overlaid.
    """
    duration = input_audio.shape[0] / sample_rate
    binaural_beats = generate_binaural_wave(base_frequency, beat_frequency, duration, sample_rate)
    binaural_beats *= volume  # Adjust the volume of the binaural beats
    overlaid_audio = input_audio + binaural_beats
    return normalize_audio(overlaid_audio)

def play_binaural_beats_with_audio(base_frequency, beat_frequency, audio_file, sample_rate=44100, volume=0.5):
    """
    Play binaural beats overlaid on an audio file.

    Parameters:
        base_frequency (float): Base frequency in Hz (e.g., 200 Hz).
        beat_frequency (float): Beat frequency in Hz (e.g., 10 Hz).
        audio_file (str): Path to the audio file (MP3 or other formats).
        sample_rate (int): Sample rate in Hz (default is 44100 Hz).
        volume (float): Volume of the binaural beats (0.0 to 1.0).
    """
    data, file_sample_rate = sf.read(audio_file)
    if file_sample_rate != sample_rate:
        raise ValueError("Audio file sample rate does not match the expected sample rate.")

    if len(data.shape) == 1:  # If mono audio, convert to stereo
        data = np.stack((data, data), axis=-1)

    overlaid_audio = overlay_binaural_beats_on_audio(base_frequency, beat_frequency, data, sample_rate, volume)

    print("Playing audio with binaural beats...")
    sd.play(overlaid_audio, samplerate=sample_rate)
    sd.wait()
    print("Done.")

def overlay_and_save_binaural_beats(base_frequency, beat_frequency, input_file, output_file, sample_rate=44100, volume=0.5):
    """
    Overlay binaural beats on an input audio file and save the result to an output file.

    Parameters:
        base_frequency (float): Base frequency in Hz (e.g., 200 Hz).
        beat_frequency (float): Beat frequency in Hz (e.g., 10 Hz).
        input_file (str): Path to the input audio file.
        output_file (str): Path to the output audio file.
        sample_rate (int): Sample rate in Hz (default is 44100 Hz).
        volume (float): Volume of the binaural beats (0.0 to 1.0).
    """
    data, file_sample_rate = sf.read(input_file)
    if file_sample_rate != sample_rate:
        raise ValueError("Audio file sample rate does not match the expected sample rate.")

    if len(data.shape) == 1:  # If mono audio, convert to stereo
        data = np.stack((data, data), axis=-1)

    overlaid_audio = overlay_binaural_beats_on_audio(base_frequency, beat_frequency, data, sample_rate, volume)

    sf.write(output_file, overlaid_audio, sample_rate)
    print(f"Saved overlaid audio to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Binaural Beats Generator")
    parser.add_argument("--base-frequency", type=float, help="Base frequency in Hz (e.g., 200)")
    parser.add_argument("--beat-frequency", type=float, help="Beat frequency in Hz (e.g., 10)")
    parser.add_argument("--duration", type=float, help="Duration in seconds (e.g., 30). If omitted, plays indefinitely.")
    parser.add_argument("--hemi-sync", action="store_true", help="Play a Hemi-Sync compatible beat (defaults to 100 Hz base and 4 Hz beat)")
    parser.add_argument("--audio-file", type=str, help="Path to an audio file (e.g., MP3) to overlay binaural beats on.")
    parser.add_argument("--input-file", type=str, help="Path to an input audio file for overlaying binaural beats.")
    parser.add_argument("--output-file", type=str, help="Path to save the output audio file.")
    parser.add_argument("--output-format", type=str, choices=["wav", "flac"], default="wav", help="Output audio file format (wav or flac). Default is wav.")
    parser.add_argument("--volume", type=float, default=0.5, help="Volume of the binaural beats (0.0 to 1.0). Default is 0.5.")

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

    if args.input_file and args.output_file:
        overlay_and_save_binaural_beats(base_frequency, beat_frequency, args.input_file, args.output_file, volume=args.volume)
    elif args.audio_file:
        play_binaural_beats_with_audio(base_frequency, beat_frequency, args.audio_file, volume=args.volume)
    elif args.duration:
        # Generate a fixed duration signal
        stereo_signal = generate_binaural_wave(base_frequency, beat_frequency, args.duration)
        print("Playing binaural beats...")
        sd.play(stereo_signal, samplerate=44100)
        sd.wait()
        print("Done.")
    else:
        generate_binaural_beats_stream(base_frequency, beat_frequency)

if __name__ == "__main__":
    main()

# Binaural Beats Agent

## Mission

This workspace is for practical binaural-beat experimentation: generating stereo tones, overlaying them on music, and analyzing stereo music/audio files to estimate whether binaural beats are present.

An agent working here should become useful in three areas:

1. Binaural beat fundamentals: carrier frequencies, interaural frequency differences, common brainwave-band labels, and the limits of the evidence.
2. Audio generation: producing clean stereo tones or overlays with controlled carrier frequency, beat frequency, duration, volume, fades, sample rate, and output format.
3. Audio analysis: inspecting stereo files for sustained left/right frequency offsets, estimating carrier and beat frequencies, visualizing spectrograms, and explaining uncertainty.

## Current Workspace

- `README.md`: usage examples and reference links.
- `add_binaural.py`: generates and plays binaural beats, overlays beats on existing audio, and saves output audio.
- `detect_beat.py`: decodes audio with `ffmpeg`, analyzes stereo channels with spectrograms, detects sustained channel frequency differences, and can save visualizations.
- `requirements.txt`: Python dependencies for the existing scripts.

## Setup

Use a local virtual environment for Python work:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

For MP3 decoding, make sure `ffmpeg` is installed and available on `PATH`.

Prefer running project scripts through the local environment:

```bash
.venv/bin/python add_binaural.py --base-frequency 200 --beat-frequency 10 --duration 30
.venv/bin/python detect_beat.py input.mp3 --visualize analysis.png
```

## Binaural Beat Concepts

A binaural beat is created when the left and right ears receive nearby but different pure-tone frequencies. The perceived beat frequency is the absolute difference between the two channels:

```text
left carrier = 200 Hz
right carrier = 204 Hz
beat = 4 Hz
```

Keep generation and analysis language precise:

- Carrier frequency: the audible tone in one channel, usually between about 80 Hz and 900 Hz for this project.
- Beat frequency: the left/right difference, usually below about 40 Hz for classic binaural-beat use.
- Base frequency: this code usually treats the left channel as base and the right channel as base plus beat.
- Monaural beats are amplitude beats mixed before playback; binaural beats require stereo separation.
- Isochronic tones are pulsed amplitude tones; they are not binaural beats.

Common reference bands:

- Delta: 0.5 to 4 Hz, often associated with sleep or very deep relaxation.
- Theta: 4 to 8 Hz, often associated with meditation, imagery, and hypnagogic states.
- Alpha: 8 to 14 Hz, often associated with relaxed wakefulness.
- Beta: 14 to 30 Hz, often associated with alertness or focus.
- Gamma: above 30 Hz, sometimes associated with higher-frequency cognitive activity.

Treat these as conventional labels, not guaranteed outcomes. Avoid medical, therapeutic, or certainty claims unless a cited source directly supports them.

## Generation Workflow

Use or extend `add_binaural.py` for generation tasks.

Important behaviors:

- `--hemi-sync` currently uses a 100 Hz carrier and a 3.78 Hz beat.
- `--schumann` uses a 7.83 Hz beat frequency.
- `--solfeggio` cycles through 396, 417, 528, 639, 741, and 852 Hz carriers.
- `--audio-file` plays an overlay in memory.
- `--input-file` and `--output-file` save an overlay to disk.

Generation standards:

- Always produce stereo output for binaural beats.
- Normalize after mixing to avoid clipping.
- Use gentle fade-in/fade-out when adding longer generated material.
- Keep the beat layer volume explicit and conservative when overlaying music.
- Preserve sample rate intentionally; if resampling is needed, do it explicitly and document it.
- Prefer WAV or FLAC for analysis and intermediate files; MP3 is acceptable for convenience but can obscure fine analysis.

Useful examples:

```bash
.venv/bin/python add_binaural.py --base-frequency 200 --beat-frequency 10 --duration 30
.venv/bin/python add_binaural.py --hemi-sync --volume 0.5
.venv/bin/python add_binaural.py --solfeggio --schumann
.venv/bin/python add_binaural.py --input-file input.mp3 --output-file output.wav --hemi-sync
```

## Analysis Workflow

Use or extend `detect_beat.py` for detection tasks.

The core analysis question is:

```text
Are there sustained, channel-separated frequency components whose difference matches a plausible binaural beat?
```

Recommended analysis steps:

1. Confirm the input is stereo.
2. Convert to a lossless working representation when possible.
3. Analyze left and right channels separately.
4. Use spectrograms or STFT peak tracking to find sustained tones in each channel.
5. Match left/right tones by time span and amplitude.
6. Report carrier frequency, right/left frequency, beat frequency, start time, duration, amplitude, and confidence.
7. Save a visualization when useful.

Use focused time ranges for long files:

```bash
.venv/bin/python detect_beat.py input.mp3 --start 30 --end 180 --visualize analysis.png
```

Detection caveats:

- Music is dense; harmonics, bass lines, compression artifacts, stereo widening, chorus, phasing, and mastering effects can look like frequency offsets.
- MP3 encoding can blur or remove narrow components.
- A detected left/right frequency difference is evidence of a possible binaural beat, not proof of listener intent.
- Very low beat frequencies need enough duration and frequency resolution to estimate accurately.
- Pure-tone overlays are much easier to detect than beats embedded under complex music.

When reporting results, include uncertainty:

```text
Possible binaural beat from 42.0s to 132.0s:
- left carrier: 200.1 Hz
- right carrier: 203.9 Hz
- beat: 3.8 Hz
- confidence: medium, because the tones are sustained but partially masked by music
```

## Reference Sources From README

Use these as project references and preserve source links in research notes:

- Monroe Institute and Gateway Experience discussion: https://forums-archive.anarchy-online.com/showthread.php?536724-Monroe-Institute-Studies
- SBaGen binaural beat generator: https://uazu.net/sbagen/
- SBaGen documentation: https://uazu.net/sbagen/sbagen.txt
- Binaural Beats Visualization and Analysis: https://uazu.net/bavsa/
- CIA Gateway document: https://www.cia.gov/readingroom/docs/cia-rdp96-00788r001700210016-5.pdf

When making factual claims from these sources, fetch and verify the source first. Distinguish historical, experiential, engineering, and scientific claims.

## Implementation Priorities

Near-term improvements that fit this workspace:

- Add a first-class `--output-file` path for generated pure tones without requiring input audio.
- Add fade-in/fade-out controls for generated and overlaid audio.
- Add explicit sample-rate conversion instead of rejecting mismatched input files.
- Support WAV/FLAC input directly in `detect_beat.py`, not only MP3.
- Improve detection confidence scoring and JSON/CSV output.
- Add synthetic fixture generation so detection can be tested against known beats.
- Add tests that generate known stereo tones and verify detected beat frequency.

## Engineering Rules

- Prefer the existing Python scripts and data model before adding new tooling.
- Use `numpy`, `scipy`, `soundfile`, `ffmpeg`, and `matplotlib` for audio and analysis work unless there is a clear reason to add another library.
- If a Python dependency is needed, add it to `requirements.txt` and install it into `.venv`.
- Keep audio processing deterministic and parameterized.
- Avoid destructive edits to user audio files; write new output files.
- Do not commit generated audio, large visualizations, or temporary analysis files unless the user explicitly asks.
- For significant analysis changes, create or update tests using generated known-frequency audio.

## Personal Data

This project does not require personal data. Do not read or use global personal-data files unless a future task explicitly requires them and authorization has been verified according to the global rules.

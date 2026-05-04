# Binaural Experiments

Tools for generating binaural beats and detecting prolonged binaural-beat sections in stereo audio files.

## Setup

Create and install the local Python environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

MP3 and general audio decoding for `detect_beat.py` uses `ffmpeg`, so make sure it is installed and available on `PATH`.

## Generate Binaural Beats

Play a generated beat:

```bash
.venv/bin/python add_binaural.py --beat-frequency 10 --base-frequency 200
```

Use the Hemi-Sync-style default:

```bash
.venv/bin/python add_binaural.py --hemi-sync
.venv/bin/python add_binaural.py --hemi-sync --volume 0.5
```

Use the Schumann beat frequency:

```bash
.venv/bin/python add_binaural.py --base-frequency 200 --schumann
```

Play through the Solfeggio carrier frequencies:

```bash
.venv/bin/python add_binaural.py --solfeggio --schumann
.venv/bin/python add_binaural.py --solfeggio --beat-frequency 1
```

Overlay beats on an audio file for playback:

```bash
.venv/bin/python add_binaural.py --hemi-sync --audio-file audio.mp3
```

Save an overlay to a new file:

```bash
.venv/bin/python add_binaural.py --input-file input.mp3 --output-file output.wav --hemi-sync
```

## Detect Binaural Beats

Run the detector on a stereo audio file:

```bash
.venv/bin/python detect_beat.py input.mp3
```

The detector looks for prolonged sections where one ear has a base tone and the other ear has a sustained shifted tone. By default, sections must last at least 20 seconds.

Example output:

```text
start   end     dur    base Hz   left Hz   right Hz  beat Hz  band   conf  int   score
------  ------  -----  --------  --------  --------  -------  -----  ----  ----  -----
   0.0    60.0   60.0    200.00    200.00    204.00     4.00  theta  high  high  100.0
  60.0   120.0   60.0    250.00    250.00    260.00    10.00  alpha  high  high  100.0
```

`conf` is detection confidence: how stable and continuous the tracked pair is. `int` and `score` are signal intensity: how prominent the carrier tones are against nearby audio, expressed as a label and a 0-100 score. This is not a claim about physiological effect strength.

Analyze only part of a file:

```bash
.venv/bin/python detect_beat.py input.mp3 --start 60 --end 180
```

The reported timestamps remain relative to the original file timeline.

Save structured JSON output:

```bash
.venv/bin/python detect_beat.py input.mp3 --json
```

Print the older detailed multi-line output:

```bash
.venv/bin/python detect_beat.py input.mp3 --verbose
```

Verbose output includes raw `Prominence` in dB, `Intensity score`, and average spectral `Amplitude` in dB.

Save optional image outputs:

```bash
.venv/bin/python detect_beat.py input.mp3 --visualize analysis.png
.venv/bin/python detect_beat.py input.mp3 --summary-image summary.png
```

Tune duration and frequency thresholds:

```bash
.venv/bin/python detect_beat.py input.mp3 \
  --min-duration 15 \
  --min-beat-freq 0.5 \
  --max-beat-freq 40 \
  --min-carrier-freq 40 \
  --max-carrier-freq 1000
```

Useful advanced options:

- `--analysis-sample-rate`: temporary decode rate used for analysis. The default is `4096`, which is much faster than analyzing full 44.1 kHz audio and is enough for the default `--max-carrier-freq 1000`.
- `--window-size`: larger values improve frequency precision but smear time boundaries more.
- `--hop-size`: smaller values improve time tracking but increase runtime.
- `--max-peaks`: number of candidate tones retained per channel per frame.
- `--max-pairs-per-frame`: maximum left/right candidate pairs tracked per frame.
- `--relative-floor-db`: how far below the loudest frame peak to keep candidate tones. Increase this to find quieter binaural carriers under music.

Fast scan example:

```bash
.venv/bin/python detect_beat.py input.mp3 \
  --max-peaks 8 \
  --max-pairs-per-frame 30 \
  --relative-floor-db 35 \
  --hop-size 2
```

## Beat Bands

The detector labels beat frequencies with conventional bands:

- Delta: 0.5 to below 4 Hz
- Theta: 4 to below 8 Hz
- Alpha: 8 to below 14 Hz
- Beta: 14 to below 30 Hz
- Gamma: 30 Hz and above

These are descriptive labels, not medical or therapeutic claims.

## Test

Run the generated-audio test suite:

```bash
.venv/bin/python -m unittest discover -v
```

The tests generate known binaural sections using `add_binaural.generate_binaural_wave()` and verify detection of:

- Multiple beat sections over time.
- Alpha/theta/beta labeling.
- Quiet binaural beats under a louder centered tone.
- Rejection of identical stereo tones as `0 Hz` false positives.
- Duration threshold behavior.
- Timestamp handling with `--start`.
- Non-integer beats such as `3.78 Hz`.

## Gateway Experience as a Reference

### Common Binaural Frequencies in the Gateway Experience

1. Delta waves, 0.5 to 4 Hz:
   Associated with deep relaxation, sleep, and unconscious states.
2. Theta waves, 4 to 8 Hz:
   Associated with meditation, imagery, and subconscious access.
3. Alpha waves, 8 to 14 Hz:
   Associated with relaxed wakefulness and calm attention.
4. Gamma waves, 30 Hz and above:
   Sometimes associated with higher-frequency cognitive activity.

### Specific Binaural Beat Frequencies

- 3.5 Hz: often discussed as a deep meditation beat.
- 4 Hz: often discussed around theta/deep meditation use.
- 7.83 Hz: Schumann resonance reference frequency.

## Further Reference Sources

- https://forums-archive.anarchy-online.com/showthread.php?536724-Monroe-Institute-Studies
- https://uazu.net/sbagen/
- https://uazu.net/sbagen/sbagen.txt
- https://uazu.net/bavsa/
- https://www.cia.gov/readingroom/docs/cia-rdp96-00788r001700210016-5.pdf

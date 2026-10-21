# Shannon

Offline transcription of interview recordings, with speaker labels.
Nothing is uploaded. No account, no API key, no subscription.

## Status

Working on macOS. Not yet tested on Windows (ticket 07).

| | |
|---|---|
| Transcription | **12.5% content WER** vs human transcripts (mock GP consultations). |
| Speaker labelling | **91.4% of words on the correct speaker.** |
| Speed | **0.66x real time** on an M4, dense clinical speech, `large-v3-turbo`. |
| Install | 219 MB packages + 1.7 GB models. No PyTorch, no ffmpeg to install. |
| Cost | $0. MIT / Apache-2.0 / CC-BY-4.0 throughout, commercial use permitted. |

## Install

    uv venv --python 3.12 .venv
    uv pip install --python .venv faster-whisper sherpa-onnx

Models (~536 MB) download once on first run, then it works offline forever.
Ship the `models/` folder with the tool and the target machine never touches
Hugging Face at all.

## Use

    .venv/bin/python transcribe.py recording.m4a --speakers 2

Takes any audio or video file: Teams `.mp4`, phone `.m4a`/`.mp3`, `.wav`.
Writes a Markdown transcript beside the audio.

    --speakers N   how many people are on the recording (required)
    --model NAME   default large-v3-turbo. Drop to small or base if impatient.
    --out PATH     where to write the transcript
    --threads N    leave it alone. Raising it makes things SLOWER.

## How long will it take

Measured on an Apple M4, dense two-person consultation, int8:

| model | M4 | est. 60-min file on an i7-1255U | WER (Australian) |
|---|---|---|---|
| base | 0.20x | ~0.5 h | ~12%, unstable |
| small | 0.49x | ~1.0 h | 8.0% |
| **large-v3-turbo** (default) | **0.57x** | **~1.1 h** | **~5.2%** |
| medium | 1.07x | ~2.0 h | 6.4% |

Plus 0.09x for diarization. **`large-v3-turbo` costs only 15% more than `small`
and is much more accurate** - it keeps large-v3's full encoder (which is what
hears the accent) with a cheap 4-layer decoder. **`medium` is pointless**: slower
than turbo *and* worse.

Peak memory is 1.42 GB on a 60-minute file, measured at full length.

Do not raise `--threads`. Measured: 8 threads is 2.7x slower, 12 is 4.6x slower.

## Speaker labelling

Pass `--speakers` accurately; it is the single biggest lever on quality.

Measured on PriMock57 mock GP consultations (diarization error rate, 0.25 s
collar): **18.0% / 12.0% / 9.1%**, with speaker confusion at 5.8% / 5.3% / 3.3%.
For comparison the ceiling for this segmentation model is 13.0% / 6.9% / 7.2%.

(sherpa-onnx's built-in clustering fails badly here - it chains along the time
axis and puts ~99% of speech on one speaker. This script runs the segmentation
model directly and clusters with k-means instead.)

Speakers come out as `SPEAKER_00`, `SPEAKER_01`. Which is the clinician is for
you to work out from the text - usually obvious from the first question.

Expect worse results with heavy overlapping speech, three or more people on one
microphone, or a phone far from the speakers.

## Privacy

- No network access at runtime once the models are on disk.
- Audio and transcripts never leave the machine.
- The script **refuses to write into OneDrive, Dropbox or Google Drive folders**.
  Windows 11 puts Desktop and Documents inside OneDrive by default, which would
  silently upload clinical transcripts.
- Language is pinned to English; auto-detection is a known cause of garbled output.
- Whisper can invent text during silence. VAD filtering reduces this but does not
  eliminate it. **Transcripts need a human read before being trusted** - see below.

Note: Microsoft Word's Transcribe feature uploads audio to Microsoft 365. That is
what this tool replaces.

## Accuracy, measured

Against PriMock57's human transcripts (3 mock GP consultations, `large-v3-turbo`):

| | day1_c01 | day1_c02 | day2_c01 | mean |
|---|---|---|---|---|
| Words on the correct speaker | 87.6% | 93.0% | 93.6% | **91.4%** |
| Content WER | 12.9% | 12.8% | 11.9% | **12.5%** |
| Raw WER | 23.1% | 17.2% | 20.7% | 20.3% |

"Content WER" excludes filler words (`um`, `uh`, `mm`), the reference's own
`[unsure]` / `[inaudible]` annotation markers, and spelling conventions
(`ok`/`okay`, `diarrhea`/`diarrhoea`). Raw WER counts all of those as errors,
which is why it looks worse than the transcript reads.

### Why you must still read the transcript

Most remaining errors are harmless function words (`the`/`a`, `will`/`would`).
But a small number are not, and these were observed in the test set:

    cannot -> of        negation lost
    left   -> that      laterality lost
    start  -> stop      instruction reversed
    sweats -> no        symptom dropped

Roughly one or two per consultation. They are rare, but they sit inside fluent,
confident-looking text where nothing signals that something is wrong. **Do not
put an unreviewed transcript into a clinical record.**

## Known issues

- **Untested on Windows.** The likely first problem is an ONNX Runtime version
  clash: sherpa-onnx bundles 1.28.2, faster-whisper pulls 1.31.0, and Windows
  resolves DLLs by base name. If it misbehaves, import `onnxruntime` before
  `sherpa_onnx`, or pin `onnxruntime==1.28.2`.
- **Peak memory on a 60-minute file is unmeasured.** Should be modest, but check
  before trusting it on a 16 GB machine.

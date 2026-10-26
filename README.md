# Health Coach Voice Prototype

A local, voice-first health coach that runs entirely on one computer: Moonshine for
speech-to-text, Gemma 3 4B (through Ollama) for language understanding and plan
answers, and Piper for the spoken replies. It covers the seven user stories and shows,
turn by turn, where the time goes between the patient finishing a sentence and hearing
the first word of the answer.

The laptop stands in for the phone. The point is to show the client realistic response
times for this pipeline, how the no-diagnosis rule is enforced, and how the slow
API/service-bus path can be kept out of the conversation.

## What it shows

| Story | What to look at |
|---|---|
| 1. Log measurements by voice | Blood pressure, weight, glucose and heart rate, in English or Spanish, spoken numbers included ("one twenty over eighty", "ciento veinte sobre ochenta"). The assistant reads the value back with units and asks before saving. Corrections work ("no, it was 128"). |
| 2. Appointments by voice | "Next Tuesday afternoon" is resolved to a date in code and read back. When the slot is taken, it offers the closest openings. Bookings save locally at once. |
| 3. Plan answers offline | Turn the network off on the Sync tab and ask about the plan. Answers come only from the on-device copy of the patient's plan. |
| 4. Charts | "Show me my blood pressure this month" draws a chart from stored readings only. |
| 5. Natural conversation | Hands-free: one tap starts the conversation, and after each reply the mic reopens on its own with a short tone, so the patient just keeps talking. Follow-ups work without repeating context: "and my weight?", "what about Wednesday?", "and how many sessions are left?". The latency panel shows each turn against the 2-second target. |
| 6. No diagnosis | Diagnosis, interpretation and treatment questions get a fixed refusal that explains why. The assistant has no way to edit the plan. |
| 7. Accurate capture | Every saved record has the reading, units, when it was measured, when it was saved, input method, language and the patient's own words. The Records tab has a care-team summary. |

## How it works

```
 mic ──► Moonshine (streaming, built-in pause detection)
            │ transcript
            ▼
        safety checks ──► emergency / crisis template
            │
        pending confirmation? ("yes", "the second one", "no, it was 130")
            │
        rules: measurement · refusal · chart · appointment · plan question
            │                                   (most turns end here, no model call)
            ▼
        Gemma 3 router (JSON) ── only when the rules cannot place the turn
            │
        handlers ──► SQLite on the device ──► outbox ──► mock service bus (2 min) ──► "clinic"
            │
        reply: fixed template, or Gemma plan answer streamed and checked sentence by sentence
            ▼
        Piper, one sentence at a time ──► speaker
```

Three design choices carry the latency:

1. **Rules before the model.** Read-backs, confirmations, booking and charts never wait
   for Gemma. The panel labels each turn "Rules only", "Rules + Gemma answer" or "Gemma".
2. **Streaming everywhere.** Speech is transcribed while the patient talks. Plan answers
   stream from Gemma, and the first sentence is spoken while the rest is generated.
3. **The service bus is out of the loop.** Writes go to the local database and an outbox.
   The patient hears "Done" immediately; the clinic copy follows after the configured
   delay (default 120 seconds, matching the client's current path). A conflict reported
   by the clinic later is shown as a notification.

The no-diagnosis rule is enforced in four layers: pattern checks on the patient's words
before any model runs, a `clinical_advice` label in the Gemma router, a plan-answer
prompt that must answer `NOT_IN_PLAN` or `REFUSE` when the plan does not cover the
question, and a check on every generated sentence before it is spoken (advice phrasing
and numbers that are not in the plan excerpt are blocked). Refusals are fixed text in
`app/templates.py`, never generated.

## Setup

You need: Windows 10/11 or Linux (x86-64), Python 3.11 or 3.12, Chrome or Edge, a
microphone, about 6 GB of free disk, and 8 GB of RAM or more (16 GB is more
comfortable). No GPU is needed.

### 1. Install Ollama and Gemma 3

Install Ollama from https://ollama.com/download and start it. Then:

```
ollama pull gemma3:4b
```

### 2. Create a Python environment and install packages

Windows (PowerShell):

```
cd health-voice-demo
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activate script, run
`Set-ExecutionPolicy -Scope Process RemoteSigned` first.

Linux:

```
cd health-voice-demo
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download the speech models and run the self-test

```
python setup_models.py
```

This downloads the Moonshine English and Spanish models and the two Piper voices into
`./models`, checks that Ollama has Gemma, and then runs a round trip: Piper speaks a
test sentence, Moonshine transcribes it, and Gemma classifies a sentence twice. It
prints the time each step took on your machine. **The Gemma timing is the best
predictor of how plan answers will feel in the demo.**

### 4. Start the demo

```
python run.py
```

Open http://127.0.0.1:8000 in Chrome or Edge and allow the microphone. Use the
address exactly as shown: browsers only allow the microphone on `localhost` or HTTPS.

## Portable USB version (run on another Windows laptop)

To demo on a computer with nothing installed, build a self-contained folder once on
your own Windows PC and copy it to a USB drive.

1. Finish setup steps 1 to 3 above on your PC (Ollama running, `gemma3:4b` pulled).
2. From the project folder, in PowerShell:
   ```
   powershell -ExecutionPolicy Bypass -File portable\build_portable.ps1
   ```
   It bundles the app with PyInstaller, checks the bundle loads every native library,
   downloads the no-install Ollama for Windows, and copies Gemma out of your Ollama.
   The result is `dist\HealthCoachDemo` (about 5 GB).
3. Test it on your PC: double-click `dist\HealthCoachDemo\Start Demo.bat`.
4. Copy the `HealthCoachDemo` folder to the USB drive.
5. On the other laptop, copy the folder to the Desktop (Gemma loads much faster from
   the internal disk than from USB) and double-click `Start Demo.bat`. The browser
   opens when everything has loaded. `README.txt` in the folder covers the rest.

The portable Ollama runs on its own port (11435), so it never conflicts with an
Ollama already installed on that laptop. Closing the demo window stops it; nothing is
left running or installed. Windows may show "Windows protected your PC" the first
time, because the program is not code-signed: choose "More info", then "Run anyway".
Do a full dry run on the other laptop before the meeting.

## Demo script for the meeting

About ten minutes. Keep the Latency tab visible.

Tap the mic once (or press Space) and speak. The turn ends on its own when you pause.
With **Hands-free** on (the default, under the mic), the mic reopens after each reply
with a short two-note tone and the hint "Your turn, I'm listening", so the whole script
below can run without touching the computer. To control it:

- **Interrupt** the assistant by tapping the mic while it is speaking.
- **Stop** the conversation by tapping the mic while it is listening, before you speak.
- **It pauses itself** after 8 seconds of silence, or two misheard turns in a row.
- **Typing** a message pauses hands-free; tap the mic to resume.

1. **Log a reading.** "My blood pressure was one twenty over eighty this morning." →
   read-back → "Yes." Open **Records**: timestamp, input method, the patient's words,
   and "queued" for sync.
2. **Correct a reading.** "I weigh 182." → "No, it was 180." → "Yes."
3. **Book an appointment.** "Book an appointment next Tuesday afternoon." The demo data
   keeps the first upcoming Tuesday afternoon full, so it offers alternatives. "The
   second one." → "Yes." Open **Sync & plan**: the booking is saved and counting down
   to delivery, while the conversation has already moved on. That countdown is the
   2-minute service-bus path, outside the conversation.
4. **Booking conflict.** Turn on "Simulate a booking conflict", set the delay to 20
   seconds, book again, and wait for the notification.
5. **Offline plan questions.** Turn the network off. "What does my plan cover for
   physiotherapy?" → "And how many sessions do I have left?" →
   "When should I take my lisinopril?"
6. **Charts.** "Show me a graph of my blood pressure this month." → "And my weight?"
7. **Guardrails.** "Do I have high blood pressure?" → refusal. "My blood pressure is
   one fifty over ninety-five, is that bad?" → it logs the reading but will not
   interpret it. "Can you change my plan to add massage?" → refusal and an offer to
   book. "I have chest pain right now." → emergency message.
8. **Spanish.** Switch to Español. "Mi presión fue ciento veinte sobre ochenta esta
   mañana." → "Sí." → "Quiero una cita el martes por la tarde." →
   "¿Qué cubre mi plan de fisioterapia?"
9. **The numbers.** On the Latency tab, compare the "Rule turns" and "Gemma turns"
   medians and download the CSV for the write-up.

## Reading the latency panel

The headline number is **speech end to first audio**: from the moment the patient
stops talking to the moment the first word of the reply is ready. Its parts:

| Stage | Meaning |
|---|---|
| Pause detection | How long Moonshine waited in silence before deciding the patient was done. Turning off "End turn on pause" and tapping to finish removes most of it. |
| Final transcript | Transcribing the last bit of audio after the pause. |
| Understand | Rules, plus the Gemma routing call when one was needed. |
| Generate answer | For plan questions: Gemma producing its first full sentence. |
| Synthesize voice | Piper producing the first sentence's audio. |

The Gemma prompt and output token counts and times come from Ollama. "Client play" in
the CSV is measured in the browser, from the end of input to the moment audio starts
playing.

For typed turns the clock starts when you press Send, so there is no pause or
transcript stage.

**What to expect on a CPU-only computer.** Rule turns should be short, because they
are mostly pause detection plus voice synthesis. Turns that need Gemma depend on the
CPU; the self-test numbers tell you in advance. If plan answers are slow, that is the
honest finding to present: the conversational flow is fast where the logic is
deterministic, and the language model is where on-device hardware matters. A phone
with a neural accelerator, a smaller model (`DEMO_LLM=gemma3:1b`), or the cloud
fallback from the architecture discussion are the levers.

## Settings

Set these as environment variables before `python run.py`.

| Variable | Default | Purpose |
|---|---|---|
| `DEMO_LLM` | `gemma3:4b` | Any Ollama model, e.g. `gemma3:1b` for a speed comparison |
| `DEMO_ASR_SIZE` | `small` | `tiny` for faster, less accurate speech recognition (rerun `setup_models.py`) |
| `DEMO_BUS_DELAY` | `120` | Service-bus delay in seconds (also adjustable in the page) |
| `DEMO_VOICE_EN` / `DEMO_VOICE_ES` | `en_US-lessac-medium` / `es_MX-ald-medium` | Piper voices (rerun `setup_models.py` after changing) |
| `DEMO_PORT` | `8000` | Server port |
| `DEMO_DB` | `demo.db` | Database file; delete it to start fresh |

Windows PowerShell example: `$env:DEMO_LLM="gemma3:1b"; python run.py`

## Where to change things

| File | Contents |
|---|---|
| `app/templates.py` | Every fixed reply, including refusals and emergency text (English and Spanish) |
| `app/guardrails.py` | Diagnosis, emergency and plan-change patterns; generated-sentence checks |
| `app/router.py` | Intent rules, the Gemma router prompt, the plan-answer prompt |
| `app/measurements.py` | Measurement parsing, units, plausibility ranges |
| `app/dates.py`, `app/scheduling.py` | Date phrases and slot search |
| `data/plan_en.md`, `data/plan_es.md` | The sample patient's plan (edit, then "Sync now") |

## Tests

```
python -m pytest -q
```

46 tests run the conversation logic and the server with stand-in models (no downloads
needed): read-backs and confirmations, corrections, booking and conflicts, offline
queueing, follow-ups across turns, refusals in both languages, everyday phrases that
must not be refused, and blocked ungrounded answers. They check the logic around the
models, not model quality or speed; the latency log from the real demo covers that.

`DEMO_FAKE=1 python run.py` starts the page with stand-in models, which is useful for
checking the interface on a machine without the models.

## Limits of this prototype

- **Licensing.** Moonshine's Spanish model is under the non-commercial Moonshine
  Community License; a product needs a commercial license from Moonshine AI or a
  different Spanish recognizer. Check each Piper voice's model card for its license.
- **Not clinically validated.** The guardrail patterns are a starting set for the
  clinical team to review and extend. The plausibility ranges catch misheard numbers;
  they are not clinical thresholds.
- **Separate plan copies per language.** Retrieval is keyword-based and each language
  has its own plan file. A multilingual embedding model would let one plan answer
  questions in either language.
- **No cloud fallback yet.** Everything here is local. The cloud verification step from
  the architecture discussion would slot in after the plan answer and on the sync path.
- **Single user, no sign-in.** The data is synthetic and stored in a local SQLite file.
- **Laptop, not phone.** Timings come from the demo computer. A phone port would use the
  same models through native runtimes (Moonshine and Piper both run on mobile; Gemma
  through an on-device runtime such as llama.cpp or MediaPipe).
- **Mixed-language sentences.** Each Moonshine model handles one language, so the
  patient picks English or Spanish; switching mid-sentence is not supported.

## Troubleshooting

- **Red "Speech in", "Gemma" or "Voice" chip.** Hover it for the reason, and read the
  banner. Usually: Ollama not running, Gemma not pulled, or `setup_models.py` not run.
- **Microphone does nothing.** Use `http://127.0.0.1:8000` or `http://localhost:8000`,
  allow the microphone in the address bar, and check the input device in system sound
  settings. The level bar under "Tap to talk" should move when you speak.
- **"WebSocket" errors in the browser console.** Reinstall with
  `pip install -r requirements.txt`; the `uvicorn[standard]` extra provides WebSocket
  support.
- **The turn ends too early or too late.** Turn off "End turn on pause" and tap the mic
  to finish.
- **Hands-free keeps reopening in a noisy room.** The mic only opens after the reply has
  finished playing, so the assistant does not hear itself, but background noise can be
  heard as speech. Use a headset, or turn off Hands-free under the mic.
- **First turn is slow.** Models load at startup and stay loaded; the first Gemma call
  after a long idle can still be slower. Ask one warm-up question before the meeting.

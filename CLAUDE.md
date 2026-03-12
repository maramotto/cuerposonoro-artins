# CLAUDE.md — cuerposonoro-jetson

This file is the persistent context for Claude Code sessions on this project.
Read it fully before doing anything.

---

## What this project is

`cuerposonoro-jetson` is an autonomous art installation that translates full-body
human movement into real-time sound. It runs on an NVIDIA Jetson Orin Nano with
no screen and no manual intervention: it boots and runs on power alone.

This is a **separate project** from `cuerposonoro` (the main thesis). It is an
adaptation for a specific physical context with conscious technical trade-offs.
Do not conflate the two.

The full design rationale — artistic and technical — is in `DESIGN.md`.
Read it before making any design decision.

---

## Hardware

- **Compute:** NVIDIA Jetson Orin Nano, JetPack 6.1, CUDA 12.6, TensorRT 10.3
- **Camera:** Logitech C922
- **Audio:** TPA3116D2 amplifier + Visaton FRS 13 speakers (8Ω)
- **No screen. No keyboard. No mouse at runtime.**

Development happens on Mac Apple Silicon. Code is deployed to the Jetson via SSH.

---

## Tech stack

| Component | Choice | Notes |
|---|---|---|
| Vision | YOLOv8-Pose | Exports to TensorRT natively. 17 COCO landmarks. |
| Audio engine | Fluidsynth (headless) | Runs as systemd service. No GUI. |
| Soundfont | JJazzLab SoundFont SF2 | Vibraphone (ch1) + Acoustic bass (ch2). |
| MIDI routing | ALSA virtual port | `aconnect` at startup. |
| Boot | systemd | Auto-start on power. No manual steps. |
| Language | Python 3.10 | Match Jetson JetPack 6.1 environment. |
| Testing | pytest + TDD | Tests before implementation. Always. |

---

## Sonic design (do not change without explicit confirmation)

### Philosophy
No movement = no sound. Silence is part of the work.

### Two MIDI channels
- **Channel 1:** vibraphone, melody, controlled by arms
- **Channel 2:** acoustic bass pizzicato, rhythm, controlled by legs

### Landmarks used (COCO 17)
- Arms/melody: 5, 6, 7, 8, 9, 10 (shoulders, elbows, wrists)
- Legs/rhythm: 13, 14, 15, 16 (knees, ankles)
- Harmony: 3, 4 (ears, head tilt) + 5, 6, 11, 12 (torso tilt)
- Discarded: 0 (nose), 1, 2 (eyes) — facial expressions have no sonic weight

### Descriptor-to-MIDI mapping

**Melody (ch1):**
- Mean wrist height (lm 9, 10) → note within active chord (C3–C6)
- Mean velocity of full arm (lm 5–10) → note trigger + attack velocity
- Horizontal wrist separation (lm 9, 10) → brightness CC74 (0–127)

**Bass (ch2):**
- Ankle velocity (lm 15, 16) → bass note trigger
- Knee height (lm 13, 14) → note duration (50ms–500ms)

**Harmony (no channel):**
- Lateral torso tilt (shoulder axis vs. hip axis) → advance/retreat in chord progression
- Lateral head tilt (ears) → add tension (right) or simplify (left) active chord

### Chord progression (D minor, do not change without confirmation)
```
1. Dm9
2. G13sus4
3. Cmaj7#11
4. Fmaj9
5. Bø7
6. E7alt
```
Notes are always consonant with the active chord. No passing notes. No dissonance.

### Silence threshold
Body velocity below threshold for > 500ms → both channels silent.
Threshold is a configurable parameter (not hardcoded).

---

## Project structure (to be built)

```
cuerposonoro-jetson/
  CLAUDE.md               ← this file
  DESIGN.md               ← full design rationale
  README.md               ← setup and deployment instructions
  requirements.txt
  config.yaml             ← all tunable parameters (thresholds, chord prog, etc.)

  main.py                 ← entry point

  vision/
    detector.py           ← YOLOv8-Pose wrapper
    landmarks.py          ← landmark extraction and normalisation

  features/
    arms.py               ← melody descriptors
    legs.py               ← rhythm descriptors
    harmony.py            ← torso/head tilt → chord progression

  audio/
    fluidsynth.py         ← Fluidsynth process management
    midi.py               ← MIDI note/CC output via ALSA virtual port
    chords.py             ← chord voicings and note selection

  deployment/
    cuerposonoro.service  ← systemd unit file
    setup_jetson.sh       ← one-time setup script for the Jetson

  tests/
    unit/
    integration/
```

---

## Development workflow

- **Develop on Mac.** Run and test locally. Deploy to Jetson via SSH.
- **TDD always.** Write the test first, then the implementation. No exceptions.
  The sequence is: write a failing test → implement the minimum to make it pass → refactor.
  Never write production code without a test that justifies it.
- **Tests must pass before any commit.**
- **Conventional commits.** Format: `type(scope): description`
  Examples: `feat(vision): add YOLOv8 landmark extractor`, `test(audio): add chord voicing tests`
- **All code and comments in English.** No exceptions. This includes: variable names,
  function names, class names, inline comments, docstrings, commit messages, and any
  text that ends up in a source file.
- **config.yaml for all tunable values.** No magic numbers in code.

## Language

Everything in this repository is in English. This is a hard rule with no exceptions:
- Source code (variable names, function names, class names)
- Inline comments
- Docstrings
- Commit messages
- Any text inside source files

The only exception is `DESIGN.md`, which exists in both Spanish and English.

## TDD rules

This project is built with strict Test-Driven Development:

1. **Red:** write a failing test that describes the behaviour you want
2. **Green:** write the minimum production code to make it pass
3. **Refactor:** clean up without breaking the test

Rules:
- No production code without a test that requires it
- Tests live in `tests/unit/` (no hardware) or `tests/integration/` (mocked I/O)
- Unit tests must run in under 1 second with no hardware dependencies
- A test file must exist before its corresponding implementation file
- If Claude Code proposes implementation before tests, that is a workflow violation

## README.md

`README.md` is a living document. It must be kept up to date as the project evolves.
It must contain:
- What the project is and how it relates to `cuerposonoro` (the thesis)
- Hardware requirements and wiring
- One-time Jetson setup instructions
- How to run locally on Mac for development
- How to deploy to the Jetson
- How to run the tests
- A description of every configurable parameter in `config.yaml`
- Credits and licences

Every time a new feature is added, a new parameter is introduced, or the deployment
process changes, `README.md` must be updated in the same commit. A feature is not
done until the README reflects it.

---

## Autonomy and decision rules

Claude Code has autonomy over:
- Internal implementation details (variable names, helper functions, module structure)
- Test design and coverage
- Code style and organisation within the established structure

Claude Code must stop and present options before:
- Adding a new dependency to `requirements.txt`
- Changing the project structure above
- Modifying anything in `deployment/`
- Changing any sonic design parameter (chord progression, landmark mapping, thresholds)
- Any decision that affects how the installation sounds or behaves at runtime

When presenting options, always include: what each option does, what it trades off,
and a recommendation with justification. Then wait for confirmation.

---

## Constraints

- **Python 3.10** — match the Jetson environment exactly
- **No GUI anywhere** — everything must run headless
- **Latency target: under 80ms end-to-end** (camera → sound)
- **Stable for hours** — no memory leaks, no crashes, no resource exhaustion
- **Single soundfont file** — JJazzLab-SoundFont.sf2, loaded at startup
- Do not use SuperCollider. Do not use Surge XT. Do not use OSC.

---

## Key files to read first in any session

1. `DESIGN.md` — full artistic and technical rationale
2. `config.yaml` — current parameter values
3. Any file you are about to modify
# cuerposonoro-jetson: Design Decisions Report

*Artistic and technical design session, March 2026*

---

## Context

This document records the design decisions made for `cuerposonoro-jetson`, an adaptation of *Cuerpo Sonoro* for deployment as an autonomous art installation on an NVIDIA Jetson Orin Nano during URJC's Semana de la Cultura (April 2026).

`cuerposonoro-jetson` is a separate project from the main thesis (`cuerposonoro`). The separation is a deliberate decision: the thesis documents the complete system with MediaPipe Full, 33 landmarks, and MPE synthesis. The installation is an adaptation for a specific physical context, with conscious technical trade-offs and its own sonic character.

---

## 1. Hardware and Architecture Decisions

### 1.1 Deployment hardware: Jetson Orin Nano

The system runs on an NVIDIA Jetson Orin Nano (JetPack 6.1, CUDA 12.6, TensorRT 10.3). The Jetson boots autonomously on power, with no screen and no manual intervention. Switching on the power strip it is connected to is the only required operation.

### 1.2 Vision model: YOLOv8-Pose with TensorRT

**Decision:** replace MediaPipe Full (33 landmarks, CPU) with YOLOv8-Pose (17 COCO landmarks, GPU TensorRT).

**Technical justification:** MediaPipe cannot run on the Jetson GPU via `pip install`. MediaPipe's TFLite models use the `DENSIFY` operator (float16), which no standard conversion tool supports, and the GPU delegate is not compiled into the aarch64 wheel. With `jetson_clocks` enabled, MediaPipe CPU averages 55.9ms (viable, but CPU-only with no headroom for improvement).

YOLOv8-Pose exports to ONNX natively and converts to TensorRT in a single line. Expected latency below 20ms on the Ampere GPU.

**Accepted trade-off:** landmark count drops from 33 (BlazePose) to 17 (COCO). Spine, finger, and detailed facial landmarks are lost. This is consciously accepted in exchange for real GPU acceleration.

**Artistic justification:** YOLOv8-Pose is multi-person by design. The monolith, conceived for a public walkthrough space, gains a collective dimension: multiple people can interact with the installation simultaneously. This possibility was not in the original design and enriches the artistic proposal.

### 1.3 Audio engine: headless Fluidsynth

**Decision:** use Fluidsynth in headless mode instead of Surge XT.

**Technical justification:** Surge XT requires manual intervention to connect the MIDI input (GUI). On a screenless Jetson this is not viable. Fluidsynth runs as a systemd service, accepts MIDI via an ALSA virtual port, and requires no interaction after boot.

**Accepted trade-off:** MPE synthesis is lost (per-note continuous control: independent pitch bend, per-note pressure). This trade-off is accepted because the target sonic character — acoustic instruments transformed by movement — is better achieved with real-sample soundfonts than with MPE synthesis over synthetic timbres.

### 1.4 Soundfont: JJazzLab SoundFont SF2

**Decision:** use the JJazzLab SoundFont SF2 (based on SGM-v2.01 by John Nebauer).

**Justification:** contains vibraphone (preset 000-011) and acoustic bass (preset 000-032) in a single file. Optimised for jazz, completely free, and imposes no restrictions on use in art installations. Requires credit to John Nebauer and the JJazzLab project in documentation.

**Download:** `https://archive.org/download/jjazz-lab-sound-font/JJazzLab-SoundFont.sf2`

---

## 2. Sonic Design Decisions

### 2.1 Core philosophy

**No movement, no sound.** The body is the sole sound source. When no one is in front of the camera, or when the visitor becomes completely still, the installation falls into absolute silence. There is no waiting sound, no ambient backdrop. Silence is part of the work.

### 2.2 Sonic layer architecture

The system produces two independent sonic layers generated simultaneously by one body (or several, in multi-person mode):

| Layer | Instrument | Body part | MIDI channel |
|---|---|---|---|
| Melody | Vibraphone | Full arms | 1 |
| Rhythmic bass | Acoustic bass pizzicato | Legs | 2 |

**Artistic justification:** the "legs as rhythm, arms as melody" configuration is the most intuitive for an audience without musical training. Locomotion generates the pulse; gesture generates the melody. The visitor needs no instructions to discover the relationship.

### 2.3 Aesthetic references

The harmonic and timbral design is oriented towards **Jacob Collier** (extended chords, reharmonisation, harmonic complexity with warmth) and **Latin Jazz** (swing, flavour, good feeling). The goal is to sound like acoustic instruments being transformed by movement, not like a synthesiser.

---

## 3. Landmarks Used

Of the 17 COCO landmarks provided by YOLOv8-Pose:

```
0  nose
1  left eye          ← discarded (facial expression)
2  right eye         ← discarded (facial expression)
3  left ear          ← used (head tilt)
4  right ear         ← used (head tilt)
5  left shoulder     ← used (arm, melody)
6  right shoulder    ← used (arm, melody)
7  left elbow        ← used (arm, melody)
8  right elbow       ← used (arm, melody)
9  left wrist        ← used (arm, melody + pitch)
10 right wrist       ← used (arm, melody + pitch)
11 left hip          ← used (torso tilt)
12 right hip         ← used (torso tilt)
13 left knee         ← used (legs, rhythm)
14 right knee        ← used (legs, rhythm)
15 left ankle        ← used (legs, rhythm)
16 right ankle       ← used (legs, rhythm)
```

**Nose (0) discarded** as redundant with the ears for head position.
**Eyes (1, 2) discarded** by artistic decision: facial expressions carry no sonic weight in this installation. The limbs are the primary gesture.

---

## 4. Descriptor-to-MIDI Mapping

### Layer 1: Melody (vibraphone, MIDI channel 1)

| Descriptor | Landmarks | Controls | Range |
|---|---|---|---|
| Mean wrist height | 9, 10 | Note within active chord | C3 → C6 |
| Mean velocity of full arm | 5, 6, 7, 8, 9, 10 | Note trigger + attack velocity | configurable threshold → vel 120 |
| Horizontal wrist separation | 9, 10 | Brightness/timbre (CC74) | 0 → 127 |

**Always consonant notes:** the melody can only play notes belonging to the active chord. There are no passing notes or dissonance. This ensures that any movement sounds musically correct, regardless of the visitor's musical background.

**Full-arm trigger:** movement of any part of the arm (shoulder, elbow, wrist) can trigger a note. Moving only the wrist is not required. This makes the interaction more natural and intuitive.

**Slow vs. fast movement:** arm velocity directly affects MIDI attack velocity. Slow movement produces soft, less bright notes. Fast movement produces notes with more attack and more brightness. The visitor perceives the change clearly.

### Layer 2: Rhythmic bass (acoustic bass, MIDI channel 2)

| Descriptor | Landmarks | Controls | Range |
|---|---|---|---|
| Ankle velocity | 15, 16 | Bass note trigger (attack) | configurable threshold |
| Knee height | 13, 14 | Note duration | 50ms → 500ms |

The bass note is always the **root of the active chord**. The bass follows the harmony; it has no melody of its own. This creates the feeling of an accompanying double bass, not a solo instrument.

Walking produces natural rhythmic attacks. Raising the knees lengthens the notes. Standing still from the waist down produces silence in the bass, even if the arms keep sounding.

### Layer 3: Harmony (no dedicated MIDI channel, controls the progression)

| Descriptor | Landmarks | Controls |
|---|---|---|
| Lateral torso tilt | shoulder axis (5,6) vs. hip axis (11,12) | Advances/retreats through the 6-chord progression |
| Lateral head tilt | ears (3, 4) | Modifies the active chord: tension (right) or simplification (left) |

The two gestures have clearly distinct and audible effects, allowing the visitor to discover them independently.

---

## 5. Harmonic Progression

**Key:** D minor.
**Style:** expanded ii-V-I with jazz/contemporary extensions. Jacob Collier and Latin Jazz influence.

| Position | Chord | Colour |
|---|---|---|
| 1 | Dm9 | Resting point, warm |
| 2 | G13sus4 | Suspended dominant, floating tension |
| 3 | Cmaj7#11 | Lydian, bright, Collier flavour |
| 4 | Fmaj9 | Subdominant with 9th, expansive |
| 5 | Bø7 | Half-diminished, dark Latin colour |
| 6 | E7alt | Altered dominant, maximum tension back to Dm |

The cycle is continuous: E7alt resolves back to Dm9 with high tension, making the progression fluid and open-ended in time.

Head tilt to the right adds a tension to the active chord (e.g. Dm9 → Dm11). To the left it simplifies or resolves it. This allows the visitor to explore harmonic variations within each chord without leaving the progression.

---

## 6. Silence Thresholds

If overall body velocity drops below a threshold for more than **500ms**, both channels fall silent. The threshold is a configurable parameter, not a fixed value, so it can be adjusted during installation setup based on the space and camera distance.

---

## 7. cuerposonoro vs. cuerposonoro-jetson

| | cuerposonoro (thesis) | cuerposonoro-jetson (installation) |
|---|---|---|
| Hardware | Mac Apple Silicon | Jetson Orin Nano |
| Vision model | MediaPipe Full | YOLOv8-Pose TensorRT |
| Landmarks | 33 (BlazePose) | 17 (COCO) |
| People | 1 | Multiple |
| Audio engine | Surge XT (MPE) | Fluidsynth headless |
| Synthesis | MPE, continuous per-note expressivity | SF2 soundfonts |
| Boot | Manual | Autonomous (systemd) |
| Screen | Yes (development) | No |

The separation into two repositories is a deliberate decision. Keeping them together would mean the thesis repo carries Jetson code, hardware detection logic, and YOLOv8 dependencies that do not belong to the core academic work. If the installation breaks the day before the exhibition, the thesis repo is unaffected.

What they share: the design of the kinematic descriptor-to-MIDI mapping, which is the artistic core of the system.

---

## 8. Credits and Licences

- **JJazzLab SoundFont SF2:** based on SGM-v2.01-NicePianosGuitarsBass by John Nebauer. Free to use, attribution required.
- **YOLOv8-Pose:** Ultralytics, AGPL-3.0 licence.
- **Fluidsynth:** LGPL-2.1 licence.

---

*Document generated in design session with Claude, March 2026.*
# Problem Statement (verbatim reference)

**Title:** Development of an AI-Based Virtual Camera Tracking System for Coarse Alignment of
Mobile Free Space Optical Communication (FSOC) Terminals

**Organisation:** Department of Space / Indian Space Research Organisation (ISRO)
**Category:** Software
**Theme:** Smart Automation, Space Technology
**Dataset provided:** None

---

## Background

Free Space Optical Communication (FSOC) offers advantages for next-generation mobile networks,
including gigabit-to-terabit data rates, license-free spectrum operation, and high immunity to
electromagnetic interference. However, deploying FSOC links between mobile platforms (satellites,
UAVs) presents a severe challenge of pointing, acquisition and tracking (PAT) of highly narrow
laser beams.

PAT typically happens in two stages: **coarse alignment** and **fine alignment**. Coarse alignment
is one of the key challenges of PAT, where the transmitting terminal must first locate and
maintain the remote terminal within its camera Field-of-View (FOV).

Developing and testing such algorithms on real hardware requires expensive cameras, pan-tilt
mechanisms, and optical components. A software-based virtual camera tracking system provides an
inexpensive and accessible platform for algorithm development and learning.

## Description

Unlike conventional radio-frequency systems, FSOC relies on a highly directional optical beam.
Even a small angular error can prevent successful communication. Before the fine pointing
mechanism can take over, a coarse alignment stage must:

- Observe the surrounding environment
- Acquire and detect the remote terminal or beacon
- Estimate the position
- Continuously adjust the pointing direction to maintain visibility

## Functional Objective

Develop a software system that autonomously detects, identifies, and continuously tracks a
designated moving target within a virtual scene by controlling a virtual camera viewport.

---

## Parameters and Specifications

### Camera Parameters

| # | Parameter | Suggested Value | Remarks |
|---|---|---|---|
| 1 | Screen Size (min.) | 2000 × 2000 pixels | Optional: user-defined |
| 2 | Camera Type | Monochrome, Focal Plane Array | Optional: colour |
| 3 | Camera Resolution | 640 × 480 pixels | Optional: user-defined |
| 4 | Camera FOV | User-defined | Default: 4° × 3° |
| 5 | Camera Update Rate | 30 Hz (min.) | |
| 6 | Initial Camera Position | Centre of the screen | |

### Target Parameters

| # | Parameter | Suggested Value | Remarks |
|---|---|---|---|
| 7 | Target Type | Beacon Spot | |
| 8 | Number of Targets | 1 mandatory, multiple optional | |
| 9 | Target Shape | User-defined | Default: Square |
| 10 | Target Size | 5–20 × 5–20 pixels (user-defined) | Default: 10 × 10 |
| 11 | Initial Target Location | User-defined | Default: Random |
| 12 | Motion | Selectable, at least four: Straight Line, Circular, Figure of 8, Random | Optional: Spiral, Sinusoidal, User-defined |

### Camera Motion Constraints

| # | Parameter | Suggested Value | Remarks |
|---|---|---|---|
| 13 | Max. Pan Speed | 5–10 °/s (user-defined) | Default: 5 °/s |
| 14 | Max. Tilt Speed | 5–10 °/s (user-defined) | Default: 5 °/s |
| 15 | Update Interval | ≥ 20 Hz | |

### Performance Specifications

| # | Parameter | Requirement |
|---|---|---|
| 16 | Acquisition Time | ≤ 2 sec |
| 17 | Tracking Error | ≤ 10 pixels |
| 18 | Target Loss | < 5 % |
| 19 | Re-acquisition Time | ≤ 1 sec |
| 20 | Processing Speed | ≥ 20 FPS |

### Disturbances and Noise

| # | Parameter | Suggested Value | Remarks |
|---|---|---|---|
| 21 | Image Noise | Salt & Pepper (~10% of image), Gaussian, Poisson | User selectable (one or more) |
| 22 | Max. Std. Deviation of Noise | 20 pixels | User-defined |
| 23 | Max. Camera Jitter | ± 20 pixels / frame | User-defined |
| 24 | Atmospheric Disturbance | Clear, Haze, Fog, Rain, Low light | User-defined reduction in contrast and brightness |
| 25 | Platform Motion | ± 20 pixels/frame (max.) | User selectable. Default/Mandatory: Linear. Optional: Circular, random, spiral, figure of 8 |

---

## Expected Solution

Participants shall develop an AI-assisted camera tracking system capable of automatically
detecting and continuously tracking a moving optical beacon in a simulated video stream while
controlling a virtual pan-tilt camera.

The developed software shall be able to:

- Generate a configurable virtual environment
- Generate one or more moving targets
- Implement a movable virtual camera
- Detect the target beacon automatically
- Track the beacon continuously using computer vision
- Control and reposition the virtual camera
- Generate and introduce disturbances due to atmospheric turbulence, platform vibrations, camera
  motion, noise, etc., in the virtual camera feed
- Display tracking performance and statistics in real time

---

## Mandatory Deliverables

1. **Software Application** — a standalone executable application implementing the complete
   virtual camera tracking system, providing all mandatory functions and features.
2. **Source Code** — complete source code with proper documentation. The code shall be modular
   and adequately commented.
3. **Technical Report** — about 10–15 pages containing problem understanding, system
   architecture, description of software modules, tracking methods, AI methods (if used), test
   methodology, performance analysis and future improvements.
4. **User Manual** — description of installation, application operation, parameter configuration,
   GUI description. An optional 3–5 minute demonstration video may also be provided.
5. **Performance Log** — the software must be capable of automatically generating a performance
   report containing simulation duration, FPS, acquisition time, average and maximum tracking
   error, lock retention rate, processing time, etc.

---

## Evaluation Method and Criteria

| Stage | Description | Criteria | Marks |
|---|---|---|---|
| **Functional Verification** | Teams are given 10–15 minutes to demonstrate the software and its functionality. | 1. Implementation of all mandatory functions 2. Operational success 3. GUI | **20%** |
| **Benchmark Performance-1** | Each team will be given a few scenarios. | 1. Execution of the scenario 2. Log of centroiding error 3. Automatically generated performance logs | **30%** |
| **Benchmark Performance-2** | Each team will be given a few video files (`.mp4`) @30 fps, covering a complete screen with noise and a moving beacon spot. The software needs to **bypass its PTZ camera** and take this video as an input to the coarse pointing system. | 1. Comparison of centroiding error with predefined error values 2. Performance w.r.t. RMSE, acquisition and re-acquisition time, lock retention rate, FPS, etc. | **30%** |
| **Technical Evaluation** | Teams present their approach, methods, architecture and design to the evaluators. | 1. Understanding of the problem 2. System architecture and software design 3. Selection of algorithms 4. AI and computer vision 5. Innovation and novelty 6. Technical documentation and presentation 7. Technical discussion and Q&A | **20%** |

---

## Known ambiguities in the spec (our resolutions)

The spec contains genuine ambiguities. We resolve them as follows and document the resolution in
the technical report and in the generated logs.

1. **Three separate rates are listed** (camera update 30 Hz, control update ≥20 Hz, processing
   ≥20 FPS). We treat them as three distinct clocks and log all three.
2. **"Tracking Error ≤ 10 pixels" is unit-ambiguous** — it could mean centroid estimate vs.
   ground truth (measurement error) or target vs. boresight (pointing error). We compute and log
   **both**, labelled. Benchmark-2 says "centroiding error", so measurement error is the scored
   quantity.
3. **The acquisition clock start/stop is undefined.** We define it explicitly (see `CLAUDE.md`
   → Metric definitions) and state the definition in every generated log header.
4. **Maximum trackable target velocity is not specified**, but is physically bounded by the
   pan/tilt rate limit. We quantify the trackable envelope rather than only testing targets
   slow enough to succeed.
5. **The "AI-based" requirement does not specify what must be learned.** We use a hybrid:
   classical CV primary path, Kalman state estimation, lightweight CNN fallback.

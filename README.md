# FSOC Coarse Alignment — Virtual Camera Tracking System

AI-assisted virtual camera tracking system for coarse alignment of mobile Free Space Optical
Communication (FSOC) terminals. Smart India Hackathon, Department of Space / ISRO,
Problem Statement 4.

## What it does

Simulates the coarse alignment stage of a laser-communication Pointing, Acquisition and Tracking
(PAT) system entirely in software:

1. Renders a 2000x2000 virtual scene containing a moving optical beacon
2. Extracts a narrow (640x480, 4 deg x 3 deg) camera viewport from it
3. Degrades that viewport with sensor noise, atmospheric effects, jitter and platform motion
4. Automatically detects the beacon and estimates its centroid to sub-pixel accuracy
5. Drives a virtual pan/tilt camera to keep the beacon centred
6. Logs tracking performance in real time and generates a performance report

It also runs in a second mode that bypasses the virtual camera entirely and ingests pre-recorded
`.mp4` video, for benchmark evaluation.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# headless simulation run
python -m src.main --config config/default.json --headless

# GUI
python -m src.main --config config/default.json

# benchmark video mode (Mode B)
python -m src.main --config config/default.json --video path/to/evaluator.mp4
```

## Documentation

| File | Contents |
|---|---|
| `CLAUDE.md` | Project context, constraints, conventions. **Read first.** |
| `docs/PROBLEM_STATEMENT.md` | Verbatim official spec + our resolution of its ambiguities |
| `docs/DESIGN.md` | Architecture, algorithms, mathematical models |
| `docs/ROADMAP.md` | Phased build order and current status |
| `config/default.json` | Every tunable parameter in the system |

## Performance targets

| Metric | Requirement |
|---|---|
| Tracking / centroiding error | <= 10 px |
| Acquisition time | <= 2 s |
| Re-acquisition time | <= 1 s |
| Target loss rate | < 5 % |
| Processing throughput | >= 20 FPS |
| Camera update rate | >= 30 Hz |

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

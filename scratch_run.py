import sys
import os
import subprocess
from pathlib import Path
import json

from scenarios.generate import VideoSpec, generate_video, DEFAULT_SPECS

specs = {s.name: s for s in DEFAULT_SPECS}

# Create sz15 and sz20
specs["sz15"] = VideoSpec("sz15", width=1920, height=1080, frames=120, peak=240, background=60, gaussian_sigma=6, bitrate_kbps=4000, motion="figure8", size_px=15.0, sigma_px=15.0/2.355)
specs["sz20"] = VideoSpec("sz20", width=1920, height=1080, frames=120, peak=240, background=60, gaussian_sigma=6, bitrate_kbps=4000, motion="figure8", size_px=20.0, sigma_px=20.0/2.355)

to_run = ["sz15", "sz20", "baseline_640", "square_bright", "fhd_1920_bigspot"]

print("Generating and running...")
for name in to_run:
    spec = specs[name]
    video_path, sidecar_path = generate_video(spec, Path("sweep"))
    
    scenario = Path("sweep") / f"{name}.json"
    scenario.write_text(json.dumps({
        "run": {"mode": "video"},
        "video_input": {"path": str(video_path), "ground_truth_path": str(sidecar_path)},
    }))
    
    result = subprocess.run([sys.executable, "-m", "src.main", "--config", "config/default.json", "--scenario", str(scenario), "--headless"], capture_output=True, env={"PYTHONPATH": ".", **os.environ}, text=True)
    if result.returncode != 0:
        print(f"{name} FAILED TO RUN: {result.stderr}")
    
    # Check failure rate
    log_file = Path("logs/frames.csv")
    if not log_file.exists():
        print(f"{name}: NO LOG GENERATED")
        continue
    
    lines = [l for l in log_file.read_text().splitlines() if not l.startswith("#")]
    if not lines:
        print(f"{name}: NO DATA IN LOG")
        continue
        
    headers = lines[0].split(",")
    reason_idx = headers.index("track_reason")
    
    gated_out = 0
    for l in lines[1:]:
        cols = l.split(",")
        if "gated_out" in cols[reason_idx]:
            gated_out += 1
            
    fail_rate = gated_out / len(lines[1:]) * 100
    print(f"{name} failure rate: {fail_rate:.1f}%")

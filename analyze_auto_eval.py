"""
SmartVisionX Offline CS Evaluation Analyzer
-------------------------------------------
Use this AFTER any live/mobile demo session.
It converts the automatically recorded CSV into paper-ready evaluation tables and figures.

M5 additions:
  • Reads the new CSV columns `location_danger_level` /
    `location_danger_message` and reports them in the summary.
  • Generates an extra chart (08) showing the danger-level distribution
    so you can see how often the location navigator warned the user
    about obstacles / STOPs during the session.

Run examples:
    python analyze_auto_eval.py
    python analyze_auto_eval.py Logs/mobile_session_20260611_170000.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def latest_csv(log_dir: Path) -> Path:
    files = sorted(log_dir.glob("mobile_session_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        files = sorted(log_dir.glob("session_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No session CSV found in {log_dir}")
    return files[0]


def num_col(df, col):
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def save_bar(series, title, xlabel, ylabel, out_path, top=15):
    s = series.dropna()
    s = s[s.astype(str).str.len() > 0]
    if s.empty:
        return
    counts = s.value_counts().head(top).sort_values()
    plt.figure(figsize=(9, 5))
    counts.plot(kind="barh")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_line(y, title, ylabel, out_path):
    y = pd.to_numeric(y, errors="coerce").dropna()
    if y.empty:
        return
    plt.figure(figsize=(10, 4.8))
    plt.plot(range(len(y)), y)
    plt.title(title)
    plt.xlabel("Frame index")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default=None,
                        help="Path to session CSV. If omitted, latest Logs/mobile_session_*.csv is used.")
    parser.add_argument("--out", default=None, help="Output directory. Default: Reports/<csv_stem>")
    args = parser.parse_args()

    project_dir = Path.cwd()
    csv_path = Path(args.csv) if args.csv else latest_csv(project_dir / "Logs")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = Path(args.out) if args.out else project_dir / "Reports" / csv_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    fps = num_col(df, "fps")
    if fps.empty:
        fps = num_col(df, "fps_estimate")
    latency = num_col(df, "latency_ms")
    if latency.empty:
        latency = num_col(df, "total_latency_ms")

    alerts_col = "alert" if "alert" in df.columns else "message_spoken"
    alerts = df[alerts_col].fillna("").astype(str) if alerts_col in df.columns else pd.Series([""] * len(df))
    alert_rows = df[alerts.str.strip() != ""].copy()

    # M5: danger summary
    loc_danger = df["location_danger_level"].fillna("").astype(str) if "location_danger_level" in df.columns else pd.Series([""] * len(df))
    non_idle = df[df["location_state"].fillna("").astype(str).str.lower().ne("idle")] if "location_state" in df.columns else df
    danger_summary = {}
    if not loc_danger.empty:
        for lvl in ("stop", "warning", "caution", "none"):
            danger_summary[lvl] = int((loc_danger == lvl).sum())

    metrics = {
        "session_csv": str(csv_path),
        "total_frames_logged": int(len(df)),
        "total_spoken_alerts": int(len(alert_rows)),
        "avg_fps": round(float(fps.mean()), 2) if not fps.empty else "",
        "median_fps": round(float(fps.median()), 2) if not fps.empty else "",
        "avg_latency_ms": round(float(latency.mean()), 2) if not latency.empty else "",
        "median_latency_ms": round(float(latency.median()), 2) if not latency.empty else "",
        "avg_detection_ms": round(float(num_col(df, "det_ms").mean()), 2) if "det_ms" in df.columns else "",
        "avg_scene_ms": round(float(num_col(df, "scene_ms").mean()), 2) if "scene_ms" in df.columns else "",
        "avg_ocr_ms": round(float(num_col(df, "ocr_ms").mean()), 2) if "ocr_ms" in df.columns else "",
        "avg_navigation_ms": round(float(num_col(df, "nav_ms").mean()), 2) if "nav_ms" in df.columns else "",
        "saved_evidence_frames": int(df.get("evidence_frame", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("").sum()) if "evidence_frame" in df.columns else 0,
        "most_common_scene": df["scene"].mode().iloc[0] if "scene" in df.columns and not df["scene"].dropna().empty else "",
        "most_common_object": df["top_object"].mode().iloc[0] if "top_object" in df.columns and not df["top_object"].dropna().empty else "",
        "most_common_navigation": df["nav_command"].mode().iloc[0] if "nav_command" in df.columns and not df["nav_command"].dropna().empty else "",
        # M5
        "danger_stop_count":    danger_summary.get("stop", 0),
        "danger_warning_count": danger_summary.get("warning", 0),
        "danger_caution_count": danger_summary.get("caution", 0),
    }

    pd.DataFrame([metrics]).to_csv(out_dir / "summary_metrics.csv", index=False)
    alert_rows.to_csv(out_dir / "spoken_alert_log.csv", index=False)

    if "scene" in df.columns:
        df["scene"].value_counts().to_csv(out_dir / "scene_distribution.csv")
    if "top_object" in df.columns:
        df["top_object"].value_counts().to_csv(out_dir / "object_distribution.csv")
    if "nav_command" in df.columns:
        df["nav_command"].value_counts().to_csv(out_dir / "navigation_distribution.csv")
    if "location_danger_level" in df.columns:
        df["location_danger_level"].value_counts().to_csv(out_dir / "location_danger_distribution.csv")

    save_line(fps, "FPS over Time", "FPS", out_dir / "01_fps_over_time.png")
    save_line(latency, "End-to-End Latency over Time", "Latency (ms)", out_dir / "02_latency_over_time.png")

    module_cols = [c for c in ["det_ms", "scene_ms", "ocr_ms", "nav_ms"] if c in df.columns]
    if module_cols:
        vals = [pd.to_numeric(df[c], errors="coerce").mean() for c in module_cols]
        plt.figure(figsize=(7, 4.5))
        plt.bar([c.replace("_ms", "") for c in module_cols], vals)
        plt.title("Average Module Latency")
        plt.ylabel("Milliseconds")
        plt.tight_layout()
        plt.savefig(out_dir / "03_average_module_latency.png", dpi=220)
        plt.close()

    if "scene" in df.columns:
        save_bar(df["scene"], "Scene Distribution", "Frames", "Scene", out_dir / "04_scene_distribution.png")
    if "top_object" in df.columns:
        save_bar(df["top_object"], "Detected Object Distribution", "Frames", "Object", out_dir / "05_object_distribution.png")
    if "nav_command" in df.columns:
        save_bar(df["nav_command"], "Navigation Command Distribution", "Frames", "Command", out_dir / "06_navigation_distribution.png")
    if "priority" in df.columns:
        save_bar(df["priority"].astype(str), "Alert Priority Distribution", "Frames", "Priority", out_dir / "07_priority_distribution.png")

    # M5: danger-level distribution chart
    if "location_danger_level" in df.columns and not loc_danger.empty:
        save_bar(loc_danger, "Location Danger-Level Distribution",
                 "Frames", "Danger Level", out_dir / "08_location_danger_distribution.png")

    summary_text = f"""SmartVisionX runtime evaluation summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
CSV: {csv_path}

The live session logged {metrics['total_frames_logged']} frames and produced {metrics['total_spoken_alerts']} spoken alerts.
Average FPS was {metrics['avg_fps']} and average end-to-end latency was {metrics['avg_latency_ms']} ms.
The most frequent scene was {metrics['most_common_scene']}, the most frequent detected object was {metrics['most_common_object']}, and the most common navigation command was {metrics['most_common_navigation']}.
Saved evidence frames: {metrics['saved_evidence_frames']}.

M5 danger breakdown (while a location navigation was active):
  stop:    {metrics['danger_stop_count']} frames
  warning: {metrics['danger_warning_count']} frames
  caution: {metrics['danger_caution_count']} frames
"""
    (out_dir / "report_summary.txt").write_text(summary_text, encoding="utf-8")

    print("Analysis complete.")
    print("CSV:", csv_path)
    print("Output:", out_dir)


if __name__ == "__main__":
    main()
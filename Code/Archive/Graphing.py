#!/usr/bin/env python3
"""
Graph one IMU sensor recording from the Brunner dataset.

Example:
    python Code/Graphing.py --data-dir "Brunner Data/swimming-recognition-lap-counting-master/data/processed_30hz_relabeled" \
        --output-dir "Code/freestyle_graphs"
"""

from pathlib import Path
import argparse

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "Brunner Data" / "swimming-recognition-lap-counting-master" / "data" / "processed_30hz_relabeled"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Code" / "freestyle_graphs"
DEFAULT_FILE_NAME = "Freestyle_1526810816300.csv"


ACC_COLUMNS = ["ACC_0", "ACC_1", "ACC_2"]
GYRO_COLUMNS = ["GYRO_0", "GYRO_1", "GYRO_2"]
MAG_COLUMNS = ["MAG_0", "MAG_1", "MAG_2"]

def find_recording(data_dir: Path, file_name: str) -> Path:
    """Find the requested recording in the participant subdirectories."""
    matches = sorted(data_dir.rglob(file_name))
    if not matches:
        raise FileNotFoundError(f"Recording not found: {file_name} in {data_dir}")
    return matches[0]


def plot_freestyle_file(csv_path: Path, output_dir: Path):
    """Create an interactive HTML plot for one freestyle CSV file."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[col for col in df.columns if col.startswith("Unnamed")], errors="ignore")

    sample_count = len(df)
    if sample_count == 0:
        return

    timestamps = pd.to_numeric(df["timestamp"], errors="coerce")
    time_axis = (timestamps - timestamps.iloc[0]) / 1_000_000_000

    output_dir.mkdir(parents=True, exist_ok=True)
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Accelerometer", "Gyroscope", "Magnetometer"),
    )

    signal_groups = [
        ("Accelerometer", ACC_COLUMNS, "ACC"),
        ("Gyroscope", GYRO_COLUMNS, "GYRO"),
        ("Magnetometer", MAG_COLUMNS, "MAG"),
    ]

    for row, (sensor_name, group_columns, short_name) in enumerate(signal_groups, start=1):
        for column in group_columns:
            figure.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=pd.to_numeric(df[column], errors="coerce"),
                    mode="lines",
                    name=column,
                    hovertemplate="Time: %{x:.6f} s<br>Value: %{y:.6f}<extra>%{fullData.name}</extra>",
                ),
                row=row,
                col=1,
            )
        figure.update_yaxes(title_text=sensor_name, row=row, col=1, showgrid=True, gridcolor="lightgray")

    figure.update_xaxes(title_text="Time from recording start (seconds)", row=3, col=1, showgrid=True)
    figure.update_layout(
        title=f"Interactive freestyle IMU recording: {csv_path.name}",
        height=900,
        hovermode="x unified",
        hoverlabel=dict(showarrow=True),
        template="plotly_white",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(t=110, r=30, b=60, l=70),
    )
    figure.update_layout(dragmode="zoom")

    output_file = output_dir / f"{csv_path.stem}.html"
    figure.write_html(output_file, include_plotlyjs=True, full_html=True)


def main():
    parser = argparse.ArgumentParser(description="Create an interactive graph for one freestyle IMU recording.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--file", default=DEFAULT_FILE_NAME, help="Recording filename to plot.")
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = find_recording(args.data_dir, args.file)
    plot_freestyle_file(csv_path, args.output_dir)
    print(f"Created interactive graph for {csv_path.name} in {args.output_dir}")


if __name__ == "__main__":
    main()

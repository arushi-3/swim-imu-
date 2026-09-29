#!/usr/bin/env python3
"""
Create one interactive HTML IMU graph for each selected freestyle recording.

Each HTML contains:
    - Accelerometer: ACC_0, ACC_1, ACC_2
    - Gyroscope:     GYRO_0, GYRO_1, GYRO_2
    - Magnetometer:  MAG_0, MAG_1, MAG_2

The goal is to inspect each recording manually and write down the
transition / turn / non-clean-swimming time ranges.

By default, only the first 130 seconds of each recording are displayed.

Example:
    python Code/MultiGraphing.py
"""

from pathlib import Path
import argparse

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "Brunner Data"
    / "swimming-recognition-lap-counting-master"
    / "data"
    / "processed_30hz_relabeled"
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "Code"
    / "manual_transition_graphs"
)


# ============================================================
# DISPLAY SETTINGS
# ============================================================

MAX_DISPLAY_SECONDS = 130.0


# ============================================================
# RECORDINGS TO GRAPH
# ============================================================

RECORDINGS = [
    "0/Freestyle_1527873200322.csv",
    "1/Freestyle_1527676714707.csv",
    "2/Freestyle_1527072298681.csv",
    "4/Freestyle_1526810816300.csv",
    "4/Freestyle_1526810977061.csv",
]


# ============================================================
# SENSOR COLUMNS
# ============================================================

ACC_COLUMNS = ["ACC_0", "ACC_1", "ACC_2"]
GYRO_COLUMNS = ["GYRO_0", "GYRO_1", "GYRO_2"]
MAG_COLUMNS = ["MAG_0", "MAG_1", "MAG_2"]


# ============================================================
# LOAD RECORDING
# ============================================================

def load_recording(csv_path: Path):
    """Load one IMU recording and create an elapsed-time axis."""

    df = pd.read_csv(csv_path)

    df = df.drop(
        columns=[
            col
            for col in df.columns
            if col.startswith("Unnamed")
        ],
        errors="ignore"
    )

    required_columns = [
        "timestamp",
        *ACC_COLUMNS,
        *GYRO_COLUMNS,
        *MAG_COLUMNS,
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required columns: {missing}"
        )

    for col in required_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=["timestamp"]
    ).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            f"No usable samples found in {csv_path}"
        )

    timestamps = df["timestamp"]

    df["time"] = (
        timestamps - timestamps.iloc[0]
    ) / 1_000_000_000

    # Save the full duration before trimming.
    full_duration = float(df["time"].iloc[-1])

    # Only keep the first 130 seconds for manual inspection.
    df = df[
        df["time"] <= MAX_DISPLAY_SECONDS
    ].copy()

    return df, full_duration


# ============================================================
# CREATE ONE HTML GRAPH
# ============================================================

def create_graph(csv_path: Path, output_dir: Path):
    """Create an interactive IMU graph for one recording."""

    df, full_duration = load_recording(csv_path)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Accelerometer",
            "Gyroscope",
            "Magnetometer",
        ),
    )

    signal_groups = [
        ("Accelerometer", ACC_COLUMNS),
        ("Gyroscope", GYRO_COLUMNS),
        ("Magnetometer", MAG_COLUMNS),
    ]

    for row, (sensor_name, columns) in enumerate(
        signal_groups,
        start=1
    ):

        for column in columns:

            figure.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df[column],
                    mode="lines",
                    name=column,
                    hovertemplate=(
                        f"{column}"
                        "<br>Time: %{x:.3f} s"
                        "<br>Value: %{y:.6f}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )

        figure.update_yaxes(
            title_text=sensor_name,
            row=row,
            col=1,
            showgrid=True,
            gridcolor="lightgray",
        )

    figure.update_xaxes(
        title_text="Time from recording start (seconds)",
        row=3,
        col=1,
        showgrid=True,
    )

    displayed_duration = float(df["time"].iloc[-1])

    figure.update_layout(
        title=(
            f"Manual transition inspection: {csv_path.name}"
            f"<br><sup>"
            f"Participant folder: {csv_path.parent.name} | "
            f"Displayed: 0-{displayed_duration:.2f} s | "
            f"Full recording: {full_duration:.2f} s"
            f"</sup>"
        ),
        height=900,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h",
            y=1.03,
            x=0,
        ),
        margin=dict(
            t=120,
            r=30,
            b=60,
            l=70,
        ),
        dragmode="zoom",
    )

    output_file = (
        output_dir
        / f"{csv_path.parent.name}_{csv_path.stem}_manual_transition_graph.html"
    )

    figure.write_html(
        output_file,
        include_plotlyjs=True,
        full_html=True,
    )

    print(f"Created: {output_file}")
    print(f"  Full duration: {full_duration:.2f} s")
    print(f"  Displayed through: {displayed_duration:.2f} s")

    return output_file


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create interactive HTML IMU graphs for several "
            "freestyle recordings so transition times can be "
            "labeled manually."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    if not args.data_dir.exists():

        raise FileNotFoundError(
            f"Data directory not found: {args.data_dir}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 60)
    print("CREATING MANUAL TRANSITION GRAPHS")
    print("=" * 60)
    print()

    print(
        f"Only the first {MAX_DISPLAY_SECONDS:.0f} seconds "
        f"of each recording will be displayed."
    )
    print()

    created_files = []

    for relative_path in RECORDINGS:

        csv_path = (
            args.data_dir
            / relative_path
        )

        if not csv_path.exists():

            print(
                f"WARNING: file not found, skipping:\n"
                f"{csv_path}\n"
            )

            continue

        try:

            output_file = create_graph(
                csv_path,
                args.output_dir
            )

            created_files.append(
                output_file
            )

        except Exception as error:

            print(
                f"ERROR while processing "
                f"{csv_path.name}: {error}"
            )

    print()
    print("=" * 60)
    print("FINISHED")
    print("=" * 60)

    print(
        f"Created {len(created_files)} HTML file(s)."
    )

    print(
        f"Output folder:\n{args.output_dir}"
    )

    print()
    print(
        "Open each HTML file, zoom into the recording, and "
        "write down the start/end times of every transition "
        "within the first 130 seconds."
    )

    print()
    print(
        "Then enter those times in the FILE_CONFIG "
        "transition_ranges section of your multi-file "
        "TransitionClassifier.py."
    )


if __name__ == "__main__":
    main()

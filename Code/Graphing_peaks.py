#!/usr/bin/env python3
"""
Graph one IMU sensor recording from the Brunner dataset
and detect stroke peaks and troughs.

Example:
    python Code/Graphing.py --data-dir "Brunner Data/swimming-recognition-lap-counting-master/data/processed_30hz_relabeled" \
        --output-dir "Code/freestyle_graphs"
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from scipy.signal import find_peaks


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "Brunner Data"
    / "swimming-recognition-lap-counting-master"
    / "data"
    / "processed_30hz_relabeled"
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "Code" / "freestyle_graphs"

DEFAULT_FILE_NAME = "Freestyle_1526810816300.csv"


ACC_COLUMNS = ["ACC_0", "ACC_1", "ACC_2"]
GYRO_COLUMNS = ["GYRO_0", "GYRO_1", "GYRO_2"]
MAG_COLUMNS = ["MAG_0", "MAG_1", "MAG_2"]


# ------------------------------------------------------------
# PEAK DETECTION SETTINGS
# ------------------------------------------------------------

# Change this to GYRO_0, GYRO_1, or GYRO_2
STROKE_AXIS = "GYRO_1"

# Minimum time allowed between detected peaks/troughs
MIN_PEAK_DISTANCE_SECONDS = 0.75

# Controls how much a peak/trough needs to stand out
PROMINENCE_FACTOR = 0.7

# Helps remove smaller "in-between" positive peaks.
# Higher number = stricter.
POSITIVE_PEAK_PERCENTILE = 80

# Helps remove smaller "in-between" negative troughs.
# Lower number = stricter.
NEGATIVE_TROUGH_PERCENTILE = 20


def find_recording(data_dir: Path, file_name: str) -> Path:
    """Find the requested recording in the participant subdirectories."""

    matches = sorted(data_dir.rglob(file_name))

    if not matches:
        raise FileNotFoundError(
            f"Recording not found: {file_name} in {data_dir}"
        )

    return matches[0]


def plot_freestyle_file(csv_path: Path, output_dir: Path):
    """Create an interactive HTML plot for one freestyle CSV file."""

    df = pd.read_csv(csv_path)

    df = df.drop(
        columns=[
            col
            for col in df.columns
            if col.startswith("Unnamed")
        ],
        errors="ignore"
    )

    sample_count = len(df)

    if sample_count == 0:
        return

    # --------------------------------------------------------
    # TIME AXIS
    # --------------------------------------------------------

    timestamps = pd.to_numeric(
        df["timestamp"],
        errors="coerce"
    )

    time_axis = (
        timestamps - timestamps.iloc[0]
    ) / 1_000_000_000

    # --------------------------------------------------------
    # ESTIMATE SAMPLING FREQUENCY
    # --------------------------------------------------------

    time_differences = np.diff(time_axis)

    time_differences = time_differences[
        time_differences > 0
    ]

    fs = 1 / np.median(time_differences)

    print(
        f"Estimated sampling frequency: {fs:.2f} Hz"
    )

    # --------------------------------------------------------
    # SELECT GYROSCOPE SIGNAL
    # --------------------------------------------------------

    gyro_signal = pd.to_numeric(
        df[STROKE_AXIS],
        errors="coerce"
    ).to_numpy()

    # --------------------------------------------------------
    # PEAK DETECTION SETTINGS
    # --------------------------------------------------------

    minimum_peak_distance = int(
        MIN_PEAK_DISTANCE_SECONDS * fs
    )

    prominence = (
        np.std(gyro_signal)
        * PROMINENCE_FACTOR
    )

    # --------------------------------------------------------
    # HEIGHT THRESHOLDS
    #
    # These are used to decide which candidate extrema
    # will actually be used for stroke timing.
    # --------------------------------------------------------

    positive_threshold = np.percentile(
        gyro_signal,
        POSITIVE_PEAK_PERCENTILE
    )

    negative_threshold = np.percentile(
        gyro_signal,
        NEGATIVE_TROUGH_PERCENTILE
    )

    print(
        f"Positive peak threshold: "
        f"{positive_threshold:.3f}"
    )

    print(
        f"Negative trough threshold: "
        f"{negative_threshold:.3f}"
    )

    # --------------------------------------------------------
    # DETECT ALL POSITIVE PEAK CANDIDATES
    # --------------------------------------------------------

    candidate_peaks, peak_properties = find_peaks(
        gyro_signal,
        distance=minimum_peak_distance,
        prominence=prominence
    )

    candidate_peak_times = time_axis.iloc[candidate_peaks]

    # --------------------------------------------------------
    # KEEP ONLY POSITIVE PEAKS THAT WILL BE USED
    # --------------------------------------------------------

    peaks = candidate_peaks[
        gyro_signal[candidate_peaks] >= positive_threshold
    ]

    peak_times = time_axis.iloc[peaks]

    print(
        f"Detected {len(candidate_peaks)} positive peak candidates"
    )

    print(
        f"Using {len(peaks)} positive peaks for stroke timing"
    )

    # --------------------------------------------------------
    # DETECT ALL NEGATIVE TROUGH CANDIDATES
    #
    # Multiplying by -1 flips the signal, so negative
    # troughs become positive peaks for find_peaks().
    # --------------------------------------------------------

    candidate_troughs, trough_properties = find_peaks(
        -gyro_signal,
        distance=minimum_peak_distance,
        prominence=prominence
    )

    candidate_trough_times = time_axis.iloc[candidate_troughs]

    # --------------------------------------------------------
    # KEEP ONLY NEGATIVE TROUGHS THAT WILL BE USED
    # --------------------------------------------------------

    troughs = candidate_troughs[
        gyro_signal[candidate_troughs] <= negative_threshold
    ]

    trough_times = time_axis.iloc[troughs]

    print(
        f"Detected {len(candidate_troughs)} negative trough candidates"
    )

    print(
        f"Using {len(troughs)} negative troughs for stroke timing"
    )

    # --------------------------------------------------------
    # CALCULATE POSITIVE PEAK-TO-PEAK PERIODS
    #
    # IMPORTANT:
    # These calculations use ONLY the filtered/starred peaks.
    # --------------------------------------------------------

    peak_periods = np.diff(
        peak_times.to_numpy()
    )

    if len(peak_periods) > 0:

        print("\nPOSITIVE PEAK-TO-PEAK")

        print(
            f"Mean period: "
            f"{np.mean(peak_periods):.3f} s"
        )

        print(
            f"Period standard deviation: "
            f"{np.std(peak_periods):.3f} s"
        )

        print(
            f"Mean frequency: "
            f"{np.mean(1 / peak_periods):.3f} Hz"
        )

        print(
            f"Mean stroke rate: "
            f"{np.mean(60 / peak_periods):.2f} strokes/min"
        )

    # --------------------------------------------------------
    # CALCULATE NEGATIVE TROUGH-TO-TROUGH PERIODS
    #
    # IMPORTANT:
    # These calculations use ONLY the filtered/starred troughs.
    # --------------------------------------------------------

    trough_periods = np.diff(
        trough_times.to_numpy()
    )

    if len(trough_periods) > 0:

        print("\nNEGATIVE TROUGH-TO-TROUGH")

        print(
            f"Mean period: "
            f"{np.mean(trough_periods):.3f} s"
        )

        print(
            f"Period standard deviation: "
            f"{np.std(trough_periods):.3f} s"
        )

        print(
            f"Mean frequency: "
            f"{np.mean(1 / trough_periods):.3f} Hz"
        )

        print(
            f"Mean stroke rate: "
            f"{np.mean(60 / trough_periods):.2f} strokes/min"
        )

    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

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
            "Magnetometer"
        ),
    )

    signal_groups = [
        (
            "Accelerometer",
            ACC_COLUMNS,
            "ACC"
        ),
        (
            "Gyroscope",
            GYRO_COLUMNS,
            "GYRO"
        ),
        (
            "Magnetometer",
            MAG_COLUMNS,
            "MAG"
        ),
    ]

    # --------------------------------------------------------
    # ORIGINAL SIGNAL PLOTS
    # --------------------------------------------------------

    for row, (
        sensor_name,
        group_columns,
        short_name
    ) in enumerate(
        signal_groups,
        start=1
    ):

        for column in group_columns:

            figure.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=pd.to_numeric(
                        df[column],
                        errors="coerce"
                    ),
                    mode="lines",
                    name=column,
                    hovertemplate=(
                        "Time: %{x:.6f} s"
                        "<br>Value: %{y:.6f}"
                        "<extra>%{fullData.name}</extra>"
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
            gridcolor="lightgray"
        )

    # --------------------------------------------------------
    # ALL POSITIVE PEAK CANDIDATES
    # --------------------------------------------------------

    figure.add_trace(
        go.Scatter(
            x=candidate_peak_times,
            y=gyro_signal[candidate_peaks],
            mode="markers",
            name="Positive peak candidates",
            marker=dict(
                size=7,
                symbol="x"
            ),
            hovertemplate=(
                "Positive peak candidate"
                "<br>Time: %{x:.3f} s"
                "<br>Value: %{y:.3f}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    # --------------------------------------------------------
    # POSITIVE PEAKS ACTUALLY USED
    # --------------------------------------------------------

    figure.add_trace(
        go.Scatter(
            x=peak_times,
            y=gyro_signal[peaks],
            mode="markers",
            name="Positive peaks USED",
            marker=dict(
                size=15,
                symbol="star"
            ),
            hovertemplate=(
                "POSITIVE PEAK USED"
                "<br>Time: %{x:.3f} s"
                "<br>Value: %{y:.3f}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    # --------------------------------------------------------
    # ALL NEGATIVE TROUGH CANDIDATES
    # --------------------------------------------------------

    figure.add_trace(
        go.Scatter(
            x=candidate_trough_times,
            y=gyro_signal[candidate_troughs],
            mode="markers",
            name="Negative trough candidates",
            marker=dict(
                size=7,
                symbol="circle-open"
            ),
            hovertemplate=(
                "Negative trough candidate"
                "<br>Time: %{x:.3f} s"
                "<br>Value: %{y:.3f}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    # --------------------------------------------------------
    # NEGATIVE TROUGHS ACTUALLY USED
    # --------------------------------------------------------

    figure.add_trace(
        go.Scatter(
            x=trough_times,
            y=gyro_signal[troughs],
            mode="markers",
            name="Negative troughs USED",
            marker=dict(
                size=15,
                symbol="star"
            ),
            hovertemplate=(
                "NEGATIVE TROUGH USED"
                "<br>Time: %{x:.3f} s"
                "<br>Value: %{y:.3f}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    # --------------------------------------------------------
    # SHOW THRESHOLDS ON GRAPH
    # --------------------------------------------------------

    figure.add_hline(
        y=positive_threshold,
        line_dash="dash",
        row=2,
        col=1
    )

    figure.add_hline(
        y=negative_threshold,
        line_dash="dash",
        row=2,
        col=1
    )

    # --------------------------------------------------------
    # FORMATTING
    # --------------------------------------------------------

    figure.update_xaxes(
        title_text="Time from recording start (seconds)",
        row=3,
        col=1,
        showgrid=True
    )

    figure.update_layout(
        title=(
            f"Interactive freestyle IMU recording: "
            f"{csv_path.name}"
        ),
        height=900,
        hovermode="x unified",
        hoverlabel=dict(
            showarrow=True
        ),
        template="plotly_white",
        legend=dict(
            orientation="h",
            y=1.02,
            x=0
        ),
        margin=dict(
            t=110,
            r=30,
            b=60,
            l=70
        ),
    )

    figure.update_layout(
        dragmode="zoom"
    )

    # --------------------------------------------------------
    # SAVE HTML
    # --------------------------------------------------------

    output_file = (
        output_dir
        / f"{csv_path.stem}_peaks_and_troughs.html"
    )

    figure.write_html(
        output_file,
        include_plotlyjs=True,
        full_html=True
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create an interactive graph for one "
            "freestyle IMU recording."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR
    )

    parser.add_argument(
        "--file",
        default=DEFAULT_FILE_NAME,
        help="Recording filename to plot."
    )

    args = parser.parse_args()

    if not args.data_dir.exists():

        raise FileNotFoundError(
            f"Data directory not found: "
            f"{args.data_dir}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = find_recording(
        args.data_dir,
        args.file
    )

    plot_freestyle_file(
        csv_path,
        args.output_dir
    )

    print(
        f"Created interactive graph for "
        f"{csv_path.name} in "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Build and manage a one-recording-per-participant manual labeling queue.

What this script does
---------------------
1. Reads:
      Code/labels/recording_manifest.csv
      Code/labels/participant_split.csv
      Code/labels/label_coverage.csv
      Code/labels/transition_ranges.csv

2. Chooses ONE freestyle recording per participant for the first labeling pass.
   - If that participant already has a labeled recording, it keeps that recording.
   - Otherwise it chooses a representative recording whose duration is closest
     to that participant's median recording duration.

3. Creates/refreshes:
      Code/labels/labeling_queue.csv

4. Finds the NEXT recording that still needs labeling.

5. Creates an interactive HTML graph for the labeling interval and opens it
   automatically in your browser.

6. Prints the exact CSV row formats to add to:
      Code/labels/transition_ranges.csv
      Code/labels/label_coverage.csv

Run:
    python Code/labeling_queue.py

Optional:
    python Code/labeling_queue.py --participant 6
    python Code/labeling_queue.py --no-open
    python Code/labeling_queue.py --queue-only
"""

from pathlib import Path
import argparse
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    REPO_ROOT
    / "Brunner Data"
    / "swimming-recognition-lap-counting-master"
    / "data"
    / "processed_30hz_relabeled"
)

LABEL_DIR = REPO_ROOT / "Code" / "labels"

MANIFEST_FILE = LABEL_DIR / "recording_manifest.csv"
PARTICIPANT_SPLIT_FILE = LABEL_DIR / "participant_split.csv"
TRANSITION_FILE = LABEL_DIR / "transition_ranges.csv"
COVERAGE_FILE = LABEL_DIR / "label_coverage.csv"

QUEUE_FILE = LABEL_DIR / "labeling_queue.csv"

GRAPH_OUTPUT_DIR = (
    REPO_ROOT
    / "Code"
    / "outputs"
    / "labeling_queue"
)


# ============================================================
# LABELING SETTINGS
# ============================================================

TARGET_LABEL_SECONDS = 130.0

ACC_COLUMNS = [
    "ACC_0",
    "ACC_1",
    "ACC_2",
]

GYRO_COLUMNS = [
    "GYRO_0",
    "GYRO_1",
    "GYRO_2",
]

SPLIT_ORDER = {
    "train": 0,
    "validation": 1,
    "test": 2,
}


# ============================================================
# HELPERS
# ============================================================

def normalize_participant(series):
    """
    Make participant IDs comparable across all CSV files.
    """
    return (
        series
        .astype(str)
        .str.strip()
    )


def require_file(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def merge_ranges(ranges):
    """
    Merge overlapping or touching time ranges.
    """
    if not ranges:
        return []

    ranges = sorted(
        (float(start), float(end))
        for start, end in ranges
    )

    merged = [
        [ranges[0][0], ranges[0][1]]
    ]

    for start, end in ranges[1:]:

        previous = merged[-1]

        if start <= previous[1]:
            previous[1] = max(
                previous[1],
                end
            )
        else:
            merged.append(
                [start, end]
            )

    return [
        (start, end)
        for start, end in merged
    ]


def interval_is_covered(
    start,
    end,
    coverage_ranges,
    tolerance=0.25,
):
    """
    Return True only if the entire requested labeling interval
    is contained inside the manually inspected coverage.
    """

    for coverage_start, coverage_end in coverage_ranges:

        if (
            coverage_start <= start + tolerance
            and coverage_end >= end - tolerance
        ):
            return True

    return False


def find_csv_path(participant, filename):
    """
    Construct the recording path from participant + file.
    """
    path = (
        DATA_DIR
        / str(participant)
        / filename
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Recording not found:\n{path}"
        )

    return path


def add_elapsed_time(df):
    """
    Convert nanosecond timestamps to elapsed seconds.
    """
    if "timestamp" not in df.columns:
        raise ValueError(
            "CSV does not contain a 'timestamp' column."
        )

    timestamps = pd.to_numeric(
        df["timestamp"],
        errors="coerce"
    )

    if timestamps.isna().all():
        raise ValueError(
            "Could not parse timestamps."
        )

    first_valid = timestamps.dropna().iloc[0]

    df = df.copy()

    df["time"] = (
        timestamps - first_valid
    ) / 1_000_000_000

    return df


# ============================================================
# LOAD PROJECT TABLES
# ============================================================

def load_tables():

    require_file(
        MANIFEST_FILE,
        "Recording manifest"
    )

    require_file(
        PARTICIPANT_SPLIT_FILE,
        "Participant split"
    )

    require_file(
        TRANSITION_FILE,
        "Transition label file"
    )

    require_file(
        COVERAGE_FILE,
        "Label coverage file"
    )

    manifest = pd.read_csv(
        MANIFEST_FILE
    )

    splits = pd.read_csv(
        PARTICIPANT_SPLIT_FILE
    )

    transitions = pd.read_csv(
        TRANSITION_FILE
    )

    coverage = pd.read_csv(
        COVERAGE_FILE
    )

    for frame in [
        manifest,
        splits,
        transitions,
        coverage,
    ]:
        frame["participant"] = (
            normalize_participant(
                frame["participant"]
            )
        )

    required_split_columns = {
        "participant",
        "split",
    }

    missing = (
        required_split_columns
        - set(splits.columns)
    )

    if missing:
        raise ValueError(
            "participant_split.csv is missing "
            f"columns: {missing}"
        )

    return (
        manifest,
        splits,
        transitions,
        coverage,
    )


# ============================================================
# CHOOSE ONE RECORDING PER PARTICIPANT
# ============================================================

def choose_recording_for_participant(
    participant_manifest,
    participant_coverage,
):
    """
    Selection strategy for the first labeling pass:

    1. Prefer a recording that already has manual coverage.
       This preserves work already completed.

    2. Otherwise choose the recording whose duration is closest
       to the participant's median recording duration.
       This avoids automatically choosing an extreme short/long file.
    """

    participant_manifest = (
        participant_manifest
        .copy()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # FIRST: preserve an already-labeled recording
    # --------------------------------------------------------

    covered_files = set(
        participant_coverage["file"]
        .astype(str)
        .tolist()
    )

    already_covered = (
        participant_manifest[
            participant_manifest["file"]
            .isin(covered_files)
        ]
        .copy()
    )

    if not already_covered.empty:

        # Compute how many manually inspected seconds each candidate has.
        coverage_seconds = {}

        for filename in already_covered["file"]:

            rows = participant_coverage[
                participant_coverage["file"]
                == filename
            ]

            ranges = merge_ranges(
                list(
                    zip(
                        rows["labeled_start_s"],
                        rows["labeled_end_s"],
                    )
                )
            )

            total = sum(
                end - start
                for start, end in ranges
            )

            coverage_seconds[filename] = total

        already_covered[
            "_coverage_seconds"
        ] = (
            already_covered["file"]
            .map(coverage_seconds)
        )

        # Prefer the file with the most existing labeled coverage.
        # Tie-break by shorter duration.
        already_covered = (
            already_covered
            .sort_values(
                [
                    "_coverage_seconds",
                    "duration_s",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
        )

        selected = (
            already_covered
            .iloc[0]
        )

        return (
            selected,
            "existing manual labels"
        )

    # --------------------------------------------------------
    # OTHERWISE: choose a representative-duration recording
    # --------------------------------------------------------

    durations = pd.to_numeric(
        participant_manifest[
            "duration_s"
        ],
        errors="coerce"
    )

    valid = (
        participant_manifest[
            durations.notna()
        ]
        .copy()
    )

    if valid.empty:
        selected = (
            participant_manifest
            .iloc[0]
        )

        return (
            selected,
            "first available recording"
        )

    valid["duration_s"] = (
        pd.to_numeric(
            valid["duration_s"],
            errors="coerce"
        )
    )

    median_duration = (
        valid["duration_s"]
        .median()
    )

    valid[
        "_distance_from_median"
    ] = (
        valid["duration_s"]
        - median_duration
    ).abs()

    valid = valid.sort_values(
        [
            "_distance_from_median",
            "file",
        ]
    )

    selected = valid.iloc[0]

    return (
        selected,
        "representative duration"
    )


# ============================================================
# BUILD / REFRESH LABELING QUEUE
# ============================================================

def build_queue(
    manifest,
    splits,
    coverage,
):

    rows = []

    for _, split_row in splits.iterrows():

        participant = str(
            split_row["participant"]
        )

        split = str(
            split_row["split"]
        ).strip().lower()

        participant_manifest = (
            manifest[
                manifest["participant"]
                == participant
            ]
            .copy()
        )

        if participant_manifest.empty:

            print(
                f"WARNING: participant {participant} "
                "has no freestyle recordings in the manifest."
            )

            continue

        participant_coverage = (
            coverage[
                coverage["participant"]
                == participant
            ]
            .copy()
        )

        selected, reason = (
            choose_recording_for_participant(
                participant_manifest,
                participant_coverage,
            )
        )

        filename = str(
            selected["file"]
        )

        duration_s = float(
            selected["duration_s"]
        )

        label_start_s = 0.0

        label_end_s = min(
            TARGET_LABEL_SECONDS,
            duration_s,
        )

        file_coverage = (
            participant_coverage[
                participant_coverage["file"]
                == filename
            ]
        )

        coverage_ranges = merge_ranges(
            list(
                zip(
                    file_coverage[
                        "labeled_start_s"
                    ],
                    file_coverage[
                        "labeled_end_s"
                    ],
                )
            )
        )

        done = interval_is_covered(
            label_start_s,
            label_end_s,
            coverage_ranges,
        )

        status = (
            "done"
            if done
            else "to_label"
        )

        html_name = (
            f"participant_{participant}_"
            f"{Path(filename).stem}_"
            f"{label_start_s:.0f}_to_"
            f"{label_end_s:.0f}s.html"
        )

        html_path = (
            GRAPH_OUTPUT_DIR
            / html_name
        )

        rows.append(
            {
                "participant":
                    participant,

                "split":
                    split,

                "file":
                    filename,

                "duration_s":
                    round(
                        duration_s,
                        2
                    ),

                "label_start_s":
                    round(
                        label_start_s,
                        2
                    ),

                "label_end_s":
                    round(
                        label_end_s,
                        2
                    ),

                "status":
                    status,

                "selection_reason":
                    reason,

                "html_file":
                    str(
                        html_path
                    ),
            }
        )

    queue = pd.DataFrame(
        rows
    )

    if queue.empty:
        raise ValueError(
            "No queue entries were created."
        )

    queue["_split_order"] = (
        queue["split"]
        .map(SPLIT_ORDER)
        .fillna(99)
    )

    queue["_participant_number"] = (
        pd.to_numeric(
            queue["participant"],
            errors="coerce"
        )
    )

    queue = (
        queue
        .sort_values(
            [
                "_split_order",
                "_participant_number",
                "participant",
            ]
        )
        .drop(
            columns=[
                "_split_order",
                "_participant_number",
            ]
        )
        .reset_index(drop=True)
    )

    LABEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    queue.to_csv(
        QUEUE_FILE,
        index=False
    )

    return queue


# ============================================================
# CREATE INTERACTIVE MANUAL-LABELING GRAPH
# ============================================================

def create_labeling_graph(
    participant,
    split,
    filename,
    label_start_s,
    label_end_s,
    transitions,
):
    """
    Make one HTML graph containing:
        - ACC_0, ACC_1, ACC_2
        - GYRO_0, GYRO_1, GYRO_2
        - acceleration and gyroscope magnitudes

    Only the requested labeling interval is displayed.
    """

    csv_path = find_csv_path(
        participant,
        filename,
    )

    df = pd.read_csv(
        csv_path
    )

    df = df.drop(
        columns=[
            column
            for column in df.columns
            if str(column).startswith(
                "Unnamed"
            )
        ],
        errors="ignore"
    )

    df = add_elapsed_time(
        df
    )

    required_sensor_columns = (
        ACC_COLUMNS
        + GYRO_COLUMNS
    )

    missing = [
        column
        for column
        in required_sensor_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{filename} is missing sensor columns: "
            f"{missing}"
        )

    # Keep only the manually labeling interval.
    display_df = (
        df[
            (df["time"] >= label_start_s)
            & (df["time"] <= label_end_s)
        ]
        .copy()
    )

    if display_df.empty:
        raise ValueError(
            "The requested labeling interval "
            "contains no samples."
        )

    for column in required_sensor_columns:
        display_df[column] = pd.to_numeric(
            display_df[column],
            errors="coerce"
        )

    display_df["ACC_MAG"] = np.sqrt(
        display_df["ACC_0"] ** 2
        + display_df["ACC_1"] ** 2
        + display_df["ACC_2"] ** 2
    )

    display_df["GYRO_MAG"] = np.sqrt(
        display_df["GYRO_0"] ** 2
        + display_df["GYRO_1"] ** 2
        + display_df["GYRO_2"] ** 2
    )

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            "Accelerometer",
            "Gyroscope",
            "Signal magnitudes",
        ),
    )

    for column in ACC_COLUMNS:

        figure.add_trace(
            go.Scatter(
                x=display_df["time"],
                y=display_df[column],
                mode="lines",
                name=column,
                hovertemplate=(
                    "Time: %{x:.3f} s"
                    "<br>Value: %{y:.6f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            ),
            row=1,
            col=1,
        )

    for column in GYRO_COLUMNS:

        figure.add_trace(
            go.Scatter(
                x=display_df["time"],
                y=display_df[column],
                mode="lines",
                name=column,
                hovertemplate=(
                    "Time: %{x:.3f} s"
                    "<br>Value: %{y:.6f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            ),
            row=2,
            col=1,
        )

    for column in [
        "ACC_MAG",
        "GYRO_MAG",
    ]:

        figure.add_trace(
            go.Scatter(
                x=display_df["time"],
                y=display_df[column],
                mode="lines",
                name=column,
                hovertemplate=(
                    "Time: %{x:.3f} s"
                    "<br>Value: %{y:.6f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            ),
            row=3,
            col=1,
        )

    # --------------------------------------------------------
    # If this recording already has transition labels,
    # display them as shaded regions.
    # --------------------------------------------------------

    current_transitions = (
        transitions[
            (
                transitions["participant"]
                == str(participant)
            )
            &
            (
                transitions["file"]
                == filename
            )
        ]
    )

    for _, transition in (
        current_transitions.iterrows()
    ):

        start = float(
            transition["start_s"]
        )

        end = float(
            transition["end_s"]
        )

        if (
            end < label_start_s
            or start > label_end_s
        ):
            continue

        figure.add_vrect(
            x0=max(
                start,
                label_start_s
            ),
            x1=min(
                end,
                label_end_s
            ),
            opacity=0.16,
            line_width=0,
            annotation_text="Existing transition label",
            annotation_position="top left",
        )

    figure.update_yaxes(
        title_text="Acceleration",
        row=1,
        col=1,
    )

    figure.update_yaxes(
        title_text="Angular velocity",
        row=2,
        col=1,
    )

    figure.update_yaxes(
        title_text="Magnitude",
        row=3,
        col=1,
    )

    figure.update_xaxes(
        title_text="Time from recording start (seconds)",
        row=3,
        col=1,
    )

    figure.update_layout(
        title=(
            f"MANUAL TRANSITION LABELING"
            f"<br><sup>"
            f"Participant {participant} | "
            f"{split.upper()} | "
            f"{filename} | "
            f"Inspect {label_start_s:.1f}–"
            f"{label_end_s:.1f} s"
            f"</sup>"
        ),
        height=950,
        template="plotly_white",
        hovermode="x unified",
        dragmode="zoom",
        legend=dict(
            orientation="h",
            y=1.04,
            x=0,
        ),
        margin=dict(
            t=125,
            r=30,
            b=60,
            l=70,
        ),
    )

    GRAPH_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    html_name = (
        f"participant_{participant}_"
        f"{Path(filename).stem}_"
        f"{label_start_s:.0f}_to_"
        f"{label_end_s:.0f}s.html"
    )

    html_path = (
        GRAPH_OUTPUT_DIR
        / html_name
    )

    figure.write_html(
        html_path,
        include_plotlyjs=True,
        full_html=True,
    )

    return html_path


# ============================================================
# PRINT USER INSTRUCTIONS
# ============================================================

def print_labeling_instructions(
    row,
    html_path,
):

    participant = str(
        row["participant"]
    )

    split = str(
        row["split"]
    )

    filename = str(
        row["file"]
    )

    start = float(
        row["label_start_s"]
    )

    end = float(
        row["label_end_s"]
    )

    print()
    print("=" * 78)
    print("NEXT RECORDING TO LABEL")
    print("=" * 78)

    print(
        f"Participant: {participant}"
    )

    print(
        f"Split:       {split}"
    )

    print(
        f"Recording:   {filename}"
    )

    print(
        f"Inspect:     {start:.2f} s "
        f"to {end:.2f} s"
    )

    print()
    print("HTML graph:")
    print(html_path)

    print()
    print(
        "Look through that entire time interval and "
        "identify every transition / turn / "
        "non-clean-freestyle region."
    )

    print()
    print("-" * 78)
    print("1. ADD EACH TRANSITION TO transition_ranges.csv")
    print("-" * 78)

    print()
    print(
        "File:"
    )
    print(
        TRANSITION_FILE
    )

    print()
    print(
        "Use ONE row per transition:"
    )

    print()
    print(
        f"{participant},{filename},"
        "<start_s>,<end_s>"
    )

    print()
    print("Example only:")
    print(
        f"{participant},{filename},"
        "61.2,67.4"
    )

    print()
    print(
        "If there are 4 transitions, add 4 rows."
    )

    print(
        "If there are ZERO transitions, add no "
        "transition rows."
    )

    print()
    print("-" * 78)
    print("2. MARK THE INTERVAL AS MANUALLY INSPECTED")
    print("-" * 78)

    print()
    print(
        "After you have inspected the ENTIRE graph, "
        "add this ONE row to label_coverage.csv:"
    )

    print()
    print(
        f"{participant},{filename},"
        f"{start:.2f},{end:.2f}"
    )

    print()
    print(
        "File:"
    )
    print(
        COVERAGE_FILE
    )

    print()
    print(
        "IMPORTANT: add the coverage row even if "
        "you found zero transitions."
    )

    print()
    print("-" * 78)
    print("3. SAVE BOTH CSV FILES AND RUN THIS SCRIPT AGAIN")
    print("-" * 78)

    print()
    print(
        "python Code/labeling_queue.py"
    )

    print()
    print(
        "The queue will detect that this interval "
        "is complete and move to the next swimmer."
    )

    print()
    print("=" * 78)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build the manual transition labeling queue "
            "and open the next IMU graph."
        )
    )

    parser.add_argument(
        "--participant",
        default=None,
        help=(
            "Generate the queue graph for a specific "
            "participant instead of the next unfinished one."
        ),
    )

    parser.add_argument(
        "--no-open",
        action="store_true",
        help=(
            "Create the HTML file but do not automatically "
            "open it in the browser."
        ),
    )

    parser.add_argument(
        "--queue-only",
        action="store_true",
        help=(
            "Refresh labeling_queue.csv but do not create/open "
            "a graph."
        ),
    )

    args = parser.parse_args()

    (
        manifest,
        splits,
        transitions,
        coverage,
    ) = load_tables()

    queue = build_queue(
        manifest,
        splits,
        coverage,
    )

    done_count = (
        queue["status"]
        == "done"
    ).sum()

    remaining_count = (
        queue["status"]
        == "to_label"
    ).sum()

    print()
    print("=" * 78)
    print("LABELING QUEUE")
    print("=" * 78)

    print(
        f"Participants in queue: {len(queue)}"
    )

    print(
        f"Done:                  {done_count}"
    )

    print(
        f"Still to label:        {remaining_count}"
    )

    print()
    print(
        f"Queue CSV:\n{QUEUE_FILE}"
    )

    if args.queue_only:
        return

    # --------------------------------------------------------
    # SELECT RECORDING
    # --------------------------------------------------------

    if args.participant is not None:

        requested = str(
            args.participant
        ).strip()

        matches = queue[
            queue["participant"]
            == requested
        ]

        if matches.empty:
            raise ValueError(
                f"Participant {requested} is not in "
                "participant_split.csv or has no "
                "freestyle recording."
            )

        selected = (
            matches.iloc[0]
        )

    else:

        remaining = queue[
            queue["status"]
            == "to_label"
        ]

        if remaining.empty:

            print()
            print(
                "All participants in the first-pass "
                "labeling queue are complete."
            )

            return

        selected = (
            remaining.iloc[0]
        )

    # --------------------------------------------------------
    # CREATE GRAPH
    # --------------------------------------------------------

    html_path = create_labeling_graph(
        participant=selected[
            "participant"
        ],
        split=selected[
            "split"
        ],
        filename=selected[
            "file"
        ],
        label_start_s=float(
            selected[
                "label_start_s"
            ]
        ),
        label_end_s=float(
            selected[
                "label_end_s"
            ]
        ),
        transitions=transitions,
    )

    print_labeling_instructions(
        selected,
        html_path,
    )

    if not args.no_open:

        webbrowser.open(
            html_path.resolve().as_uri()
        )


if __name__ == "__main__":
    main()

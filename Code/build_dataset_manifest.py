#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


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

TRANSITION_LABEL_FILE = (
    REPO_ROOT
    / "Code"
    / "labels"
    / "transition_ranges.csv"
)

LABEL_COVERAGE_FILE = (
    REPO_ROOT
    / "Code"
    / "labels"
    / "label_coverage.csv"
)

MANIFEST_OUTPUT_FILE = (
    REPO_ROOT
    / "Code"
    / "labels"
    / "recording_manifest.csv"
)

PARTICIPANT_OUTPUT_FILE = (
    REPO_ROOT
    / "Code"
    / "labels"
    / "participant_summary.csv"
)


# ============================================================
# LOAD LABEL FILES
# ============================================================

transition_labels = pd.read_csv(
    TRANSITION_LABEL_FILE
)

coverage = pd.read_csv(
    LABEL_COVERAGE_FILE
)


# Make participant IDs consistent
transition_labels["participant"] = (
    transition_labels["participant"]
    .astype(str)
)

coverage["participant"] = (
    coverage["participant"]
    .astype(str)
)


# ============================================================
# HELPER: MERGE COVERAGE RANGES
# ============================================================

def merge_ranges(ranges):
    """
    Merge overlapping labeled time ranges.

    Example:

        [(0, 130), (100, 200)]

    becomes:

        [(0, 200)]
    """

    if not ranges:
        return []

    ranges = sorted(ranges)

    merged = [
        list(ranges[0])
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
        tuple(x)
        for x in merged
    ]


# ============================================================
# FIND ALL FREESTYLE FILES
# ============================================================

freestyle_files = sorted(
    DATA_DIR.glob(
        "*/Freestyle_*.csv"
    )
)

print()
print(
    f"Found {len(freestyle_files)} "
    f"freestyle recordings."
)


# ============================================================
# BUILD RECORDING MANIFEST
# ============================================================

rows = []

for csv_path in freestyle_files:

    participant = csv_path.parent.name
    filename = csv_path.name

    print(
        f"Reading participant "
        f"{participant}: {filename}"
    )

    df = pd.read_csv(
        csv_path
    )

    num_samples = len(df)

    # --------------------------------------------------------
    # RECORDING DURATION
    # --------------------------------------------------------

    duration_s = None

    if (
        num_samples > 0
        and "timestamp" in df.columns
    ):

        timestamps = pd.to_numeric(
            df["timestamp"],
            errors="coerce"
        ).dropna()

        if len(timestamps) >= 2:

            duration_s = (
                timestamps.iloc[-1]
                - timestamps.iloc[0]
            ) / 1_000_000_000

    # --------------------------------------------------------
    # TRANSITION RANGES
    # --------------------------------------------------------

    file_transitions = transition_labels[
        (
            transition_labels["participant"]
            == participant
        )
        &
        (
            transition_labels["file"]
            == filename
        )
    ]

    num_transition_ranges = len(
        file_transitions
    )

    # --------------------------------------------------------
    # LABEL COVERAGE
    # --------------------------------------------------------

    file_coverage = coverage[
        (
            coverage["participant"]
            == participant
        )
        &
        (
            coverage["file"]
            == filename
        )
    ]

    coverage_ranges = []

    for _, row in file_coverage.iterrows():

        coverage_ranges.append(
            (
                float(
                    row["labeled_start_s"]
                ),
                float(
                    row["labeled_end_s"]
                ),
            )
        )

    coverage_ranges = merge_ranges(
        coverage_ranges
    )

    labeled_duration_s = sum(
        end - start
        for start, end
        in coverage_ranges
    )

    # --------------------------------------------------------
    # LABEL STATUS
    # --------------------------------------------------------

    if len(coverage_ranges) == 0:

        label_status = "unlabeled"

    else:

        first_start = (
            coverage_ranges[0][0]
        )

        last_end = (
            coverage_ranges[-1][1]
        )

        # Allow a small tolerance for timestamp differences.
        tolerance_s = 0.5

        covers_start = (
            first_start
            <= tolerance_s
        )

        covers_end = (
            duration_s is not None
            and last_end
            >= duration_s - tolerance_s
        )

        # Full coverage should also not contain gaps.
        one_continuous_range = (
            len(coverage_ranges) == 1
        )

        if (
            covers_start
            and covers_end
            and one_continuous_range
        ):

            label_status = "full"

        else:

            label_status = "partial"

    # --------------------------------------------------------
    # PERCENT OF RECORDING LABELED
    # --------------------------------------------------------

    if (
        duration_s is not None
        and duration_s > 0
    ):

        percent_labeled = (
            100
            * labeled_duration_s
            / duration_s
        )

        percent_labeled = min(
            percent_labeled,
            100
        )

    else:

        percent_labeled = 0

    # --------------------------------------------------------
    # SAVE RECORDING
    # --------------------------------------------------------

    rows.append(
        {
            "participant":
                participant,

            "file":
                filename,

            "duration_s":
                round(
                    duration_s,
                    2
                )
                if duration_s
                is not None
                else None,

            "num_samples":
                num_samples,

            "label_status":
                label_status,

            "labeled_duration_s":
                round(
                    labeled_duration_s,
                    2
                ),

            "percent_labeled":
                round(
                    percent_labeled,
                    1
                ),

            "num_transition_ranges":
                num_transition_ranges,
        }
    )


# ============================================================
# RECORDING MANIFEST
# ============================================================

manifest = pd.DataFrame(
    rows
)

manifest[
    "participant_number"
] = pd.to_numeric(
    manifest["participant"],
    errors="coerce"
)

manifest = manifest.sort_values(
    [
        "participant_number",
        "file",
    ]
)

manifest = manifest.drop(
    columns=[
        "participant_number"
    ]
)

manifest.to_csv(
    MANIFEST_OUTPUT_FILE,
    index=False
)


# ============================================================
# PARTICIPANT SUMMARY
# ============================================================

participant_summary = (
    manifest.groupby(
        "participant"
    )
    .agg(
        recordings=(
            "file",
            "count"
        ),

        total_duration_s=(
            "duration_s",
            "sum"
        ),

        labeled_duration_s=(
            "labeled_duration_s",
            "sum"
        ),

        transition_ranges=(
            "num_transition_ranges",
            "sum"
        ),

        full_recordings=(
            "label_status",
            lambda x:
            (x == "full").sum()
        ),

        partial_recordings=(
            "label_status",
            lambda x:
            (x == "partial").sum()
        ),

        unlabeled_recordings=(
            "label_status",
            lambda x:
            (x == "unlabeled").sum()
        ),
    )
    .reset_index()
)

participant_summary[
    "participant_number"
] = pd.to_numeric(
    participant_summary[
        "participant"
    ],
    errors="coerce"
)

participant_summary = (
    participant_summary
    .sort_values(
        "participant_number"
    )
    .drop(
        columns=[
            "participant_number"
        ]
    )
)

participant_summary[
    "total_duration_s"
] = (
    participant_summary[
        "total_duration_s"
    ]
    .round(2)
)

participant_summary[
    "labeled_duration_s"
] = (
    participant_summary[
        "labeled_duration_s"
    ]
    .round(2)
)

participant_summary.to_csv(
    PARTICIPANT_OUTPUT_FILE,
    index=False
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print()
print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(
    f"Freestyle recordings: "
    f"{len(manifest)}"
)

print(
    f"Participants with freestyle: "
    f"{manifest['participant'].nunique()}"
)

print(
    f"Fully labeled recordings: "
    f"{(manifest['label_status'] == 'full').sum()}"
)

print(
    f"Partially labeled recordings: "
    f"{(manifest['label_status'] == 'partial').sum()}"
)

print(
    f"Unlabeled recordings: "
    f"{(manifest['label_status'] == 'unlabeled').sum()}"
)

print(
    f"Total labeled duration: "
    f"{manifest['labeled_duration_s'].sum() / 60:.2f} minutes"
)


# ============================================================
# CURRENT LABELED RECORDINGS
# ============================================================

print()
print("=" * 70)
print("CURRENTLY LABELED RECORDINGS")
print("=" * 70)

currently_labeled = manifest[
    manifest["label_status"]
    != "unlabeled"
]

print(
    currently_labeled[
        [
            "participant",
            "file",
            "duration_s",
            "label_status",
            "labeled_duration_s",
            "percent_labeled",
            "num_transition_ranges",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# PARTICIPANT SUMMARY
# ============================================================

print()
print("=" * 70)
print("PARTICIPANT SUMMARY")
print("=" * 70)

print(
    participant_summary.to_string(
        index=False
    )
)


print()
print(
    f"Recording manifest saved to:\n"
    f"{MANIFEST_OUTPUT_FILE}"
)

print()
print(
    f"Participant summary saved to:\n"
    f"{PARTICIPANT_OUTPUT_FILE}"
)
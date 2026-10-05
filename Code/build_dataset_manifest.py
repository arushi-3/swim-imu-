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

LABEL_FILE = (
    REPO_ROOT
    / "Code"
    / "labels"
    / "transition_ranges.csv"
)

OUTPUT_FILE = (
    REPO_ROOT
    / "Code"
    / "labels"
    / "recording_manifest.csv"
)


# ============================================================
# LOAD TRANSITION LABELS
# ============================================================

if not LABEL_FILE.exists():
    raise FileNotFoundError(
        f"Transition label file not found:\n{LABEL_FILE}"
    )

labels = pd.read_csv(LABEL_FILE)

required_columns = {
    "participant",
    "file",
    "start_s",
    "end_s",
}

missing_columns = required_columns - set(labels.columns)

if missing_columns:
    raise ValueError(
        f"transition_ranges.csv is missing columns: "
        f"{missing_columns}"
    )

# Make participant IDs strings so comparisons are consistent.
labels["participant"] = (
    labels["participant"]
    .astype(str)
)


# ============================================================
# FIND ALL FREESTYLE RECORDINGS
# ============================================================

freestyle_files = sorted(
    DATA_DIR.glob("*/Freestyle_*.csv")
)

print()
print(
    f"Found {len(freestyle_files)} freestyle recordings."
)


# ============================================================
# BUILD MANIFEST
# ============================================================

rows = []

for csv_path in freestyle_files:

    participant = csv_path.parent.name
    filename = csv_path.name

    print(
        f"Reading participant {participant}: "
        f"{filename}"
    )

    df = pd.read_csv(csv_path)

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
    # CHECK WHETHER THIS RECORDING HAS LABELS
    # --------------------------------------------------------

    recording_labels = labels[
        (labels["participant"] == participant)
        & (labels["file"] == filename)
    ]

    number_of_transitions = len(
        recording_labels
    )

    labeled = (
        "yes"
        if number_of_transitions > 0
        else "no"
    )

    # --------------------------------------------------------
    # SAVE ROW
    # --------------------------------------------------------

    rows.append(
        {
            "participant": participant,
            "file": filename,
            "duration_s": (
                round(duration_s, 2)
                if duration_s is not None
                else None
            ),
            "num_samples": num_samples,
            "labeled": labeled,
            "num_transition_ranges":
                number_of_transitions,
        }
    )


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest = pd.DataFrame(rows)

# Sort numerically by participant
manifest["participant_number"] = (
    pd.to_numeric(
        manifest["participant"],
        errors="coerce"
    )
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
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

number_of_recordings = len(
    manifest
)

number_of_participants = (
    manifest["participant"]
    .nunique()
)

number_labeled = (
    manifest["labeled"]
    == "yes"
).sum()

number_unlabeled = (
    manifest["labeled"]
    == "no"
).sum()


print()
print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(
    f"Freestyle recordings: "
    f"{number_of_recordings}"
)

print(
    f"Participants with freestyle: "
    f"{number_of_participants}"
)

print(
    f"Labeled recordings: "
    f"{number_labeled}"
)

print(
    f"Unlabeled recordings: "
    f"{number_unlabeled}"
)

print()
print(
    f"Manifest saved to:\n"
    f"{OUTPUT_FILE}"
)


# ============================================================
# SHOW CURRENTLY LABELED FILES
# ============================================================

print()
print("=" * 60)
print("CURRENTLY LABELED RECORDINGS")
print("=" * 60)

labeled_files = manifest[
    manifest["labeled"] == "yes"
]

print(
    labeled_files[
        [
            "participant",
            "file",
            "duration_s",
            "num_transition_ranges",
        ]
    ].to_string(
        index=False
    )
)
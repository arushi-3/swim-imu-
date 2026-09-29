#!/usr/bin/env python3
"""
Train one shallow Random Forest transition classifier using multiple
freestyle IMU CSV recordings.

Labels:
    1 = clean freestyle swimming
    0 = transition / turn / noise

Workflow:
    1. Load five configured freestyle recordings.
    2. Manually label transition ranges for each recording.
    3. Split each recording into 2-second, 50%-overlap windows.
    4. Extract the same 54 features from every window.
    5. Combine the TRAIN recordings and fit ONE Random Forest.
    6. Choose the clean-probability threshold on the VALIDATION recording.
    7. Evaluate once on the TEST recording.
    8. Save per-recording prediction CSVs and HTML plots.

Only the first 130 seconds of every recording are used throughout this script.

If any configured recording still has transition_ranges=None, the script
creates a simple GYRO_1 HTML plot for that recording and stops before
training. Use those plots to identify the transition ranges, enter them in
FILE_CONFIG, and run the script again.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scipy.signal import find_peaks, periodogram

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


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
    / "transition_classifier_multi_file"
)


# ============================================================
# FIVE RECORDINGS
# ============================================================
#
# I selected five freestyle files visible in your dataset tree.
#
# split:
#     train      -> used to fit the Random Forest
#     validation -> used to choose the probability threshold
#     test       -> used only for final evaluation
#
# IMPORTANT:
# Replace every None below with the correct transition ranges before
# training. The one recording you already labeled is filled in.
#
# Example:
#     "transition_ranges": [
#         (0.0, 8.4),
#         (55.2, 61.0),
#         (110.0, 120.0),
#     ]
#
# Do NOT use [] unless the recording genuinely contains no transitions.

FILE_CONFIG = {
    "0/Freestyle_1527873200322.csv": {
        "split": "train",
        "transition_ranges": [
            (0.0, 73.9),
            (117.7, 119.3),
        ],
    },
    "1/Freestyle_1527676714707.csv": {
        "split": "train",
        "transition_ranges": [
            (0.0, 91.7),
        ],
    },
    "2/Freestyle_1527072298681.csv": {
        "split": "train",
        "transition_ranges": [
            (0.0, 40.5),
            (75, 81.2),
            (120.0, 130.0),
        ],
    },
    "4/Freestyle_1526810816300.csv": {
        "split": "validation",
        "transition_ranges": [
            (0.0, 12.1),
            (60.9, 66.9),
            (115.2, 124.2),
        ],
    },
    "4/Freestyle_1526810977061.csv": {
        "split": "test",
        "transition_ranges": [
            (0.0, 18.07),
            (65.1, 68.9),
            (120.0, 125.4),
        ],
    },
}


# ============================================================
# SENSOR COLUMNS
# ============================================================

ACC_COLUMNS = ["ACC_0", "ACC_1", "ACC_2"]
GYRO_COLUMNS = ["GYRO_0", "GYRO_1", "GYRO_2"]


# ============================================================
# RECORDING TIME LIMIT
# ============================================================

# Only the first 130 seconds of every recording are used for:
#   - manual-labeling plots
#   - feature extraction
#   - Random Forest training
#   - validation
#   - testing
#   - prediction plots
MAX_RECORDING_SECONDS = 130.0


# ============================================================
# WINDOW SETTINGS
# ============================================================

WINDOW_SECONDS = 2.0
WINDOW_OVERLAP = 0.5
CLEAN_FRACTION_THRESHOLD = 0.90


# ============================================================
# RANDOM FOREST SETTINGS
# ============================================================

NUMBER_OF_TREES = 100
MAX_TREE_DEPTH = 4
MIN_SAMPLES_LEAF = 3


# ============================================================
# CLEAN-PROBABILITY THRESHOLD SETTINGS
# ============================================================

CLEAN_PROBABILITY_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]


# ============================================================
# PEAK FEATURE SETTINGS
# ============================================================

STROKE_AXIS = "GYRO_1"
PEAK_PROMINENCE_FACTOR = 0.5
MIN_PEAK_DISTANCE_SECONDS = 0.5


# ============================================================
# HELPERS
# ============================================================

def calculate_rms(signal):
    signal = np.asarray(signal, dtype=float)
    return np.sqrt(np.mean(signal ** 2))


def dominant_frequency(
    signal,
    fs,
    minimum_frequency=0.2,
    maximum_frequency=2.0,
):
    """Return the strongest frequency in the chosen swimming band."""

    signal = np.asarray(signal, dtype=float)

    if len(signal) < 4:
        return 0.0

    signal = signal - np.mean(signal)

    frequencies, power = periodogram(
        signal,
        fs=fs,
    )

    valid = (
        (frequencies >= minimum_frequency)
        & (frequencies <= maximum_frequency)
    )

    frequencies = frequencies[valid]
    power = power[valid]

    if len(power) == 0:
        return 0.0

    maximum_index = np.argmax(power)
    return frequencies[maximum_index]


def periodicity_strength(
    signal,
    fs,
    minimum_frequency=0.2,
    maximum_frequency=2.0,
):
    """Fraction of band power concentrated in the strongest frequency."""

    signal = np.asarray(signal, dtype=float)

    if len(signal) < 4:
        return 0.0

    signal = signal - np.mean(signal)

    frequencies, power = periodogram(
        signal,
        fs=fs,
    )

    valid = (
        (frequencies >= minimum_frequency)
        & (frequencies <= maximum_frequency)
    )

    power = power[valid]

    if len(power) == 0:
        return 0.0

    total_power = np.sum(power)

    if total_power == 0:
        return 0.0

    return np.max(power) / total_power


# ============================================================
# LOAD / PREPARE ONE RECORDING
# ============================================================

def load_recording(csv_path: Path):
    df = pd.read_csv(csv_path)

    df = df.drop(
        columns=[
            column
            for column in df.columns
            if column.startswith("Unnamed")
        ],
        errors="ignore",
    )

    required_columns = [
        "timestamp",
        *ACC_COLUMNS,
        *GYRO_COLUMNS,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required columns: {missing}"
        )

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            f"No usable samples were found in {csv_path.name}."
        )

    return df


def add_time_axis(df):
    timestamps = df["timestamp"]

    df["time"] = (
        timestamps - timestamps.iloc[0]
    ) / 1_000_000_000

    time_differences = np.diff(df["time"])

    time_differences = time_differences[
        np.isfinite(time_differences)
        & (time_differences > 0)
    ]

    if len(time_differences) == 0:
        raise ValueError("Could not determine sampling frequency.")

    fs = 1 / np.median(time_differences)
    return df, fs


def limit_recording_time(
    df,
    max_seconds=MAX_RECORDING_SECONDS,
):
    """
    Keep only the beginning of a recording up to max_seconds.

    Returns:
        limited_df
        full_duration
    """

    full_duration = float(
        df["time"].iloc[-1]
    )

    limited_df = df[
        df["time"] <= max_seconds
    ].copy().reset_index(drop=True)

    if len(limited_df) == 0:
        raise ValueError(
            f"No samples exist within the first "
            f"{max_seconds:.1f} seconds."
        )

    return limited_df, full_duration


def add_magnitudes(df):
    df["ACC_MAG"] = np.sqrt(
        df["ACC_0"] ** 2
        + df["ACC_1"] ** 2
        + df["ACC_2"] ** 2
    )

    df["GYRO_MAG"] = np.sqrt(
        df["GYRO_0"] ** 2
        + df["GYRO_1"] ** 2
        + df["GYRO_2"] ** 2
    )

    return df


def add_manual_labels(df, transition_ranges):
    """
    1 = clean freestyle
    0 = transition / turn / noise
    """

    df["clean_swimming"] = 1

    for start_time, end_time in transition_ranges:
        transition = (
            (df["time"] >= start_time)
            & (df["time"] <= end_time)
        )

        df.loc[
            transition,
            "clean_swimming",
        ] = 0

    return df


# ============================================================
# LABELING PLOTS FOR FILES THAT STILL NEED TRANSITION RANGES
# ============================================================

def create_labeling_plot(df, csv_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=df["time"],
            y=df[STROKE_AXIS],
            mode="lines",
            name=STROKE_AXIS,
            hovertemplate=(
                "Time: %{x:.3f} s"
                "<br>Value: %{y:.6f}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title=f"Label transition periods: {csv_path.name}",
        xaxis_title="Time (seconds)",
        yaxis_title=STROKE_AXIS,
        template="plotly_white",
        height=650,
        hovermode="x unified",
        dragmode="zoom",
    )

    output_file = output_dir / f"{csv_path.stem}_labeling.html"

    figure.write_html(
        output_file,
        include_plotlyjs=True,
        full_html=True,
    )

    return output_file


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(window, fs):
    features = {}

    signals = [
        "ACC_0",
        "ACC_1",
        "ACC_2",
        "ACC_MAG",
        "GYRO_0",
        "GYRO_1",
        "GYRO_2",
        "GYRO_MAG",
    ]

    # 8 signals x 6 statistics = 48 features
    for column in signals:
        signal = window[column].to_numpy()

        features[f"{column}_mean"] = np.mean(signal)
        features[f"{column}_std"] = np.std(signal)
        features[f"{column}_rms"] = calculate_rms(signal)
        features[f"{column}_minimum"] = np.min(signal)
        features[f"{column}_maximum"] = np.max(signal)
        features[f"{column}_range"] = np.max(signal) - np.min(signal)

    stroke_signal = window[STROKE_AXIS].to_numpy()

    # +2 frequency features
    features["dominant_frequency"] = dominant_frequency(
        stroke_signal,
        fs,
    )

    features["periodicity_strength"] = periodicity_strength(
        stroke_signal,
        fs,
    )

    # +4 peak/trough features
    prominence = (
        np.std(stroke_signal)
        * PEAK_PROMINENCE_FACTOR
    )

    minimum_distance = int(
        MIN_PEAK_DISTANCE_SECONDS * fs
    )
    minimum_distance = max(minimum_distance, 1)

    positive_peaks, positive_properties = find_peaks(
        stroke_signal,
        distance=minimum_distance,
        prominence=prominence,
    )

    negative_troughs, negative_properties = find_peaks(
        -stroke_signal,
        distance=minimum_distance,
        prominence=prominence,
    )

    features["positive_peak_count"] = len(positive_peaks)
    features["negative_trough_count"] = len(negative_troughs)

    if len(positive_peaks) > 0:
        features["mean_positive_prominence"] = np.mean(
            positive_properties["prominences"]
        )
    else:
        features["mean_positive_prominence"] = 0.0

    if len(negative_troughs) > 0:
        features["mean_negative_prominence"] = np.mean(
            negative_properties["prominences"]
        )
    else:
        features["mean_negative_prominence"] = 0.0

    return features


def create_windows(
    df,
    fs,
    recording,
    participant,
    split,
):
    samples_per_window = int(
        round(WINDOW_SECONDS * fs)
    )

    step = int(
        round(
            samples_per_window
            * (1 - WINDOW_OVERLAP)
        )
    )
    step = max(step, 1)

    feature_rows = []

    for start_index in range(
        0,
        len(df) - samples_per_window + 1,
        step,
    ):
        end_index = start_index + samples_per_window
        window = df.iloc[start_index:end_index]

        features = extract_features(window, fs)

        start_time = window["time"].iloc[0]
        end_time = window["time"].iloc[-1]
        center_time = (start_time + end_time) / 2

        clean_fraction = window["clean_swimming"].mean()

        label = int(
            clean_fraction >= CLEAN_FRACTION_THRESHOLD
        )

        # Metadata -- NOT used as Random Forest input features.
        features["recording"] = recording
        features["participant"] = participant
        features["split"] = split
        features["start_time"] = start_time
        features["end_time"] = end_time
        features["center_time"] = center_time
        features["clean_fraction"] = clean_fraction
        features["label"] = label

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)


# ============================================================
# FEATURE COLUMNS
# ============================================================

def get_feature_columns(df):
    ignore_columns = [
        "recording",
        "participant",
        "split",
        "start_time",
        "end_time",
        "center_time",
        "clean_fraction",
        "label",
    ]

    return [
        column
        for column in df.columns
        if column not in ignore_columns
    ]


# ============================================================
# RANDOM FOREST
# ============================================================

def train_classifier(train_df, feature_columns):
    X_train = train_df[feature_columns]
    y_train = train_df["label"]

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data contains only one class. "
            "Check the transition ranges for the training recordings."
        )

    model = RandomForestClassifier(
        n_estimators=NUMBER_OF_TREES,
        max_depth=MAX_TREE_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)
    return model


def get_clean_probabilities(
    model,
    dataframe,
    feature_columns,
):
    X = dataframe[feature_columns]
    probabilities = model.predict_proba(X)

    clean_index = list(model.classes_).index(1)
    return probabilities[:, clean_index]


def predictions_from_clean_probability(
    clean_probabilities,
    clean_threshold,
):
    return (
        clean_probabilities >= clean_threshold
    ).astype(int)


# ============================================================
# VALIDATION THRESHOLD SEARCH
# ============================================================

def test_probability_thresholds(
    model,
    dataframe,
    feature_columns,
):
    if len(dataframe) == 0:
        raise ValueError("Validation dataset is empty.")

    y = dataframe["label"].to_numpy()

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Validation data needs both transition and clean windows."
        )

    clean_probabilities = get_clean_probabilities(
        model,
        dataframe,
        feature_columns,
    )

    rows = []

    print()
    print("=" * 78)
    print("CLEAN-PROBABILITY THRESHOLD TEST - VALIDATION FILE(S) ONLY")
    print("=" * 78)
    print()
    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Trans Precision':<18}"
        f"{'Trans Recall':<15}"
        f"{'Trans F1':<12}"
        f"{'Pred Trans':<12}"
    )

    for clean_threshold in CLEAN_PROBABILITY_THRESHOLDS:
        predictions = predictions_from_clean_probability(
            clean_probabilities,
            clean_threshold,
        )

        accuracy = accuracy_score(y, predictions)

        transition_precision = precision_score(
            y,
            predictions,
            pos_label=0,
            zero_division=0,
        )

        transition_recall = recall_score(
            y,
            predictions,
            pos_label=0,
            zero_division=0,
        )

        transition_f1 = f1_score(
            y,
            predictions,
            pos_label=0,
            zero_division=0,
        )

        predicted_transition_count = int(
            np.sum(predictions == 0)
        )

        rows.append(
            {
                "clean_probability_threshold": clean_threshold,
                "accuracy": accuracy,
                "transition_precision": transition_precision,
                "transition_recall": transition_recall,
                "transition_f1": transition_f1,
                "predicted_transition_windows": predicted_transition_count,
            }
        )

        print(
            f"{clean_threshold:<12.2f}"
            f"{accuracy:<12.3f}"
            f"{transition_precision:<18.3f}"
            f"{transition_recall:<15.3f}"
            f"{transition_f1:<12.3f}"
            f"{predicted_transition_count:<12d}"
        )

    return pd.DataFrame(rows)


def choose_probability_threshold(threshold_results):
    ranked = (
        threshold_results
        .sort_values(
            by=[
                "transition_f1",
                "transition_recall",
                "transition_precision",
                "clean_probability_threshold",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    best = ranked.iloc[0]
    clean_threshold = float(
        best["clean_probability_threshold"]
    )

    print()
    print("=" * 50)
    print("SELECTED CLEAN-PROBABILITY THRESHOLD")
    print("=" * 50)
    print(f"Threshold: {clean_threshold:.2f}")
    print(
        "Validation transition precision: "
        f"{best['transition_precision']:.3f}"
    )
    print(
        "Validation transition recall: "
        f"{best['transition_recall']:.3f}"
    )
    print(
        "Validation transition F1: "
        f"{best['transition_f1']:.3f}"
    )
    print()
    print(
        "This threshold was selected using the VALIDATION "
        "recording(s) only."
    )

    return clean_threshold


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    dataframe,
    feature_columns,
    name,
    clean_threshold,
):
    if len(dataframe) == 0:
        print(f"\n{name}: no windows available.")
        return

    y = dataframe["label"].to_numpy()

    clean_probabilities = get_clean_probabilities(
        model,
        dataframe,
        feature_columns,
    )

    predictions = predictions_from_clean_probability(
        clean_probabilities,
        clean_threshold,
    )

    print()
    print("=" * 50)
    print(name)
    print("=" * 50)
    print(f"\nClean-probability threshold: {clean_threshold:.2f}")

    print("\nActual class counts:")
    print(f"Transition (0): {np.sum(y == 0)}")
    print(f"Clean freestyle (1): {np.sum(y == 1)}")

    print("\nPredicted class counts:")
    print(f"Transition (0): {np.sum(predictions == 0)}")
    print(f"Clean freestyle (1): {np.sum(predictions == 1)}")

    print(f"\nAccuracy: {accuracy_score(y, predictions):.3f}")

    print("\nConfusion matrix:\n")
    print(
        confusion_matrix(
            y,
            predictions,
            labels=[0, 1],
        )
    )

    print("\nClassification report:\n")
    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=[
                "Transition",
                "Clean freestyle",
            ],
            zero_division=0,
        )
    )


# ============================================================
# PREDICTIONS / FEATURE IMPORTANCE
# ============================================================

def predict_all_windows(
    model,
    feature_df,
    feature_columns,
    clean_threshold,
):
    output = feature_df.copy()

    clean_probabilities = get_clean_probabilities(
        model,
        output,
        feature_columns,
    )

    output["clean_probability"] = clean_probabilities
    output["prediction"] = predictions_from_clean_probability(
        clean_probabilities,
        clean_threshold,
    )
    output["clean_probability_threshold"] = clean_threshold

    return output


def create_feature_importance(
    model,
    feature_columns,
):
    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    )

    return (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# PER-RECORDING PREDICTION GRAPH
# ============================================================

def create_prediction_plot(
    df,
    prediction_df,
    csv_path,
    transition_ranges,
    output_dir,
    clean_threshold,
    split,
):
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=df["time"],
            y=df[STROKE_AXIS],
            mode="lines",
            name=STROKE_AXIS,
        )
    )

    # Actual / manually labeled transitions.
    for start_time, end_time in transition_ranges:
        figure.add_vrect(
            x0=start_time,
            x1=end_time,
            fillcolor="red",
            opacity=0.12,
            line_width=1,
            line_color="red",
            layer="below",
        )

    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=10,
                color="red",
                symbol="square",
            ),
            name="Actual transition",
        )
    )

    # Random Forest predictions.
    for _, row in prediction_df.iterrows():
        if row["prediction"] == 0:
            figure.add_vrect(
                x0=row["start_time"],
                x1=row["end_time"],
                fillcolor="blue",
                opacity=0.20,
                line_width=0,
                layer="below",
            )

    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=10,
                color="blue",
                symbol="square",
            ),
            name=(
                "RF predicted transition "
                f"(clean threshold={clean_threshold:.2f})"
            ),
        )
    )

    figure.update_layout(
        title=(
            f"Random Forest transition classification: {csv_path.name}"
            f"<br><sup>Split = {split}; clean probability threshold = "
            f"{clean_threshold:.2f}</sup>"
        ),
        xaxis_title="Time (seconds)",
        yaxis_title=STROKE_AXIS,
        template="plotly_white",
        height=650,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.07,
            x=0,
        ),
    )

    output_file = (
        output_dir
        / f"{csv_path.stem}_{split}_transition_predictions.html"
    )

    figure.write_html(
        output_file,
        include_plotlyjs=True,
        full_html=True,
    )

    return output_file


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def print_class_distribution(dataframe, name):
    print()
    print(name)
    print("-" * len(name))
    print(f"Total windows: {len(dataframe)}")
    print(f"Transition (0): {(dataframe['label'] == 0).sum()}")
    print(f"Clean freestyle (1): {(dataframe['label'] == 1).sum()}")


def print_recording_distribution(all_features):
    print()
    print("=" * 90)
    print("WINDOWS BY RECORDING")
    print("=" * 90)

    summary = (
        all_features
        .groupby(["split", "participant", "recording", "label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "transition", 1: "clean"})
        .reset_index()
    )

    for column in ["transition", "clean"]:
        if column not in summary.columns:
            summary[column] = 0

    print(
        summary[
            [
                "split",
                "participant",
                "recording",
                "transition",
                "clean",
            ]
        ].to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train one shallow Random Forest on multiple freestyle "
            "IMU recordings."
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
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CHECK THE FIVE CONFIGURED FILES EXIST
    # --------------------------------------------------------

    recording_paths = {}

    for relative_path in FILE_CONFIG:
        csv_path = args.data_dir / relative_path

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Configured recording not found:\n{csv_path}"
            )

        recording_paths[relative_path] = csv_path

    # --------------------------------------------------------
    # IF LABELS ARE MISSING, MAKE LABELING PLOTS AND STOP
    # --------------------------------------------------------

    missing_labels = [
        relative_path
        for relative_path, config in FILE_CONFIG.items()
        if config["transition_ranges"] is None
    ]

    if missing_labels:
        labeling_dir = args.output_dir / "labeling_plots"
        labeling_dir.mkdir(parents=True, exist_ok=True)

        print()
        print("=" * 78)
        print("TRANSITION LABELS STILL NEEDED")
        print("=" * 78)
        print()
        print(
            "The script will create GYRO_1 HTML plots for the files "
            "that still have transition_ranges=None."
        )
        print(
            "Open each plot, record the transition start/end times, "
            "enter them in FILE_CONFIG, and run the script again."
        )

        for relative_path in missing_labels:
            csv_path = recording_paths[relative_path]

            df = load_recording(csv_path)
            df, fs = add_time_axis(df)
            df, full_duration = limit_recording_time(df)

            output_file = create_labeling_plot(
                df,
                csv_path,
                labeling_dir,
            )

            print()
            print(f"{relative_path}")
            print(f"  sampling frequency: {fs:.2f} Hz")
            print(f"  full duration: {full_duration:.2f} s")
            print(
                f"  data used/displayed: "
                f"0-{df['time'].iloc[-1]:.2f} s"
            )
            print(f"  labeling plot: {output_file}")

        print()
        print("Example of what to enter:")
        print()
        print('"0/Freestyle_1527873200322.csv": {')
        print('    "split": "train",')
        print('    "transition_ranges": [')
        print('        (0.0, 10.5),')
        print('        (61.0, 67.0),')
        print('        (115.0, 124.0),')
        print('    ],')
        print('},')
        print()
        print("Training has NOT been run yet because the missing files need ground-truth labels.")
        return

    # --------------------------------------------------------
    # PROCESS ALL FIVE RECORDINGS
    # --------------------------------------------------------

    all_feature_frames = []
    processed_recordings = {}

    for relative_path, config in FILE_CONFIG.items():
        csv_path = recording_paths[relative_path]
        split = config["split"]
        transition_ranges = config["transition_ranges"]
        participant = Path(relative_path).parent.name

        print()
        print("=" * 90)
        print(f"PROCESSING: {relative_path}")
        print(f"SPLIT: {split.upper()}")
        print("=" * 90)

        df = load_recording(csv_path)
        df, fs = add_time_axis(df)
        df, full_duration = limit_recording_time(df)
        df = add_magnitudes(df)
        df = add_manual_labels(
            df,
            transition_ranges,
        )

        print(f"Sampling frequency: {fs:.2f} Hz")
        print(f"Full recording duration: {full_duration:.2f} s")
        print(
            f"Using data from 0 to "
            f"{df['time'].iloc[-1]:.2f} s "
            f"(limit = {MAX_RECORDING_SECONDS:.1f} s)"
        )

        feature_df = create_windows(
            df,
            fs,
            recording=csv_path.name,
            participant=participant,
            split=split,
        )

        print(f"Created windows: {len(feature_df)}")
        print(
            "Transition windows: "
            f"{(feature_df['label'] == 0).sum()}"
        )
        print(
            "Clean windows: "
            f"{(feature_df['label'] == 1).sum()}"
        )

        all_feature_frames.append(feature_df)

        processed_recordings[relative_path] = {
            "df": df,
            "fs": fs,
            "csv_path": csv_path,
            "split": split,
            "transition_ranges": transition_ranges,
        }

    all_features = pd.concat(
        all_feature_frames,
        ignore_index=True,
    )

    all_features_output = (
        args.output_dir
        / "all_window_features.csv"
    )

    all_features.to_csv(
        all_features_output,
        index=False,
    )

    # --------------------------------------------------------
    # SPLIT BY WHOLE RECORDING
    # --------------------------------------------------------

    train_df = all_features[
        all_features["split"] == "train"
    ].copy()

    validation_df = all_features[
        all_features["split"] == "validation"
    ].copy()

    test_df = all_features[
        all_features["split"] == "test"
    ].copy()

    print_recording_distribution(all_features)

    print()
    print("=" * 50)
    print("COMBINED CLASS DISTRIBUTION")
    print("=" * 50)

    print_class_distribution(train_df, "TRAINING")
    print_class_distribution(validation_df, "VALIDATION")
    print_class_distribution(test_df, "TESTING")

    for name, dataframe in [
        ("Training", train_df),
        ("Validation", validation_df),
        ("Testing", test_df),
    ]:
        if len(dataframe) == 0:
            raise ValueError(f"{name} split is empty.")

        if dataframe["label"].nunique() < 2:
            raise ValueError(
                f"{name} split contains only one class. "
                "Check its transition labels."
            )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    feature_columns = get_feature_columns(all_features)

    print()
    print(
        "Number of Random Forest features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # TRAIN ONE RANDOM FOREST ON ALL TRAIN RECORDINGS
    # --------------------------------------------------------

    model = train_classifier(
        train_df,
        feature_columns,
    )

    # --------------------------------------------------------
    # CHOOSE THRESHOLD USING VALIDATION RECORDING(S) ONLY
    # --------------------------------------------------------

    threshold_results = test_probability_thresholds(
        model,
        validation_df,
        feature_columns,
    )

    threshold_output = (
        args.output_dir
        / "threshold_results.csv"
    )

    threshold_results.to_csv(
        threshold_output,
        index=False,
    )

    clean_threshold = choose_probability_threshold(
        threshold_results
    )

    # --------------------------------------------------------
    # VALIDATION AND FINAL TEST
    # --------------------------------------------------------

    evaluate_model(
        model,
        validation_df,
        feature_columns,
        "VALIDATION RESULTS",
        clean_threshold,
    )

    evaluate_model(
        model,
        test_df,
        feature_columns,
        "FINAL TEST RESULTS",
        clean_threshold,
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance_df = create_feature_importance(
        model,
        feature_columns,
    )

    importance_output = (
        args.output_dir
        / "feature_importance.csv"
    )

    importance_df.to_csv(
        importance_output,
        index=False,
    )

    print()
    print("=" * 50)
    print("MOST IMPORTANT FEATURES")
    print("=" * 50)
    print()
    print(
        importance_df.head(15).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # PREDICT + PLOT EACH RECORDING
    # --------------------------------------------------------

    prediction_dir = args.output_dir / "recording_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    prediction_frames = []
    html_files = []

    for relative_path, info in processed_recordings.items():
        csv_path = info["csv_path"]
        split = info["split"]

        recording_features = all_features[
            all_features["recording"] == csv_path.name
        ].copy()

        prediction_df = predict_all_windows(
            model,
            recording_features,
            feature_columns,
            clean_threshold,
        )

        prediction_frames.append(prediction_df)

        prediction_output = (
            prediction_dir
            / f"{csv_path.stem}_{split}_predictions.csv"
        )

        prediction_df.to_csv(
            prediction_output,
            index=False,
        )

        html_file = create_prediction_plot(
            info["df"],
            prediction_df,
            csv_path,
            info["transition_ranges"],
            prediction_dir,
            clean_threshold,
            split,
        )

        html_files.append(html_file)

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    all_predictions_output = (
        args.output_dir
        / "all_window_predictions.csv"
    )

    all_predictions.to_csv(
        all_predictions_output,
        index=False,
    )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINISHED")
    print("=" * 70)
    print()
    print(f"All window features:\n{all_features_output}")
    print()
    print(f"All window predictions:\n{all_predictions_output}")
    print()
    print(f"Feature importance:\n{importance_output}")
    print()
    print(f"Threshold comparison:\n{threshold_output}")
    print()
    print(
        "Selected clean-probability threshold:\n"
        f"{clean_threshold:.2f}"
    )
    print()
    print("Per-recording HTML graphs:")

    for html_file in html_files:
        print(f"  {html_file}")


if __name__ == "__main__":
    main()

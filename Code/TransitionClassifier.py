#!/usr/bin/env python3
"""
Classify sections of a freestyle IMU recording as:

    1 = clean freestyle swimming
    0 = transition / turn / noise

Uses a shallow Random Forest classifier.

The recording is divided into overlapping time windows.
Statistical and frequency-domain features are extracted from
each window and used by the classifier.

This first version is intended for ONE recording.
Later, training/testing should be separated by recording
and eventually by swimmer.
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
    / "transition_classifier"
)

DEFAULT_FILE_NAME = "Freestyle_1526810816300.csv"


# ============================================================
# SENSOR COLUMNS
# ============================================================

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


# ============================================================
# MANUAL TRANSITION LABELS
# ============================================================

# These are the transition / turn / non-clean-swimming regions
# you identified manually from the graph.

TRANSITION_RANGES = [
    (0.0, 12.1),
    (60.9, 66.9),
    (110.3, 124.2),
]


# ============================================================
# TRAIN / VALIDATION / TEST TIME BLOCKS
# ============================================================

# IMPORTANT:
#
# We intentionally leave gaps between these blocks.
#
# Since the windows overlap, putting train and validation
# directly beside each other could cause nearly identical
# raw sensor data to appear in both datasets.

TRAIN_RANGES = [
    (0.0, 45.0),
]

VALIDATION_RANGES = [
    (50.0, 85.0),
]

TEST_RANGES = [
    (90.0, 124.2),
]


# ============================================================
# WINDOW SETTINGS
# ============================================================

# Your detected arm-cycle period was around 2 seconds,
# so this is a reasonable starting window length.
WINDOW_SECONDS = 2.0

# 50% overlap
WINDOW_OVERLAP = 0.5


# ============================================================
# RANDOM FOREST SETTINGS
# ============================================================

NUMBER_OF_TREES = 100

# Shallow trees to reduce overfitting
MAX_TREE_DEPTH = 4

MIN_SAMPLES_LEAF = 3


# ============================================================
# PEAK FEATURE SETTINGS
# ============================================================

STROKE_AXIS = "GYRO_1"

PEAK_PROMINENCE_FACTOR = 0.5

MIN_PEAK_DISTANCE_SECONDS = 0.5


# ============================================================
# FIND RECORDING
# ============================================================

def find_recording(data_dir: Path, file_name: str) -> Path:
    """Find recording inside participant directories."""

    matches = sorted(
        data_dir.rglob(file_name)
    )

    if not matches:
        raise FileNotFoundError(
            f"Recording not found: {file_name} in {data_dir}"
        )

    return matches[0]


# ============================================================
# RMS
# ============================================================

def calculate_rms(signal):
    """Calculate root mean square."""

    signal = np.asarray(
        signal,
        dtype=float
    )

    return np.sqrt(
        np.mean(signal ** 2)
    )


# ============================================================
# DOMINANT FREQUENCY
# ============================================================

def dominant_frequency(
    signal,
    fs,
    minimum_frequency=0.2,
    maximum_frequency=2.0
):
    """
    Return the strongest frequency inside a reasonable
    swimming-frequency range.
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    if len(signal) < 4:
        return 0.0

    # Remove DC component
    signal = signal - np.mean(signal)

    frequencies, power = periodogram(
        signal,
        fs=fs
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


# ============================================================
# PERIODICITY STRENGTH
# ============================================================

def periodicity_strength(
    signal,
    fs,
    minimum_frequency=0.2,
    maximum_frequency=2.0
):
    """
    Measure how much of the frequency-band power is
    concentrated in the strongest frequency.

    Higher value = more strongly periodic.
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    if len(signal) < 4:
        return 0.0

    signal = signal - np.mean(signal)

    frequencies, power = periodogram(
        signal,
        fs=fs
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
# LOAD DATA
# ============================================================

def load_recording(csv_path):
    """Load and clean one IMU recording."""

    df = pd.read_csv(csv_path)

    df = df.drop(
        columns=[
            column
            for column in df.columns
            if column.startswith("Unnamed")
        ],
        errors="ignore"
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
            f"Missing required columns: {missing}"
        )

    for column in required_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=required_columns
    ).reset_index(
        drop=True
    )

    if len(df) == 0:
        raise ValueError(
            "No usable samples were found."
        )

    return df


# ============================================================
# TIME AXIS
# ============================================================

def add_time_axis(df):
    """Calculate elapsed time and sampling frequency."""

    timestamps = df["timestamp"]

    df["time"] = (
        timestamps - timestamps.iloc[0]
    ) / 1_000_000_000

    time_differences = np.diff(
        df["time"]
    )

    time_differences = time_differences[
        np.isfinite(time_differences)
        & (time_differences > 0)
    ]

    if len(time_differences) == 0:
        raise ValueError(
            "Could not determine sampling frequency."
        )

    fs = 1 / np.median(
        time_differences
    )

    return df, fs


# ============================================================
# SENSOR MAGNITUDES
# ============================================================

def add_magnitudes(df):
    """Calculate accelerometer and gyroscope magnitudes."""

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


# ============================================================
# MANUAL LABELS
# ============================================================

def add_manual_labels(df):
    """
    Label each raw sample:

    1 = clean freestyle
    0 = transition / turn / noise
    """

    df["clean_swimming"] = 1

    for start_time, end_time in TRANSITION_RANGES:

        transition = (
            (df["time"] >= start_time)
            & (df["time"] <= end_time)
        )

        df.loc[
            transition,
            "clean_swimming"
        ] = 0

    return df


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(window, fs):
    """
    Calculate statistical and frequency features
    from one window.
    """

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

    # --------------------------------------------------------
    # BASIC STATISTICAL FEATURES
    # --------------------------------------------------------

    for column in signals:

        signal = window[
            column
        ].to_numpy()

        features[
            f"{column}_mean"
        ] = np.mean(signal)

        features[
            f"{column}_std"
        ] = np.std(signal)

        features[
            f"{column}_rms"
        ] = calculate_rms(signal)

        features[
            f"{column}_minimum"
        ] = np.min(signal)

        features[
            f"{column}_maximum"
        ] = np.max(signal)

        features[
            f"{column}_range"
        ] = (
            np.max(signal)
            - np.min(signal)
        )

    # --------------------------------------------------------
    # FREQUENCY FEATURES
    # --------------------------------------------------------

    stroke_signal = window[
        STROKE_AXIS
    ].to_numpy()

    features[
        "dominant_frequency"
    ] = dominant_frequency(
        stroke_signal,
        fs
    )

    features[
        "periodicity_strength"
    ] = periodicity_strength(
        stroke_signal,
        fs
    )

    # --------------------------------------------------------
    # PEAK / TROUGH FEATURES
    # --------------------------------------------------------

    prominence = (
        np.std(stroke_signal)
        * PEAK_PROMINENCE_FACTOR
    )

    minimum_distance = int(
        MIN_PEAK_DISTANCE_SECONDS
        * fs
    )

    minimum_distance = max(
        minimum_distance,
        1
    )

    positive_peaks, positive_properties = find_peaks(
        stroke_signal,
        distance=minimum_distance,
        prominence=prominence
    )

    negative_troughs, negative_properties = find_peaks(
        -stroke_signal,
        distance=minimum_distance,
        prominence=prominence
    )

    features[
        "positive_peak_count"
    ] = len(
        positive_peaks
    )

    features[
        "negative_trough_count"
    ] = len(
        negative_troughs
    )

    # Mean prominence can help distinguish clean periodic
    # movement from irregular transition movement.

    if len(positive_peaks) > 0:

        features[
            "mean_positive_prominence"
        ] = np.mean(
            positive_properties[
                "prominences"
            ]
        )

    else:

        features[
            "mean_positive_prominence"
        ] = 0.0

    if len(negative_troughs) > 0:

        features[
            "mean_negative_prominence"
        ] = np.mean(
            negative_properties[
                "prominences"
            ]
        )

    else:

        features[
            "mean_negative_prominence"
        ] = 0.0

    return features


# ============================================================
# CREATE WINDOWS
# ============================================================

def create_windows(df, fs):
    """
    Divide recording into overlapping windows.
    """

    samples_per_window = int(
        round(
            WINDOW_SECONDS
            * fs
        )
    )

    step = int(
        round(
            samples_per_window
            * (1 - WINDOW_OVERLAP)
        )
    )

    step = max(
        step,
        1
    )

    feature_rows = []

    for start_index in range(
        0,
        len(df) - samples_per_window + 1,
        step
    ):

        end_index = (
            start_index
            + samples_per_window
        )

        window = df.iloc[
            start_index:end_index
        ]

        features = extract_features(
            window,
            fs
        )

        start_time = (
            window["time"].iloc[0]
        )

        end_time = (
            window["time"].iloc[-1]
        )

        center_time = (
            start_time
            + end_time
        ) / 2

        clean_fraction = (
            window[
                "clean_swimming"
            ].mean()
        )

        # Majority rule
        label = int(
            clean_fraction >= 0.9
        )

        features[
            "start_time"
        ] = start_time

        features[
            "end_time"
        ] = end_time

        features[
            "center_time"
        ] = center_time

        features[
            "clean_fraction"
        ] = clean_fraction

        features[
            "label"
        ] = label

        feature_rows.append(
            features
        )

    return pd.DataFrame(
        feature_rows
    )


# ============================================================
# TIME-RANGE SPLITTING
# ============================================================

def windows_inside_ranges(
    feature_df,
    ranges
):
    """
    Select only windows that are completely contained
    inside one of the supplied time ranges.

    Full containment prevents overlapping windows from
    crossing train/validation/test boundaries.
    """

    mask = pd.Series(
        False,
        index=feature_df.index
    )

    for start_time, end_time in ranges:

        inside = (
            (feature_df["start_time"] >= start_time)
            & (feature_df["end_time"] <= end_time)
        )

        mask = mask | inside

    return feature_df.loc[
        mask
    ].copy()


def split_dataset(feature_df):
    """
    Split using separated time blocks instead of a simple
    chronological 60/20/20 split.
    """

    train_df = windows_inside_ranges(
        feature_df,
        TRAIN_RANGES
    )

    validation_df = windows_inside_ranges(
        feature_df,
        VALIDATION_RANGES
    )

    test_df = windows_inside_ranges(
        feature_df,
        TEST_RANGES
    )

    return (
        train_df,
        validation_df,
        test_df
    )


# ============================================================
# PRINT CLASS DISTRIBUTION
# ============================================================

def print_class_distribution(
    dataframe,
    name
):

    print()
    print(name)
    print("-" * len(name))

    print(
        f"Total windows: "
        f"{len(dataframe)}"
    )

    print(
        f"Transition (0): "
        f"{(dataframe['label'] == 0).sum()}"
    )

    print(
        f"Clean freestyle (1): "
        f"{(dataframe['label'] == 1).sum()}"
    )


# ============================================================
# FEATURE COLUMNS
# ============================================================

def get_feature_columns(df):

    ignore_columns = [
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
# TRAIN RANDOM FOREST
# ============================================================

def train_classifier(
    train_df,
    feature_columns
):

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        "label"
    ]

    if y_train.nunique() < 2:

        raise ValueError(
            "\nTraining dataset contains only one class.\n"
            "Training requires both transition and clean "
            "freestyle windows."
        )

    model = RandomForestClassifier(
        n_estimators=NUMBER_OF_TREES,
        max_depth=MAX_TREE_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    dataframe,
    feature_columns,
    name
):

    if len(dataframe) == 0:

        print(
            f"\n{name}: no windows available."
        )

        return

    X = dataframe[
        feature_columns
    ]

    y = dataframe[
        "label"
    ]

    predictions = model.predict(
        X
    )

    print()
    print("=" * 50)
    print(name)
    print("=" * 50)

    print()
    print("Actual class counts:")

    print(
        f"Transition (0): "
        f"{(y == 0).sum()}"
    )

    print(
        f"Clean freestyle (1): "
        f"{(y == 1).sum()}"
    )

    print()
    print("Predicted class counts:")

    print(
        f"Transition (0): "
        f"{(predictions == 0).sum()}"
    )

    print(
        f"Clean freestyle (1): "
        f"{(predictions == 1).sum()}"
    )

    print()

    print(
        f"Accuracy: "
        f"{accuracy_score(y, predictions):.3f}"
    )

    print()
    print("Confusion matrix:")
    print()

    print(
        confusion_matrix(
            y,
            predictions,
            labels=[0, 1]
        )
    )

    print()
    print("Classification report:")
    print()

    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=[
                "Transition",
                "Clean freestyle"
            ],
            zero_division=0
        )
    )


# ============================================================
# PREDICT ALL WINDOWS
# ============================================================

def predict_all_windows(
    model,
    feature_df,
    feature_columns
):

    output = feature_df.copy()

    X = output[
        feature_columns
    ]

    output[
        "prediction"
    ] = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )

    clean_index = list(
        model.classes_
    ).index(1)

    output[
        "clean_probability"
    ] = probabilities[
        :,
        clean_index
    ]

    return output


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def create_feature_importance(
    model,
    feature_columns
):

    importance_df = pd.DataFrame(
        {
            "feature":
                feature_columns,

            "importance":
                model.feature_importances_
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return importance_df


# ============================================================
# PREDICTION GRAPH
# ============================================================

def create_prediction_plot(
    df,
    prediction_df,
    csv_path,
    output_dir
):

    figure = go.Figure()

    # --------------------------------------------------------
    # GYROSCOPE SIGNAL
    # --------------------------------------------------------

    figure.add_trace(
        go.Scatter(
            x=df["time"],
            y=df[STROKE_AXIS],
            mode="lines",
            name=STROKE_AXIS
        )
    )

    # --------------------------------------------------------
    # ACTUAL TRANSITION REGIONS
    #
    # These are the transition periods YOU manually entered.
    # --------------------------------------------------------

    for start_time, end_time in TRANSITION_RANGES:

        figure.add_vrect(
            x0=start_time,
            x1=end_time,

            fillcolor="red",
            opacity=0.12,

            line_width=1,
            line_color="red",

            layer="below"
        )

    # Dummy trace so "Actual transition" appears in legend
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=10,
                color="red",
                symbol="square"
            ),
            name="Actual transition"
        )
    )

    # --------------------------------------------------------
    # RANDOM FOREST PREDICTED TRANSITIONS
    #
    # prediction == 0 means transition
    # --------------------------------------------------------

    for _, row in prediction_df.iterrows():

        if row["prediction"] == 0:

            figure.add_vrect(
                x0=row["start_time"],
                x1=row["end_time"],

                fillcolor="blue",
                opacity=0.20,

                line_width=0,

                layer="below"
            )

    # Dummy trace so predicted transitions appear in legend
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=10,
                color="blue",
                symbol="square"
            ),
            name="RF predicted transition"
        )
    )

    # --------------------------------------------------------
    # FORMAT
    # --------------------------------------------------------

    figure.update_layout(

        title=(
            f"Random Forest transition classification: "
            f"{csv_path.name}"
        ),

        xaxis_title="Time (seconds)",

        yaxis_title=STROKE_AXIS,

        template="plotly_white",

        height=650,

        hovermode="x unified",

        legend=dict(
            orientation="h",
            y=1.05,
            x=0
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = (
        output_dir
        / f"{csv_path.stem}_transition_predictions.html"
    )

    figure.write_html(
        output_file,
        include_plotlyjs=True,
        full_html=True
    )

    return output_file
# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train a shallow Random Forest classifier "
            "to identify clean freestyle vs transitions."
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
        default=DEFAULT_FILE_NAME
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # FIND FILE
    # --------------------------------------------------------

    if not args.data_dir.exists():

        raise FileNotFoundError(
            f"Data directory not found: "
            f"{args.data_dir}"
        )

    csv_path = find_recording(
        args.data_dir,
        args.file
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("Loading:")
    print(csv_path)

    # --------------------------------------------------------
    # LOAD RECORDING
    # --------------------------------------------------------

    df = load_recording(
        csv_path
    )

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    df, fs = add_time_axis(
        df
    )

    print()
    print(
        f"Sampling frequency: "
        f"{fs:.2f} Hz"
    )

    print(
        f"Recording duration: "
        f"{df['time'].iloc[-1]:.2f} s"
    )

    # --------------------------------------------------------
    # MAGNITUDES
    # --------------------------------------------------------

    df = add_magnitudes(
        df
    )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    df = add_manual_labels(
        df
    )

    # --------------------------------------------------------
    # CREATE WINDOWS
    # --------------------------------------------------------

    feature_df = create_windows(
        df,
        fs
    )

    print()
    print(
        f"Created {len(feature_df)} total windows."
    )

    print(
        f"Total clean windows: "
        f"{(feature_df['label'] == 1).sum()}"
    )

    print(
        f"Total transition windows: "
        f"{(feature_df['label'] == 0).sum()}"
    )

    # --------------------------------------------------------
    # SAVE FEATURES
    # --------------------------------------------------------

    feature_output = (
        args.output_dir
        / "window_features.csv"
    )

    feature_df.to_csv(
        feature_output,
        index=False
    )

    # --------------------------------------------------------
    # SPLIT DATA
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df
    ) = split_dataset(
        feature_df
    )

    print()
    print("=" * 50)
    print("CLASS DISTRIBUTION")
    print("=" * 50)

    print_class_distribution(
        train_df,
        "TRAINING"
    )

    print_class_distribution(
        validation_df,
        "VALIDATION"
    )

    print_class_distribution(
        test_df,
        "TESTING"
    )

    # --------------------------------------------------------
    # CHECK CLASS PRESENCE
    # --------------------------------------------------------

    if train_df["label"].nunique() < 2:

        raise ValueError(
            "Training split does not contain both classes."
        )

    if validation_df["label"].nunique() < 2:

        print(
            "\nWARNING: Validation contains only one class."
        )

    if test_df["label"].nunique() < 2:

        print(
            "\nWARNING: Testing contains only one class."
        )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    feature_columns = get_feature_columns(
        feature_df
    )

    print()
    print(
        f"Number of Random Forest features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model = train_classifier(
        train_df,
        feature_columns
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    evaluate_model(
        model,
        validation_df,
        feature_columns,
        "VALIDATION RESULTS"
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    evaluate_model(
        model,
        test_df,
        feature_columns,
        "TEST RESULTS"
    )

    # --------------------------------------------------------
    # PREDICT FULL RECORDING
    # --------------------------------------------------------

    prediction_df = predict_all_windows(
        model,
        feature_df,
        feature_columns
    )

    prediction_output = (
        args.output_dir
        / "window_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_output,
        index=False
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance_df = create_feature_importance(
        model,
        feature_columns
    )

    importance_output = (
        args.output_dir
        / "feature_importance.csv"
    )

    importance_df.to_csv(
        importance_output,
        index=False
    )

    print()
    print("=" * 50)
    print("MOST IMPORTANT FEATURES")
    print("=" * 50)
    print()

    print(
        importance_df.head(
            15
        ).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    html_file = create_prediction_plot(
        df,
        prediction_df,
        csv_path,
        args.output_dir
    )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print()
    print("=" * 50)
    print("FINISHED")
    print("=" * 50)

    print()
    print(
        f"Window features:\n"
        f"{feature_output}"
    )

    print()
    print(
        f"Predictions:\n"
        f"{prediction_output}"
    )

    print()
    print(
        f"Feature importance:\n"
        f"{importance_output}"
    )

    print()
    print(
        f"HTML graph:\n"
        f"{html_file}"
    )


if __name__ == "__main__":
    main()
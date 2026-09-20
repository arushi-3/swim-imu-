#!/usr/bin/env python3
"""Train a shallow random forest to detect freestyle stroke intervals.

The Brunner CSVs provide frame labels, not hand-labelled stroke metrics. This
script therefore learns clean background/stroke states, excludes transition
label 5 and its guard band, then measures magnitude, frequency, and RMS on
the detected stroke intervals.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "Brunner Data" / "swimming-recognition-lap-counting-master" / "data" / "processed_30hz_relabeled"
SENSOR_COLUMNS = [
	"ACC_0", "ACC_1", "ACC_2", "GYRO_0", "GYRO_1", "GYRO_2",
	"MAG_0", "MAG_1", "MAG_2",
]
SAMPLE_RATE_HZ = 30.0
WINDOW_SAMPLES = 15
TRANSITION_LABEL = 5


def sensor_features(frame: pd.DataFrame) -> pd.DataFrame:
	"""Create causal-safe rolling features from raw sensor channels only."""
	sensors = frame[SENSOR_COLUMNS].apply(pd.to_numeric, errors="coerce").interpolate().bfill().ffill()
	features = {}
	for column in SENSOR_COLUMNS:
		values = sensors[column]
		features[f"{column}_mean"] = values.rolling(WINDOW_SAMPLES, min_periods=1).mean()
		features[f"{column}_std"] = values.rolling(WINDOW_SAMPLES, min_periods=2).std().fillna(0.0)
		features[f"{column}_range"] = values.rolling(WINDOW_SAMPLES, min_periods=1).max() - values.rolling(WINDOW_SAMPLES, min_periods=1).min()
	for prefix in ("ACC", "GYRO", "MAG"):
		magnitude = np.sqrt((sensors[[f"{prefix}_{axis}" for axis in range(3)]] ** 2).sum(axis=1))
		features[f"{prefix}_magnitude"] = magnitude
		features[f"{prefix}_magnitude_mean"] = magnitude.rolling(WINDOW_SAMPLES, min_periods=1).mean()
		features[f"{prefix}_magnitude_std"] = magnitude.rolling(WINDOW_SAMPLES, min_periods=2).std().fillna(0.0)
	return pd.DataFrame(features, index=frame.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def clean_training_mask(labels: pd.Series, guard_samples: int) -> pd.Series:
	"""Keep labels 0/1, excluding transition frames and nearby contamination."""
	transition = labels.eq(TRANSITION_LABEL)
	for offset in range(-guard_samples, guard_samples + 1):
		transition |= labels.shift(offset).eq(TRANSITION_LABEL)
	return labels.isin([0, 1]) & ~transition


def load_dataset(data_dir: Path, guard_samples: int) -> tuple[pd.DataFrame, pd.Series, list[Path]]:
	feature_frames = []
	target_frames = []
	source_files = []
	for path in sorted(data_dir.rglob("*.csv")):
		frame = pd.read_csv(path).drop(columns=["Unnamed: 0"], errors="ignore")
		if not set(SENSOR_COLUMNS + ["label"]).issubset(frame.columns):
			continue
		labels = pd.to_numeric(frame["label"], errors="coerce")
		mask = clean_training_mask(labels, guard_samples)
		feature_frames.append(sensor_features(frame).loc[mask])
		target_frames.append(labels.eq(1).astype(int).loc[mask])
		source_files.extend([path] * int(mask.sum()))
	if not feature_frames:
		raise ValueError(f"No usable Brunner CSV files found in {data_dir}")
	return pd.concat(feature_frames), pd.concat(target_frames), source_files


def interval_metrics(frame: pd.DataFrame, start: int, end: int) -> dict[str, float]:
	"""Measure magnitude, dominant frequency, and RMS for one stroke interval."""
	signal = frame.loc[start:end, ["ACC_0", "ACC_1", "ACC_2"]].apply(pd.to_numeric, errors="coerce").to_numpy()
	magnitude = np.sqrt((signal ** 2).sum(axis=1))
	centered = magnitude - magnitude.mean()
	spectrum = np.abs(np.fft.rfft(centered))
	frequencies = np.fft.rfftfreq(len(centered), d=1.0 / SAMPLE_RATE_HZ)
	dominant_index = int(np.argmax(spectrum[1:]) + 1) if len(spectrum) > 1 else 0
	return {
		"start_sample": float(start),
		"end_sample": float(end),
		"duration_seconds": (end - start + 1) / SAMPLE_RATE_HZ,
		"magnitude_peak": float(magnitude.max()),
		"magnitude_rms": float(np.sqrt(np.mean(magnitude ** 2))),
		"frequency_hz": float(frequencies[dominant_index]),
		"rms_error_from_mean": float(np.sqrt(np.mean(centered ** 2))),
	}


def detect_intervals(predictions: np.ndarray) -> list[tuple[int, int]]:
	intervals = []
	start = None
	for index, prediction in enumerate(predictions):
		if prediction == 1 and start is None:
			start = index
		if start is not None and (prediction != 1 or index == len(predictions) - 1):
			end = index if prediction == 1 else index - 1
			if end >= start:
				intervals.append((start, end))
			start = None
	return intervals


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
	parser.add_argument("--guard-samples", type=int, default=15, help="Samples around transition label 5 to exclude.")
	parser.add_argument("--trees", type=int, default=100)
	parser.add_argument("--max-depth", type=int, default=6, help="Shallow tree depth.")
	args = parser.parse_args()

	features, targets, source_files = load_dataset(args.data_dir, args.guard_samples)
	unique_files = sorted(set(source_files))
	test_file = unique_files[-1]
	test_mask = np.array([path == test_file for path in source_files])
	train_mask = ~test_mask
	model = RandomForestClassifier(
		n_estimators=args.trees,
		max_depth=args.max_depth,
		min_samples_leaf=5,
		class_weight="balanced",
		random_state=42,
		n_jobs=-1,
	)
	model.fit(features.iloc[train_mask], targets.iloc[train_mask])
	predictions = model.predict(features.iloc[test_mask])
	print(f"Train rows: {train_mask.sum()} | Test file: {test_file.name} | Test rows: {test_mask.sum()}")
	print(classification_report(targets.iloc[test_mask], predictions, labels=[0, 1], target_names=["background", "stroke"], zero_division=0))
	print("Confusion matrix [background, stroke]:")
	print(confusion_matrix(targets.iloc[test_mask], predictions, labels=[0, 1]))
	print("Feature importance:")
	for name, importance in sorted(zip(features.columns, model.feature_importances_), key=lambda item: item[1], reverse=True)[:10]:
		print(f"  {name}: {importance:.4f}")

	test_frame = pd.read_csv(test_file).drop(columns=["Unnamed: 0"], errors="ignore")
	test_labels = pd.to_numeric(test_frame["label"], errors="coerce")
	test_mask = clean_training_mask(test_labels, args.guard_samples)
	test_predictions = model.predict(sensor_features(test_frame).loc[test_mask])
	kept_indices = np.flatnonzero(test_mask.to_numpy())
	predicted_intervals = detect_intervals(test_predictions)
	metrics = []
	for interval_number, (start_position, end_position) in enumerate(predicted_intervals, start=1):
		start_index = int(kept_indices[start_position])
		end_index = int(kept_indices[end_position])
		result = interval_metrics(test_frame, start_index, end_index)
		result["interval"] = interval_number
		metrics.append(result)
	output_file = REPO_ROOT / "Code" / "stroke_analysis_results.csv"
	pd.DataFrame(metrics).to_csv(output_file, index=False)
	print(f"Detected intervals: {len(metrics)}")
	print(f"Wrote interval metrics to {output_file}")


if __name__ == "__main__":
	main()

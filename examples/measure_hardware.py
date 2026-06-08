"""Collect a lightweight LeanTrain hardware measurement JSON."""

from pathlib import Path

from leantrain.hardware.measure import run_measurement_suite, save_measurement


if __name__ == "__main__":
    measurement = run_measurement_suite(include_bandwidth=False)
    save_measurement(measurement, Path("profiles/local-profile.json"))

"""Print the local LeanTrain hardware profile."""

from leantrain.hardware.probe import probe_hardware


if __name__ == "__main__":
    print(probe_hardware().summary())

from configparser import ConfigParser
from pathlib import Path


class Configure:
    """
    Reads ASIMOV's configuration from config.ini.

    By default, the configuration file is resolved relative to
    the ASIMOV package, independently of the current working directory.

    A custom configuration file can optionally be supplied.
    """

    DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.ini"

    def __init__(self, config_path: str | Path | None = None):
        self.config_object = ConfigParser()

        self.config_path = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else self.DEFAULT_CONFIG_PATH
        )

        self._load_config()

    def _load_config(self):
        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found at {self.config_path}"
            )

        self.config_object.read(self.config_path, encoding="utf-8")

        # Parse DETECTION values
        detection_info = self.config_object["DETECTION"]
        self.align_faces = detection_info.getboolean("align", fallback=False)
        self.perform_detection = detection_info.getboolean(
            "perform_detection", fallback=True
        )

        # Parse RECOGNITION values
        recognition_info = self.config_object["RECOGNITION"]
        self.perform_recognition = recognition_info.getboolean(
            "perform_recognition", fallback=True
        )
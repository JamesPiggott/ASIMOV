from configparser import ConfigParser
from pathlib import Path

class Configure:
    """Configure reads config.ini and with its API permits easy configuration at startup.

    Default config.ini:

        [DETECTION]
        perform_detection = True
        align = False
        draw_bounding_box = False

        [RECOGNITION]
        perform_recognition = True
    """

    def __init__(self, config_path: str = "../core/config.ini"):
        self.config_object = ConfigParser()
        self.config_path = Path(config_path)
        self._load_config()

    def _load_config(self):
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")

        self.config_object.read(self.config_path)

        # Parse DETECTION values
        detection_info = self.config_object["DETECTION"]
        self.align_faces = detection_info.getboolean("align", fallback=False)
        self.perform_detection = detection_info.getboolean("perform_detection", fallback=True)

        # Parse RECOGNITION values
        recognition_info = self.config_object["RECOGNITION"]
        self.perform_recognition = recognition_info.getboolean("perform_recognition", fallback=True)

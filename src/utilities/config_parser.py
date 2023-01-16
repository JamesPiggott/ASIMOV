from configparser import ConfigParser


class Configure:
    """Configure reads config.ini and with its API permits easy configuration at startup

        Default config.ini:

            [DETECTION]
            perform_detection = True
            align = False
            draw_bounding_box = False

            [RECOGNITION]
            perform_recognition = True
    """

    def __init__(self):
        config_object = ConfigParser()
        config_object.read("../config.ini")

        # Parse DETECTION values
        detection_info = config_object["DETECTION"]
        self.align_faces = bool(detection_info["align"])
        self.perform_detection = bool(detection_info["perform_detection"])

        # Parse RECOGNITION values
        recognition_info = config_object["RECOGNITION"]
        self.perform_recognition = bool(recognition_info["perform_recognition"])


from configparser import ConfigParser

config_object = ConfigParser()
config_object.read("../config.ini")

detection_info = config_object["DETECTION"]
align_faces = bool(detection_info["align"])
print(align_faces)
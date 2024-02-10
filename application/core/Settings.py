import pathlib

"""
Settings class defines the values of important variables within the application, NOT the API

Values include the locations of the models and model parameters

Settings is not to be confused with config.ini with which users can alter the API configuration

These values are provided as static variables.
"""

# Location of the face detection model
location_retina_face_model = str(pathlib.Path().resolve()) + "/../core/api/detection/retinaface/"

# Location of the face recognition model
location_arcface_model = str(pathlib.Path().resolve()) + "/../core/api/recognition/arcface/"

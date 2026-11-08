from pathlib import Path

"""
Settings defines internal ASIMOV configuration values.

These settings are independent of the current working directory
and should not be confused with the user-facing config.ini.
"""

# Directory containing this Settings.py file
CORE_DIRECTORY = Path(__file__).resolve().parent

# Locations of the TensorFlow SavedModels
location_retina_face_model = CORE_DIRECTORY / "api" / "detection" / "retinaface" / "mbv2_model"

location_arcface_model = CORE_DIRECTORY / "api" / "recognition" / "arcface" / "arcface"
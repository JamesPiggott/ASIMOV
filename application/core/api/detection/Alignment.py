import math
import numpy as np
from PIL import Image
from application.core.api.result.Face import Face


def find_euclidean_distance(source_representation: np.ndarray, test_representation: np.ndarray) -> float:
    """
    Calculate Euclidean distance between points.

    Args:
        source_representation (np.ndarray): Source representation coordinates.
        test_representation (np.ndarray): Test representation coordinates.

    Returns:
        float: Euclidean distance between the two representations.
    """
    euclidean_distance = np.linalg.norm(source_representation - test_representation)
    return euclidean_distance


def alignment_procedure(face: Face) -> np.ndarray:
    """
    Align a face on the vertical axis based on the coordinates of the eyes.

    Args:
        face (Face): Face class instance containing face landmarks and crop information.

    Returns:
        np.ndarray: Aligned face image.
    """
    img = face.get_crop()
    nose = face.get_landmarks().nose
    left_eye = face.get_landmarks().left_eye
    right_eye = face.get_landmarks().right_eye

    left_eye_coord = (left_eye.get_horizontal(), left_eye.get_vertical())
    right_eye_coord = (right_eye.get_horizontal(), right_eye.get_vertical())
    nose_coord = (nose.get_horizontal(), nose.get_vertical())

    # Calculate the center between coordinates (x and y) between the eyes
    center_eyes = (int((left_eye_coord[0] + right_eye_coord[0]) / 2), int((left_eye_coord[1] + right_eye_coord[1]) / 2))

    # Find rotation direction
    if left_eye_coord[1] > right_eye_coord[1]:
        point_3rd = (right_eye_coord[0], left_eye_coord[1])
        direction = -1  # rotate clockwise
    else:
        point_3rd = (left_eye_coord[0], right_eye_coord[1])
        direction = 1  # rotate anti-clockwise

    # Find length of triangle edges
    a = find_euclidean_distance(np.array(left_eye_coord), np.array(point_3rd))
    b = find_euclidean_distance(np.array(right_eye_coord), np.array(point_3rd))
    c = find_euclidean_distance(np.array(right_eye_coord), np.array(left_eye_coord))

    # Apply cosine rule
    if b != 0 and c != 0:
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_a))

        # Rotate base image
        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

        if center_eyes[1] > nose_coord[1]:
            img = Image.fromarray(img)
            img = np.array(img.rotate(180))

    return img

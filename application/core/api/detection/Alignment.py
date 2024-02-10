import math

import numpy as np
from PIL import Image


def find_euclidean_distance(source_representation, test_representation):
    """
    Calculate Euclidean distance between points
    :param source_representation:
    :param test_representation:
    :return:
    """
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def alignment_procedure(face):
    """
    This function aligns a face on the vertical axis based on the coordinates of the eyes.

    Original implementation found @ https://github.com/serengil/deepface/blob/master/deepface/commons/functions.py
    :param face: Face class
    :return:
    """

    # Left eye is actually for the person in the image their right eye
    img = face.get_crop()
    nose = face.get_landmarks().nose

    left_eye_x, left_eye_y = face.get_landmarks().left_eye.get_horizontal(), face.get_landmarks().left_eye.get_vertical()
    right_eye_x, right_eye_y = face.get_landmarks().right_eye.get_horizontal(), face.get_landmarks().right_eye.get_vertical()

    left_eye = (left_eye_x, left_eye_y)
    right_eye = (right_eye_x, right_eye_y)
    nose = (nose.get_horizontal(), nose.get_vertical())

    # Calculate the center between coordinates (x and y) between the eyes
    center_eyes = (int((left_eye_x + right_eye_x) / 2), int((left_eye_y + right_eye_y) / 2))

    # find rotation direction
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate clockwise
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate anti-clockwise

    # find length of triangle edges
    a = find_euclidean_distance(np.array(left_eye), np.array(point_3rd))
    b = find_euclidean_distance(np.array(right_eye), np.array(point_3rd))
    c = find_euclidean_distance(np.array(right_eye), np.array(left_eye))

    # apply cosine rule
    if b != 0 and c != 0:

        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        cos_a = min(1.0, max(-1.0, cos_a))

        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # rotate base image
        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

        if center_eyes[1] > nose[1]:
            img = Image.fromarray(img)
            img = np.array(img.rotate(180))

    return img

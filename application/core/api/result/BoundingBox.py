import cv2
import numpy as np


class BoundingBox:
    """BoundingBox class represents the rectangle around the face detected using four coordinates."""

    def __init__(self):
        self.x1: int = None
        self.y1: int = None
        self.x2: int = None
        self.y2: int = None

    def draw_bounding_box(self, img: np.ndarray) -> np.ndarray:
        cv2.rectangle(img, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 2)
        return img


class Landmarks:
    """Landmarks class represents the coordinates of the eyes, nose, and both corners of the mouth."""

    def __init__(self):
        self.left_eye: Coordinate = None
        self.right_eye: Coordinate = None
        self.nose: Coordinate = None
        self.left_corner_mouth: Coordinate = None
        self.right_corner_mouth: Coordinate = None

    def draw_landmarks(self, img_raw: np.ndarray) -> np.ndarray:
        img_raw = cv2.circle(img_raw, (self.left_eye.x, self.left_eye.y), 1, (0, 255, 255), 2)
        img_raw = cv2.circle(img_raw, (self.right_eye.x, self.right_eye.y), 1, (255, 255, 0), 2)
        img_raw = cv2.circle(img_raw, (self.nose.x, self.nose.y), 1, (255, 0, 0), 2)
        img_raw = cv2.circle(img_raw, (self.left_corner_mouth.x, self.left_corner_mouth.y), 1, (0, 100, 255), 2)
        img_raw = cv2.circle(img_raw, (self.right_corner_mouth.x, self.right_corner_mouth.y), 1, (255, 0, 100), 2)
        return img_raw


class Probability:
    """Probability represents the certainty of the detection being a human face."""

    def __init__(self):
        self.probability: float = None
        self.as_text: str = None

    def draw_probability(self, img: np.ndarray, x1: int, y1: int) -> np.ndarray:
        cv2.putText(img, self.as_text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        return img


class Coordinate:
    """Coordinate represents the x and y cartesian coordinates of a point."""

    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y
        
    def get_horizontal(self):
        return self.x

    def get_vertical(self):
        return self.y

class Crop:
    """Crop represents a rectangle around the face scaled to the landmarks."""

    def __init__(self):
        self.top_right: Coordinate = None
        self.top_left: Coordinate = None
        self.bottom_right: Coordinate = None
        self.bottom_left: Coordinate = None
        self.crop: np.ndarray = None

    def get_image_crop(self, landmarks: Landmarks, img_raw: np.ndarray) -> np.ndarray:
        distance_between_eyes = landmarks.right_eye.x - landmarks.left_eye.x
        crop_distance = int(distance_between_eyes * 1.0)

        self.bottom_left = Coordinate(landmarks.nose.x - crop_distance, landmarks.nose.y + crop_distance)
        self.bottom_right = Coordinate(landmarks.nose.x + crop_distance, landmarks.nose.y + crop_distance)
        self.top_left = Coordinate(landmarks.nose.x - crop_distance, landmarks.nose.y - crop_distance)
        self.top_right = Coordinate(landmarks.nose.x + crop_distance, landmarks.nose.y - crop_distance)

        self.crop = img_raw[self.top_left.y:self.bottom_right.y, self.top_left.x:self.bottom_right.x]
        return self.crop

from application.core.api.result.BoundingBox import BoundingBox, Probability, Coordinate, Landmarks, Crop
import numpy as np

class Face:
    """Face class represents the data that together represent a human face."""

    def __init__(self):
        self.bound_box: BoundingBox = BoundingBox()
        self.crop: Crop = Crop()
        self.landmarks: Landmarks = Landmarks()
        self.prob: Probability = Probability()

    def add_bounding_box(self, ann_0: float, ann_1: float, ann_2: float, ann_3: float, img_width: int, img_height: int):
        self.bound_box.x1, self.bound_box.y1 = int(ann_0 * img_width), int(ann_1 * img_height)
        self.bound_box.x2, self.bound_box.y2 = int(ann_2 * img_width), int(ann_3 * img_height)

    def add_landmarks(self, ann: np.ndarray, img_width: int, img_height: int):
        self.landmarks.left_eye = Coordinate(int(ann[4] * img_width), int(ann[5] * img_height))
        self.landmarks.right_eye = Coordinate(int(ann[6] * img_width), int(ann[7] * img_height))
        self.landmarks.nose = Coordinate(int(ann[8] * img_width), int(ann[9] * img_height))
        self.landmarks.left_corner_mouth = Coordinate(int(ann[10] * img_width), int(ann[11] * img_height))
        self.landmarks.right_corner_mouth = Coordinate(int(ann[12] * img_width), int(ann[13] * img_height))

    def add_probability(self, ann_15: float):
        self.prob.probability = ann_15
        self.prob.as_text = f"{ann_15:.4f}"

    def add_crop(self, landmarks: Landmarks, img_raw: np.ndarray):
        self.crop = Crop().get_image_crop(landmarks, img_raw)

    def get_crop(self) -> Crop:
        return self.crop

    def get_bounding_box(self) -> BoundingBox:
        return self.bound_box

    def get_landmarks(self) -> Landmarks:
        return self.landmarks

    def get_probability(self) -> Probability:
        return self.prob

from application.core.api.result.BoundingBox import BoundingBox, Probability, Coordinate, Landmarks, Crop


class Face:
    """Face class represents the data that together represent a human face

    Features:
        - BoundingBox
        - Landmarks
        - Probability
    """

    def __init__(self):
        self.bound_box = BoundingBox()
        self.crop = Crop()
        self.landmarks = Landmarks()
        self.prob = Probability()

    def add_bounding_box(self, ann_0, ann_1, ann_2, ann_3, img_width, img_height):
        self.bound_box.x1, self.bound_box.y1, self.bound_box.x2, self.bound_box.y2 = int(ann_0 * img_width), int(ann_1 * img_height), \
                         int(ann_2 * img_width), int(ann_3 * img_height)

    def add_landmarks(self, ann, img_width, img_height):
        self.landmarks.left_eye = Coordinate(int(ann[4] * img_width), int(ann[5] * img_height))
        self.landmarks.right_eye = Coordinate(int(ann[6] * img_width), int(ann[7] * img_height))
        self.landmarks.nose = Coordinate(int(ann[8] * img_width), int(ann[9] * img_height))
        self.landmarks.left_corner_mouth = Coordinate(int(ann[10] * img_width), int(ann[11] * img_height))
        self.landmarks.right_corner_mouth = Coordinate(int(ann[12] * img_width), int(ann[13] * img_height))

    def add_probability(self, ann_15):
        self.prob.probability = ann_15
        self.prob.as_text = "{:.4f}".format(ann_15)

    def add_crop(self, landmarks, img_raw):
        self.crop = Crop().get_image_crop(landmarks, img_raw)

    def get_crop(self):
        return self.crop

    def get_bounding_box(self):
        return self.bound_box

    def get_landmarks(self):
        return self.landmarks

    def get_probability(self):
        return self.prob

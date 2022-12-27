import cv2


class BoundingBox:
    """BoundingBox class represent the rectangle around the face that was detected using four coordinates

    Features:
        - List of 8 int32 numbers
    """

    def __init__(self):
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None
        # self.x3
        # self.y3
        # self.x4
        # self.y4

    def draw_bounding_box(self, img):
        cv2.rectangle(img, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 2)

        return img


class Landmarks:
    """BoundingBox class represent the rectangle around the face that was detected using four coordinates

    Features:
        - List of 8 int32 numbers
    """

    def __init__(self):
        self.left_eye = None
        self.right_eye = None
        self.nose = None
        self.left_corner_mouth = None
        self.right_corner_mouth = None

    def draw_landmarks(self, img_raw):
        # left eye
        img_raw = cv2.circle(img_raw, (self.left_eye.get_horizontal(), self.left_eye.get_vertical()), 1, (0, 255, 255),
                             2)
        # # right eye
        img_raw = cv2.circle(img_raw, (self.right_eye.get_horizontal(), self.right_eye.get_vertical()), 1,
                             (255, 255, 0), 2)
        # # nose
        img_raw = cv2.circle(img_raw, (self.nose.get_horizontal(), self.nose.get_vertical()), 1, (255, 0, 0), 2)
        # # left mouth corner
        img_raw = cv2.circle(img_raw, (self.left_corner_mouth.get_horizontal(), self.left_corner_mouth.get_vertical()),
                             1, (0, 100, 255), 2)
        # # right mouth corner
        img_raw = cv2.circle(img_raw,
                             (self.right_corner_mouth.get_horizontal(), self.right_corner_mouth.get_vertical()), 1,
                             (255, 0, 100), 2)
        return img_raw


class Probability:
    """Probability represent the value of certainty regarding the results

    Features:
        - probability (raw output)
        - as_text (output sanitized)
    """

    def __init__(self):
        self.probability = None
        self.as_text = None

    def draw_probability(self, img, x1, y1):
        cv2.putText(img, self.as_text, (x1, y1),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        return img


class Coordinate:
    """Coordinate represents the x and y carthesian coordinates of a point

    Features:
        - x = horizontal coordinate
        - y = vertical coordinate
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_horizontal(self):
        return self.x

    def get_vertical(self):
        return self.y


class Crop:
    """
    Crop represent rectangle around to the face scaled to the landmarks. It differs from the BoundingBox in that it aims
    to represent a reproducible result suitable as input to a face recognition model

    Rectangle is based on (horizontal distance between the eyes + vertical distance between mouth and eyes)

    This value is then used in all four directions form the nose
    """

    def __init__(self):
        self.crop = None

    def get_image_crop(self, landmarks, img_raw):
        distance_between_eyes = landmarks.right_eye.get_horizontal() - landmarks.left_eye.get_horizontal()
        crop_distance = int(distance_between_eyes * 1.0)

        self.bottom_left = Coordinate(landmarks.nose.get_horizontal() - crop_distance,
                                      landmarks.nose.get_vertical() + crop_distance)
        self.bottom_right = Coordinate(landmarks.nose.get_horizontal() + crop_distance,
                                       landmarks.nose.get_vertical() + crop_distance)
        self.top_left = Coordinate(landmarks.nose.get_horizontal() - crop_distance,
                                   landmarks.nose.get_vertical() - crop_distance)
        self.top_right = Coordinate(landmarks.nose.get_horizontal() + crop_distance,
                                    landmarks.nose.get_vertical() - crop_distance)

        self.crop = img_raw[self.top_left.get_vertical():self.bottom_right.get_vertical(),
                    self.top_left.get_horizontal():self.bottom_right.get_horizontal()]

        return self.crop

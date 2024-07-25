from application.core.api.result.Face import Face
import numpy as np

class Result:
    """Result represents the output of detection model inference, containing lists of Faces and an annotated image."""

    def __init__(self):
        self.faces: list[Face] = []
        self.annotated_image: np.ndarray = None

    def process_result(self, img_raw: np.ndarray, outputs: np.ndarray, img_height_raw: int, img_width_raw: int, img: np.ndarray):
        """Process the output of the face detection model.

        Args:
            img_raw: The original input image.
            outputs: Model output.
            img_height_raw: Original image height.
            img_width_raw: Original image width.
            img: Image to annotate.
        """
        for output in outputs:
            face = self.draw_bbox_landm(output, img_height_raw, img_width_raw, img)
            self.faces.append(face)
        self.annotated_image = img_raw

    def annotate_image(self, img_raw: np.ndarray):
        """Annotate the input image with bounding boxes, landmarks, and probabilities.

        Args:
            img_raw: The original input image.

        Returns:
            The annotated image.
        """
        for face in self.faces:
            img_raw = face.get_bounding_box().draw_bounding_box(img_raw)
            img_raw = face.get_probability().draw_probability(img_raw, face.bound_box.x1, face.bound_box.y1)
            img_raw = face.get_landmarks().draw_landmarks(img_raw)
        self.annotated_image = img_raw

    def draw_bbox_landm(self, ann: np.ndarray, img_height: int, img_width: int, img_raw: np.ndarray) -> Face:
        """Draw bounding boxes and landmarks.

        Args:
            ann: Annotation data.
            img_height: Image height.
            img_width: Image width.
            img_raw: Original input image.

        Returns:
            Face object with bounding box, landmarks, and crop information.
        """
        face = Face()
        face.add_bounding_box(ann[0], ann[1], ann[2], ann[3], img_width, img_height)
        face.add_probability(ann[15])
        if ann[14] > 0:
            face.add_landmarks(ann, img_width, img_height)
        face.add_crop(face.get_landmarks(), img_raw)
        return face

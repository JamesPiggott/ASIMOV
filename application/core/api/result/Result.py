from application.core.api.result.Face import Face


class Result:
    """Result represents the output of detection model inference

    Features:
        - lists Faces
    """

    def __init__(self):
        self.faces = []
        self.annotated_image = None

    def process_result(self, img_raw, outputs, img_height_raw, img_width_raw, img):
        """Process the output of face detection model

        Args:
            box: coordinates of possible bounding boxes
            probability: chance that output is a human face
            landmarks: coordinates of five facial landmarks (eyes, nose and corners of mouth)

        Returns:
            faces: list of Face (BoundingBox, Landmarks, Probability).
        """
        for prior_index in range(len(outputs)):
            self.faces.append(self.draw_bbox_landm(outputs[prior_index], img_height_raw, img_width_raw, img))

        self.annotated_image = img_raw

    def annotate_image(self, img_raw):
        """Annotate the input image with a bounding box, landmarks and probability

        Returns:
            annotated_image: the image showing the location / details of face detections.
        """

        for face in self.faces:

            # bbox
            img = face.get_bounding_box().draw_bounding_box(img_raw)

            # confidence
            img = face.get_probability().draw_probability(img, face.bound_box.x1, face.bound_box.y1)

            # landmarks
            self.annotated_image = face.get_landmarks().draw_landmarks(img)

    # Support functions

    def draw_bbox_landm(self, ann, img_height, img_width, img_raw):
        """draw bboxes and landmarks"""

        face = Face()

        # bbox
        face.add_bounding_box(ann[0], ann[1], ann[2], ann[3], img_width, img_height)

        # confidence
        face.add_probability(ann[15])

        # landmarks
        if ann[14] > 0:
            face.add_landmarks(ann, img_width, img_height)

        # crops
        face.add_crop(face.get_landmarks(), img_raw)

        return face

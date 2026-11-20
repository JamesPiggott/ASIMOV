from application.core import Settings
from application.core.utilities.config_parser import Configure

class API:
    """
    Entryway into the face detection/recognition application.

    Provides high-level API with input being just an image with the return being of Result class.

    Example use:

        import cv2
        from application.core.API import API

        img = cv2.imread('alicia_vikander.jpg')

        api = API()
        result = api.detect_faces(img, True)
        embeds = api.recognize_faces(img)
    """

    def __init__(self):
        self.configuration = Configure()
        from application.core.FaceQualityAnalyzer import FaceQualityAnalyzer
        self.quality_analyzer = FaceQualityAnalyzer()

        if self.configuration.perform_detection:
            from application.core.api.detection.retinaface.DetectionProcessing import DetectionProcessing
            self.detector = DetectionProcessing(Settings.location_retina_face_model)

        if self.configuration.perform_recognition:
            from application.core.api.recognition.arcface.RecognitionProcessing import RecognitionProcessing
            self.recognition = RecognitionProcessing(Settings.location_arcface_model)

    def detect_faces(self, frame: 'np.ndarray', draw_bounding_box: bool) -> 'Result':
        """
        Detects coordinates within the image corresponding to human faces.

        :param frame: The input image from which we want to extract facial coordinates.
        :param draw_bounding_box: Option to draw the bounding box with the returning Result (annotated_image).
        :return: Result consists of Face, BoundingBox, and annotated_image.
        """
        return self.detector.model_inference(frame, draw_bounding_box)

    def recognize_faces(self, image: 'np.ndarray') -> 'Match':
        """
        Checks if the face in the image is known to the application.

        :param image: A face cropping.
        :return: Match consists of Euclidean distance and a boolean to indicate if a face match is found in the database.
        """
        return self.recognition.model_inference(image)

    def compare_faces(self, first_crop: 'np.ndarray', second_crop: 'np.ndarray') -> 'Match':
        """
        Compares two face crops to determine any similarity.

        :param first_crop: A face cropping.
        :param second_crop: A face cropping.
        :return: Match consists of Euclidean distance and a boolean to indicate if a face match is found.
        """
        return self.recognition.compare_faces(first_crop, second_crop)

    def assess_face_quality(self, image: 'np.ndarray', face: 'Face', crop: 'np.ndarray' = None) -> dict:
        """Measure a detected face's image quality without running model inference.

        Args:
            image: Original BGR source image.
            face: Face returned by detect_faces().
            crop: Optional face crop; defaults to face.get_crop().
        Returns:
            Dictionary of descriptive metrics and heuristic warnings.
        """
        return self.quality_analyzer.assess(image, face, crop)

    def crop_face(self, image: 'np.ndarray', face: 'Face', padding: float = 0.0, mode: str = 'legacy') -> 'np.ndarray':
        """Crop around the detected nose, scaling total width/height by 1 + padding.

        0.0 preserves the original ASIMOV crop for in-bounds faces.
        Padding is a fraction of TOTAL crop dimensions, not a fraction per side.
        Uses the source image; never enlarges an already extracted crop.
        """
        import numpy as np
        if mode == 'aligned':
            return self.align_face(image, face)
        if mode != 'legacy':
            raise ValueError(f"Unknown crop mode: {mode}")
        if image is None or image.size == 0:
            raise ValueError("Empty source image")
        if padding not in (0.0, 0.10, 0.15):
            raise ValueError("Crop padding must be 0.0, 0.10, or 0.15")
        marks = face.landmarks
        if marks.left_eye is None or marks.right_eye is None or marks.nose is None:
            return np.empty((0, 0, 3), dtype=image.dtype)
        # Preserve the legacy integer half-size at 0%, including its signed
        # horizontal eye-distance convention for exact baseline comparison.
        legacy_radius = int(marks.right_eye.x - marks.left_eye.x)
        radius = legacy_radius if padding == 0.0 else int(round(legacy_radius * (1.0 + padding)))
        if radius <= 0:
            return np.empty((0, 0, 3), dtype=image.dtype)
        cx, cy = marks.nose.x, marks.nose.y
        h, w = image.shape[:2]
        x1, y1 = max(0, cx-radius), max(0, cy-radius)
        x2, y2 = min(w, cx+radius), min(h, cy+radius)
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=image.dtype)
        return image[y1:y2, x1:x2].copy()

    def align_face(self, image: 'np.ndarray', face: 'Face') -> 'np.ndarray':
        """Five-landmark similarity alignment to the common ArcFace 112x112 template.

        Works on the original BGR source image, not an existing crop. The
        transformation preserves pose (it does not frontalize 3D head turns).
        """
        import cv2
        import numpy as np
        marks = face.landmarks
        points = (marks.left_eye, marks.right_eye, marks.nose,
                  marks.left_corner_mouth, marks.right_corner_mouth)
        if image is None or image.size == 0 or any(p is None for p in points):
            return np.empty((0, 0, 3), dtype=np.uint8)
        source = np.array([(p.x, p.y) for p in points], dtype=np.float32)
        target = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014],
            [56.0252, 71.7366], [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)
        if not np.isfinite(source).all():
            return np.empty((0, 0, 3), dtype=image.dtype)
        # Closed-form least-squares similarity transform (scale + rotation +
        # translation, no shear). More stable and deterministic than RANSAC.
        src_center = source.mean(axis=0)
        dst_center = target.mean(axis=0)
        src_zero = source - src_center
        dst_zero = target - dst_center
        denominator = float(np.sum(src_zero * src_zero))
        if denominator < 1e-6:
            return np.empty((0, 0, 3), dtype=image.dtype)
        covariance = src_zero.T @ dst_zero
        u, _, vt = np.linalg.svd(covariance)
        diagonal = np.eye(2, dtype=np.float32)
        if np.linalg.det(u @ vt) < 0:
            diagonal[-1, -1] = -1
        rotation = u @ diagonal @ vt
        scale = float(np.sum((src_zero @ rotation) * dst_zero) / denominator)
        if not np.isfinite(scale) or scale <= 0:
            return np.empty((0, 0, 3), dtype=image.dtype)
        # Row-vector formulation: y = x @ (scale * rotation) + translation.
        transform = (scale * rotation).T
        translation = dst_center - src_center @ (scale * rotation)
        matrix = np.column_stack((transform, translation)).astype(np.float32)
        return cv2.warpAffine(image, matrix, (112, 112), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

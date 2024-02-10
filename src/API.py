from src import Settings
from src.utilities.config_parser import Configure


class API:
    """
    Entryway into the face detection / recognition application.

    Provides high-level API with input being just an image with the return being of Result class.

    Example use:

        import cv2
        from src.API import API

        img = cv2.imread('alicia_vikander.jpg')

        api = API()
        result = api.detect_faces(img, True)
        embeds = api.recognize_faces(img)

    """
    def __init__(self):
        self.configuration = Configure()

        if self.configuration.perform_detection:
            from src.api.detection.retinaface.DetectionProcessing import DetectionProcessing
            self.detector = DetectionProcessing(Settings.location_retina_face_model)

        if self.configuration.perform_recognition:
            from src.api.recognition.arcface.RecognitionProcessing import RecognitionProcessing
            self.recognition = RecognitionProcessing(Settings.location_arcface_model)

    def detect_faces(self, frame, draw_bounding_box):
        """Detects coordinates within the image corresponding to human faces

           :param frame: the input image from which we want to extract facial coordinates
           :param draw_bounding_box: option to draw the bounding box with the returning Result (annotated_image)
           :return Result consists of Face, BoundingBox and annotated_image
        """
        return self.detector.model_inference(frame, draw_bounding_box)

    def recognize_faces(self, image):
        """ Checks if the face in the image is known to the application

           :param image is a face cropping
           :return Match consists of Euclidean distance and boolean to indicate if face match is found in database
        """
        return self.recognition.model_inference(image)

    def compare_faces(self, first_crop, second_crop):
        """Compares two face crops to determine any similarity.

           :param first_crop is a face cropping
           :param second_crop is a face cropping
           :return Match consists of Euclidean distance and boolean to indicate if face match is found
        """
        return self.recognition.compare_faces(first_crop, second_crop)

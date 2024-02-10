import cv2
from application.core.API import API
from application.core.api.detection.Alignment import *

api = API()

# Load the slightly misaligned image of the actress
frame = cv2.imread('test/sample_images/scarlett_johansson.jpg')
result = api.detect_faces(frame, True)

# Attempt to align the face
aligned = alignment_procedure(result.faces[0])

while True:
    cv2.imshow("test", aligned)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

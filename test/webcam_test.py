import cv2
from src.API import API

vid = cv2.VideoCapture(0)

api = API()

while True:
    ret, frame = vid.read()
    result = api.detect_faces(frame, True)
    cv2.imshow("test", result.annotated_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

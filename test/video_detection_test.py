import cv2
from src.API import API

video = cv2.VideoCapture('sample_videos/vikander.mp4')

api = API()

while True:
    ret, frame = video.read()
    result = api.detect_faces(frame, True)
    cv2.imshow("test", result.annotated_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


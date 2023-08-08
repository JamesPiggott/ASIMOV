"""
A very basic test of the face detection video. Each frame of the video is processed through RetinaFace sequentially and
the annotated image is returned and displayed.

"""

import cv2 as cv

from src.API import API

video = cv.VideoCapture('sample_videos/vikander.mp4')

api = API()

configuration = api.get_configuration()
configuration.perform_recognition(False)

while video.isOpened():
    ret, frame = video.read()
    result = api.detect_faces(frame, True)
    cv.imshow("test", result.annotated_image)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


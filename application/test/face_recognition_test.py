import pathlib

import cv2 as cv2
import tensorflow as tf

from application.core.API import API

api = API()

# Load the slightly misaligned image of the actress
frame = cv2.imread(str(pathlib.Path("sample_images/alicia_vikander.jpg")))
result = api.detect_faces(frame, True)

# Test face vectors
val = api.recognize_faces(result.faces[0].get_crop())
tensor = tf.add(val, 1)

print("The transformed tensor into a numpy array is= ", api.recognize_faces(result.faces[0].get_crop()).numpy()[0])

# TODO https://www.rathishkumar.in/2021/03/face-recognition-euclidean-distance-sql.html
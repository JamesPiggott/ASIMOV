import cv2

from src.API import API

api = API()

# Load the slightly misaligned image of the actress
frame = cv2.imread('test/sample_images/scarlett_johansson.jpg')
result = api.detect_faces(frame, True)

# Test face vectors
print("Length of face vector is: " + str(api.recognize_faces(result.faces[0].get_crop())))

# TODO https://www.rathishkumar.in/2021/03/face-recognition-euclidean-distance-sql.html
import math

import cv2
import numpy as np
from PIL import Image

from src.API import API

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


# this function copied from the deepface repository: https://github.com/serengil/deepface/blob/master/deepface/commons/functions.py
def alignment_procedure(face):
    # this function aligns given face in img based on left and right eye coordinates

    # left eye is the eye appearing on the left (right eye of the person)
    # left top point is (0, 0)
    img = face.get_crop()
    left_eye = face.get_landmarks().left_eye
    right_eye = face.get_landmarks().right_eye
    nose = face.get_landmarks().nose

    left_eye_x, left_eye_y = left_eye.get_horizontal(), left_eye.get_vertical()
    right_eye_x, right_eye_y = right_eye.get_horizontal(), right_eye.get_vertical()

    left_eye = (left_eye_x, left_eye_y)
    right_eye = (right_eye_x, right_eye_y)
    nose = (nose.get_horizontal(), nose.get_vertical())

    # -----------------------
    # decide the image is inverse

    center_eyes = (int((left_eye_x + right_eye_x) / 2), int((left_eye_y + right_eye_y) / 2))

    # if False:
    #     img = cv2.circle(img, (int(left_eye.get_horizontal()), int(left_eye.get_vertical())), 2, (0, 255, 255), 2)
    #     img = cv2.circle(img, (int(right_eye.get_horizontal()), int(right_eye.get_vertical())), 2, (255, 0, 0), 2)
    #     img = cv2.circle(img, center_eyes, 2, (0, 0, 255), 2)
    #     img = cv2.circle(img, (int(nose.get_horizontal()), int(nose.get_vertical())), 2, (255, 255, 255), 2)

    # -----------------------
    # find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    # -----------------------

    # apply cosine rule

    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

        cos_a = (b * b + c * c - a * a) / (2 * b * c)

        # PR15: While mathematically cos_a must be within the closed range [-1.0, 1.0], floating point errors would produce cases violating this
        # In fact, we did come across a case where cos_a took the value 1.0000000169176173, which lead to a NaN from the following np.arccos step
        cos_a = min(1.0, max(-1.0, cos_a))

        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

        if center_eyes[1] > nose[1]:
            img = Image.fromarray(img)
            img = np.array(img.rotate(180))

    # -----------------------

    return img  # return img anyway

api = API()

# Detect Scarlett Johansson's face, and save the crop
frame1 = cv2.imread('test/sample_images/scarlett_johansson.jpg')
result1 = api.detect_faces(frame1, True)
cv2.imwrite("test/sample_images/scarlett_crop.jpg", result1.faces[0].get_crop())

# Create another crop of Scarlett Johansson
frame2 = cv2.imread('test/sample_images/scarlett_johansson2.jpg')
result2 = api.detect_faces(frame2, True)
cv2.imwrite("test/sample_images/scarlett_crop2.jpg", result2.faces[0].get_crop())

# Create another crop of Scarlett Johansson
frame3 = cv2.imread('test/sample_images/scarlett22.jpg')
result3 = api.detect_faces(frame3, True)
cv2.imwrite("test/sample_images/scarlett22_crop.jpg", result3.faces[0].get_crop())

# Detect Alicia Vikander's face, and save the crop
frame4 = cv2.imread('test/sample_images/alicia_vikander.jpg')
result4 = api.detect_faces(frame4, True)
cv2.imwrite("test/sample_images/alicia_crop.jpg", result4.faces[0].get_crop())

# Detect Alicia lookalike face, and save the crop
frame5 = cv2.imread('test/sample_images/alicia_lookalike.JPG')
result5 = api.detect_faces(frame5, True)
cv2.imwrite("test/sample_images/alicia_lookalike_crop.jpg", result5.faces[0].get_crop())

# Detect Michael Fassbender's face, and save the crop
frame6 = cv2.imread('test/sample_images/michael_fassbender.jpg')
result6 = api.detect_faces(frame6, True)
cv2.imwrite("test/sample_images/michael_crop.jpg", result6.faces[0].get_crop())

# Compare the same crops
print("Scarlett 2 times the same            :   " + str(api.compare_faces(result1.faces[0].get_crop(), result1.faces[0].get_crop())))

# Compare two different crops of the same person
print("Scarlett 2 times different           :   " + str(api.compare_faces(result1.faces[0].get_crop(), result2.faces[0].get_crop())))
aligned1 = alignment_procedure(result1.faces[0])
aligned2 = alignment_procedure(result2.faces[0])
print("Scarlett 2 times different aligned   :   " + str(api.compare_faces(aligned1, aligned2)))

# Compare crops of Scarlett Johansson
print("Scarlett 2 times different           :   " + str(api.compare_faces(result1.faces[0].get_crop(), result3.faces[0].get_crop())))
aligned1 = alignment_procedure(result1.faces[0])
aligned2 = alignment_procedure(result3.faces[0])
print("Scarlett 2 times different aligned   :   " + str(api.compare_faces(aligned1, aligned2)))

# Compare crops of two women
print("Scarlett + Alicia                    :   " + str(api.compare_faces(result1.faces[0].get_crop(), result4.faces[0].get_crop())))
aligned1 = alignment_procedure(result1.faces[0])
aligned2 = alignment_procedure(result4.faces[0])
print("Scarlett + Alicia aligned            :   " + str(api.compare_faces(aligned1, aligned2)))

# Compare of man and woman
print("Michael + Alicia                     :   " + str(api.compare_faces(result4.faces[0].get_crop(), result6.faces[0].get_crop())))
aligned1 = alignment_procedure(result4.faces[0])
aligned2 = alignment_procedure(result6.faces[0])
print("Michael + Alicia aligned             :   " + str(api.compare_faces(aligned1, aligned2)))

# Compare of Alicia Vikander with her lookalike
print("Alicia + lookalike                   :   " + str(api.compare_faces(result4.faces[0].get_crop(), result5.faces[0].get_crop())))
aligned1 = alignment_procedure(result4.faces[0])
aligned2 = alignment_procedure(result5.faces[0])
print("Alicia + lookalike aligned           :   " + str(api.compare_faces(aligned1, aligned2)))


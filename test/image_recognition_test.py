import cv2

from src.API import API
from src.api.detection.Alignment import *

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

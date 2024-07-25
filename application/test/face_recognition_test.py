import pathlib
import cv2
import tensorflow as tf
from application.core.API import API


def main():
    # Initialize the API
    api = API()

    # Load the slightly misaligned image of the actress
    image_path = pathlib.Path("sample_images/alicia_vikander.jpg")
    frame = cv2.imread(str(image_path))

    if frame is None:
        raise FileNotFoundError(f"Image file not found at {image_path}")

    # Detect faces in the image
    result = api.detect_faces(frame, True)

    if not result.faces:
        raise ValueError("No faces detected in the image")

    # Get the crop of the first detected face
    face_crop = result.faces[0].get_crop()

    # Recognize the face
    face_vector = api.recognize_faces(face_crop)

    # Perform a tensor operation
    transformed_tensor = tf.add(face_vector, 1)

    # Print the transformed tensor as a numpy array
    print("The transformed tensor into a numpy array is:", transformed_tensor.numpy()[0])


if __name__ == "__main__":
    main()

# TODO https://www.rathishkumar.in/2021/03/face-recognition-euclidean-distance-sql.html

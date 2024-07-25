import os

import cv2
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from numpy.linalg import norm


class RecognitionProcessing:
    """
    Processes the crop input to make it suitable for model inference.
    """

    def __init__(self, model_location: str):
        """
        Initialize the RecognitionProcessing with the location of the model.

        Args:
            model_location (str): Path to the directory containing the arcface model.
        """
        self.model = tf.saved_model.load(model_location + 'arcface')

    def l2_norm(self, x: np.ndarray, axis: int = 1) -> np.ndarray:
        """
        Normalize the input using L2 norm.

        Args:
            x (np.ndarray): Input array to be normalized.
            axis (int): Axis along which to compute the norm.

        Returns:
            np.ndarray: L2-normalized array.
        """
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / norm

    def resize_and_float(self, img: np.ndarray) -> np.ndarray:
        """
        Resize the image to 112x112 and normalize pixel values to [0, 1].

        Args:
            img (np.ndarray): Input image array.

        Returns:
            np.ndarray: Resized and normalized image array.
        """
        img = cv2.resize(img, (112, 112))
        img = img.astype(np.float32) / 255.0
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        return img

    def model_inference(self, img: np.ndarray) -> np.ndarray:
        """
        Perform model inference on the input image.

        Args:
            img (np.ndarray): Input image array.

        Returns:
            np.ndarray: Embeddings from the model after L2 normalization.
        """
        img = self.resize_and_float(img)
        embeds = self.model(img)
        return self.l2_norm(embeds)

    def euclidean_distance_face_vector(self, first: np.ndarray, second: np.ndarray) -> float:
        """
        Compute the Euclidean distance between two face vectors.

        Args:
            first (np.ndarray): First face embedding vector.
            second (np.ndarray): Second face embedding vector.

        Returns:
            float: Euclidean distance between the two vectors.
        """
        return np.linalg.norm(first - second)

    def compare_faces(self, first_face: np.ndarray, second_face: np.ndarray) -> float:
        """
        Compare two face images and compute the Euclidean distance between their embeddings.

        Args:
            first_face (np.ndarray): First face image array.
            second_face (np.ndarray): Second face image array.

        Returns:
            float: Euclidean distance between the embeddings of the two faces.
        """
        first_embedding = self.model_inference(first_face).numpy()
        second_embedding = self.model_inference(second_face).numpy()
        return self.euclidean_distance_face_vector(first_embedding, second_embedding)

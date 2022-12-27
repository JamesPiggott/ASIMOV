import cv2
import numpy as np
import tensorflow as tf
from numpy.linalg import norm


class RecognitionProcessing:

    def __init__(self, model_location):
        """
        Process the crop input to make it suitable for model inference
        """
        self.model = tf.saved_model.load(model_location + 'arcface')

    def l2_norm(self, x, axis=1):
        """l2 norm"""
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        output = x / norm
        return output

    def resize_and_float(self, img):
        img = cv2.resize(img, (112, 112))
        img = img.astype(np.float32) / 255.
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        return img

    def model_inference(self, img):
        img = self.resize_and_float(img)
        # embeds = self.model(img)
        embeds = self.l2_norm(self.model(img))
        return embeds

    def euclidean_distance_face_vector(self, first, second):
        return np.linalg.norm(first - second)

        # diff = np.subtract(first, second)
        # return np.sum(np.square(diff), 1)

        # return dot(first, second) / (norm(first) * norm(second))

        # embeddings1 = first/np.linalg.norm(first, axis=1, keepdims=True)
        # embeddings2 = second/np.linalg.norm(second, axis=1, keepdims=True)
        # diff = np.subtract(embeddings1, embeddings2)
        # return np.sum(np.square(diff),1)

        # dot = np.sum(np.multiply(first, second), axis=1)
        # norm = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        # similarity = dot/norm
        # return np.arccos(similarity) / math.pi

        # s_diff = tf.math.squared_difference(first, second)
        # return tf.unstack(tf.reduce_sum(s_diff, axis=1))

    def compare_faces(self, first_face, second_face):
            first_embedding = self.model_inference(first_face).numpy()
            second_embedding = self.model_inference(second_face).numpy()
            distance = self.euclidean_distance_face_vector(first_embedding, second_embedding)
            return distance

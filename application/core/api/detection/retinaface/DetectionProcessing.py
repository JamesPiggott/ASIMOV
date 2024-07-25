import cv2
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from application.core.api.result.Result import Result


class DetectionProcessing:
    """
    Process the image to make it suitable for face detection model inference.
    """

    def __init__(self, model_location: str):
        self.model = tf.saved_model.load(model_location + 'mbv2_model')

    def pad_input_image(self, img: 'np.ndarray', max_steps: int) -> ('np.ndarray', tuple):
        """
        Pad image to suitable shape.
        """
        img_h, img_w, _ = img.shape

        img_pad_h = (max_steps - img_h % max_steps) if img_h % max_steps > 0 else 0
        img_pad_w = (max_steps - img_w % max_steps) if img_w % max_steps > 0 else 0

        padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
        img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w, cv2.BORDER_CONSTANT, value=padd_val.tolist())
        pad_params = (img_h, img_w, img_pad_h, img_pad_w)

        return img, pad_params

    def recover_pad_output(self, outputs: 'np.ndarray', pad_params: tuple) -> 'np.ndarray':
        """
        Recover the padded output effect.
        """
        img_h, img_w, img_pad_h, img_pad_w = pad_params
        recover_xy = np.reshape(outputs[:, :14], [-1, 7, 2]) * \
                     [(img_pad_w + img_w) / img_w, (img_pad_h + img_h) / img_h]
        outputs[:, :14] = np.reshape(recover_xy, [-1, 14])

        return outputs

    def model_inference(self, img_raw: 'np.ndarray', draw_bounding_box: bool) -> 'Result':
        img_height_raw, img_width_raw, _ = img_raw.shape
        img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_test = img_raw.copy()

        # Optionally resize the image
        if 0.5 < 1.0:
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

        # Pad input image to avoid unmatched shape problem
        img, pad_params = self.pad_input_image(img, max_steps=max([8, 16, 32]))

        # Run model
        outputs = self.model(img[np.newaxis, ...]).numpy()

        # Recover padding effect
        outputs = self.recover_pad_output(outputs, pad_params)

        # Process results
        result = Result()
        result.process_result(img_raw, outputs, img_height_raw, img_width_raw, img_test)

        if draw_bounding_box:
            result.annotate_image(img_raw)

        return result

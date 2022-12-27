import cv2
import numpy as np
import tensorflow as tf

from src.api.result.Result import Result


class DetectionProcessing:

    def __init__(self, model_location):
        """
        Process the image to make it suitable for face detection model inference
        """
        self.model = tf.saved_model.load(model_location + 'mbv2_model')

    def pad_input_image(self, img, max_steps):
        """pad image to suitable shape"""
        img_h, img_w, _ = img.shape

        img_pad_h = 0
        if img_h % max_steps > 0:
            img_pad_h = max_steps - img_h % max_steps

        img_pad_w = 0
        if img_w % max_steps > 0:
            img_pad_w = max_steps - img_w % max_steps

        padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
        img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,
                                 cv2.BORDER_CONSTANT, value=padd_val.tolist())
        pad_params = (img_h, img_w, img_pad_h, img_pad_w)

        return img, pad_params

    def recover_pad_output(self, outputs, pad_params):
        """recover the padded output effect"""
        img_h, img_w, img_pad_h, img_pad_w = pad_params
        recover_xy = np.reshape(outputs[:, :14], [-1, 7, 2]) * \
                     [(img_pad_w + img_w) / img_w, (img_pad_h + img_h) / img_h]
        outputs[:, :14] = np.reshape(recover_xy, [-1, 14])

        return outputs

    def model_inference(self, img_raw, draw_bounding_box):
        img_height_raw, img_width_raw, _ = img_raw.shape
        img = np.float32(img_raw.copy())
        img_test = img_raw.copy()

        if 0.5 < 1.0:
            img = cv2.resize(img, (0, 0), fx=0.5,
                             fy=0.5,
                             interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # pad input image to avoid unmatched shape problem
        img, pad_params = self.pad_input_image(img, max_steps=max([8, 16, 32]))

        # run model
        outputs = self.model(img[np.newaxis, ...]).numpy()

        # recover padding effect
        outputs = self.recover_pad_output(outputs, pad_params)

        # Process results
        result = Result()
        result.process_result(img_raw, outputs, img_height_raw, img_width_raw, img_test)

        if draw_bounding_box:
            result.annotate_image(img_raw)

        # cv2.imshow('Frame', result.annotated_image)


        return result
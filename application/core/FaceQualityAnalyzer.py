"""Reference face quality measurements and advisory, uncalibrated assessment.

Does not use ArcFace embeddings, infer identity, or estimate calibrated pose angles.
"""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class QualityThresholds:
    min_face_pixels: int = 90
    min_eye_distance: float = 35.0
    min_sharpness: float = 65.0  # Laplacian variance at standardized 160px face ROI
    min_brightness: float = 55.0
    max_brightness: float = 210.0
    min_contrast: float = 28.0
    max_dark_fraction: float = 0.30
    max_bright_fraction: float = 0.30
    max_nose_offset: float = 0.27
    max_eye_tilt: float = 14.0
    max_yaw_asymmetry: float = 0.35


class FaceQualityAnalyzer:
    """Pure OpenCV/NumPy measurements; thresholds are experimental and advisory."""

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or QualityThresholds()

    @staticmethod
    def _gray(image):
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def assess(self, image, face, crop=None):
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("A non-empty source image is required")
        if image.ndim not in (2, 3):
            raise ValueError("Expected grayscale or BGR image")
        if crop is None:
            crop = face.get_crop()
        if not isinstance(crop, np.ndarray) or crop.size == 0:
            raise ValueError("A non-empty face crop is required")
        h, w = image.shape[:2]
        box = face.get_bounding_box()
        x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
        x2, y2 = min(w, int(box.x2)), min(h, int(box.y2))
        fw, fh = max(0, x2 - x1), max(0, y2 - y1)
        if fw < 2 or fh < 2:
            raise ValueError("Detected face has no valid bounding box within the source image")
        # Analyze the central facial region, rather than background in the recognition crop.
        roi = self._gray(image[y1:y2, x1:x2])
        rh, rw = roi.shape[:2]
        yy, xx = np.ogrid[:rh, :rw]
        mask = ((xx - (rw - 1) / 2) / max(1, rw * 0.40)) ** 2 + \
               ((yy - (rh - 1) / 2) / max(1, rh * 0.43)) ** 2 <= 1.0
        pixels = roi[mask].astype(np.float64)
        # Sharpness is computed at a common face-region resolution. This improves
        # comparability but is not a scale-invariant optical blur measurement.
        standardized = cv2.resize(roi, (160, 160), interpolation=cv2.INTER_AREA if max(rh, rw) > 160 else cv2.INTER_LINEAR)
        lap = cv2.Laplacian(standardized, cv2.CV_64F)
        mask160 = cv2.resize(mask.astype(np.uint8), (160, 160), interpolation=cv2.INTER_NEAREST).astype(bool)
        sharpness = float(lap[mask160].var())
        confidence = getattr(face.get_probability(), 'probability', None)
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        metrics = {
            'face_size': (fw, fh),
            'face_coverage': round(100.0 * fw * fh / (w * h), 1),
            'confidence': confidence,
            'sharpness': round(sharpness, 1),
            'brightness': round(float(pixels.mean()), 1),
            'contrast': round(float(pixels.std()), 1),
            'dark_fraction': round(float(np.mean(pixels <= 35) * 100), 1),
            'bright_fraction': round(float(np.mean(pixels >= 220) * 100), 1),
            'eye_distance': None,
            'eye_tilt': None,
            'nose_offset': None,
            'mouth_tilt': None,
            'yaw_asymmetry': None,
            'pitch_indicator': None,
        }
        try:
            lm = face.get_landmarks()
            le, re, nose = lm.left_eye, lm.right_eye, lm.nose
            ml, mr = lm.left_corner_mouth, lm.right_corner_mouth
            if all(p is not None for p in (le, re, nose, ml, mr)):
                eye_dx, eye_dy = float(re.x - le.x), float(re.y - le.y)
                eye_distance = float(np.hypot(eye_dx, eye_dy))
                if eye_distance > 1.0:
                    eye_mid = np.array([(le.x + re.x) / 2, (le.y + re.y) / 2], dtype=float)
                    mouth_mid = np.array([(ml.x + mr.x) / 2, (ml.y + mr.y) / 2], dtype=float)
                    nose_point = np.array([nose.x, nose.y], dtype=float)
                    # Rotate the landmark geometry into the eye-line coordinate system.
                    ux = np.array([eye_dx, eye_dy]) / eye_distance
                    uy = np.array([-ux[1], ux[0]])
                    nose_relative = nose_point - eye_mid
                    mouth_relative = mouth_mid - eye_mid
                    eye_nose_left = float(np.hypot(nose.x - le.x, nose.y - le.y))
                    eye_nose_right = float(np.hypot(nose.x - re.x, nose.y - re.y))
                    metrics['eye_distance'] = round(eye_distance, 1)
                    metrics['eye_tilt'] = round(abs(float(np.degrees(np.arctan2(eye_dy, eye_dx)))), 1)
                    metrics['nose_offset'] = round(abs(float(np.dot(nose_relative, ux))) / eye_distance, 2)
                    metrics['mouth_tilt'] = round(abs(float(np.degrees(np.arctan2(mr.y - ml.y, mr.x - ml.x)))), 1)
                    metrics['yaw_asymmetry'] = round(abs(eye_nose_left - eye_nose_right) / max(1.0, (eye_nose_left + eye_nose_right) / 2), 2)
                    vertical_mouth = float(np.dot(mouth_relative, uy))
                    if abs(vertical_mouth) > 1.0:
                        metrics['pitch_indicator'] = round(float(np.dot(nose_relative, uy) / vertical_mouth), 2)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            pass
        t = self.thresholds
        findings = []
        def flag(severity, message):
            findings.append({'severity': severity, 'message': message})
        if min(fw, fh) < t.min_face_pixels:
            flag('poor', 'Detected face is too small for reliable detail')
        if metrics['eye_distance'] is not None and metrics['eye_distance'] < t.min_eye_distance:
            flag('review', 'Limited inter-eye resolution')
        if sharpness < t.min_sharpness:
            flag('review', 'Low face-region sharpness at normalized scale')
        if metrics['brightness'] < t.min_brightness or metrics['brightness'] > t.max_brightness:
            flag('review', 'Face region is unusually dark or bright')
        if metrics['dark_fraction'] / 100 > t.max_dark_fraction:
            flag('review', 'Large underexposed portion of face region')
        if metrics['bright_fraction'] / 100 > t.max_bright_fraction:
            flag('review', 'Large overexposed portion of face region')
        if metrics['contrast'] < t.min_contrast:
            flag('review', 'Low facial contrast')
        if metrics['nose_offset'] is not None and metrics['nose_offset'] > t.max_nose_offset:
            flag('review', 'Possible non-frontal orientation (nose offset)')
        if metrics['yaw_asymmetry'] is not None and metrics['yaw_asymmetry'] > t.max_yaw_asymmetry:
            flag('review', 'Possible non-frontal orientation (landmark asymmetry)')
        if metrics['eye_tilt'] is not None and metrics['eye_tilt'] > t.max_eye_tilt:
            flag('review', 'Head appears rotated in the image')
        if metrics['eye_distance'] is None:
            flag('review', 'Landmarks unavailable or degenerate; pose cannot be assessed')
        metrics['assessment'] = 'Poor' if any(f['severity'] == 'poor' for f in findings) else ('Review' if findings else 'Good')
        metrics['findings'] = findings
        metrics['warnings'] = [f['message'] for f in findings]
        metrics['assessment_note'] = 'Experimental image-quality heuristics; not a calibrated recognition-accuracy or identity score.'
        return metrics

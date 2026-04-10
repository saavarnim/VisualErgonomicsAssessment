import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

class PoseEstimator:
    def __init__(self, model_asset_path='pose_landmarker_lite.task'):
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False)
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def process(self, frame, timestamp_ms):
        """Processes the frame and returns landmarks and the mediapipe pose results."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        results = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        landmarks = {}
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            first_person = results.pose_landmarks[0]
            for idx, lm in enumerate(first_person):
                landmarks[idx] = {
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': getattr(lm, 'visibility', getattr(lm, 'presence', 0.5))
                }
        return landmarks, results

    def extract_upper_body_keypoints(self, landmarks):
        """Helper to specifically extract key joints for 2D analysis."""
        if not landmarks:
            return None

        keypoints = {
            'nose': landmarks.get(0),
            'left_ear': landmarks.get(7),
            'right_ear': landmarks.get(8),
            'left_shoulder': landmarks.get(11),
            'right_shoulder': landmarks.get(12),
            'left_hip': landmarks.get(23),
            'right_hip': landmarks.get(24)
        }
        return keypoints

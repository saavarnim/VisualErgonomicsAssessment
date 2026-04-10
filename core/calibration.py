import math

class CalibrationState:
    def __init__(self):
        self.is_calibrated = False
        self.ghost_landmarks = None
        self.reference_ear_distance = 0.0

    def calibrate(self, landmarks_dict, keypoints):
        """
        Saves the current raw landmarks as the perfect posture reference.
        Also calculates the distance between the ears to establish a baseline for screen distance.
        """
        if not keypoints or not landmarks_dict:
            return False

        self.ghost_landmarks = landmarks_dict

        l_ear = keypoints.get('left_ear')
        r_ear = keypoints.get('right_ear')

        if l_ear and r_ear and l_ear['visibility'] > 0.5 and r_ear['visibility'] > 0.5:
            # We use the horizontal pixel distance between ears as a proxy for depth
            self.reference_ear_distance = abs(r_ear['x'] - l_ear['x'])
            self.is_calibrated = True
            return True
        
        return False

    def check_lean_in(self, current_keypoints, threshold_ratio=1.2):
        """
        Checks if the current ear distance is significantly larger than the calibrated distance,
        which indicates leaning closer to the screen.
        """
        if not self.is_calibrated:
            return False

        l_ear = current_keypoints.get('left_ear')
        r_ear = current_keypoints.get('right_ear')

        if l_ear and r_ear and l_ear['visibility'] > 0.5 and r_ear['visibility'] > 0.5:
            current_distance = abs(r_ear['x'] - l_ear['x'])
            
            if self.reference_ear_distance > 0:
                ratio = current_distance / self.reference_ear_distance
                if ratio > threshold_ratio:
                    return True
        return False

import numpy as np
import math

class PostureAnalyzer:
    def __init__(self, visibility_threshold=0.5):
        self.visibility_threshold = visibility_threshold

    def is_visible(self, kp):
        return kp is not None and kp['visibility'] > self.visibility_threshold

    def _angle_between_points(self, p1, p2):
        """Angle of the line passing through p1 and p2 relative to the vertical line (y-axis)."""
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        # Compute angle in degrees from vertical
        angle = math.degrees(math.atan2(dx, dy))
        # 0 means perfectly vertical. >0 means tilted right logically.
        return angle

    def _midpoint(self, p1, p2):
        return {
            'x': (p1['x'] + p2['x']) / 2.0,
            'y': (p1['y'] + p2['y']) / 2.0
        }

    def analyze(self, keypoints):
        """
        Computes 2D ergonomic metrics.
        Returns a dictionary of angles.
        """
        metrics = {
            'neck_tilt': 0.0,
            'spine_inclination': 0.0,
            'shoulder_imbalance': 0.0,
            'neck_ratio': 1.0,  # 1.0 is healthy upright, dropping means slouching forward
            'is_valid': False
        }

        if not keypoints:
            return metrics
            
        l_shoulder = keypoints.get('left_shoulder')
        r_shoulder = keypoints.get('right_shoulder')
        nose = keypoints.get('nose')
        l_hip = keypoints.get('left_hip')
        r_hip = keypoints.get('right_hip')

        # 1. Shoulder Imbalance Let's compute angle of line between shoulders relative to horizontal
        if self.is_visible(l_shoulder) and self.is_visible(r_shoulder):
            dx = abs(r_shoulder['x'] - l_shoulder['x'])
            dy = abs(r_shoulder['y'] - l_shoulder['y'])
            # Angle relative to horizontal
            shoulder_angle = math.degrees(math.atan2(dy, dx))
            metrics['shoulder_imbalance'] = round(shoulder_angle, 2)
            
            # Find shoulder midpoint
            shoulder_mid = self._midpoint(l_shoulder, r_shoulder)
            
            # 2. Neck Tilt (between shoulder midpoint and nose)
            if self.is_visible(nose):
                # we want the angle relative to vertical. p1=nose, p2=shoulder_mid. 
                # (Remember y grows downwards in image coordinates)
                neck_angle = self._angle_between_points(nose, shoulder_mid)
                # the angle is calculated based on vector pointing down. 0 is perfectly straight.
                metrics['neck_tilt'] = round(abs(neck_angle), 2)
                
                # Forward Slouch Metric (Neck Ratio)
                # Height of nose above shoulder midline compared to shoulder width
                shoulder_width = math.hypot(r_shoulder['x'] - l_shoulder['x'], r_shoulder['y'] - l_shoulder['y'])
                neck_height = abs(shoulder_mid['y'] - nose['y'])
                
                if shoulder_width > 0.05:
                    metrics['neck_ratio'] = round(neck_height / shoulder_width, 2)

            # 3. Spine Inclination (between hip midpoint and shoulder midpoint)
            if self.is_visible(l_hip) and self.is_visible(r_hip):
                hip_mid = self._midpoint(l_hip, r_hip)
                spine_angle = self._angle_between_points(shoulder_mid, hip_mid)
                metrics['spine_inclination'] = round(abs(spine_angle), 2)
                
            metrics['is_valid'] = True

        return metrics

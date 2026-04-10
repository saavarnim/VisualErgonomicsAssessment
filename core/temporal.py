from collections import deque
import time

class FatigueClassifier:
    def __init__(self, window_size=60, fps=30):
        # We hold the history of posture metrics
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.fps = fps
        
        # Thresholds for 'poor' posture
        self.th_neck = 18.0      # degrees
        self.th_spine = 15.0     # degrees
        self.th_shoulder = 8.0   # degrees
        self.th_neck_ratio_min = 0.50  # If nose drops too close to shoulders compared to width, indicates forward slouch

        # States: Active, Stagnant, Fatigued
        self.current_state = "Active"
        
        # Track how long we have been in poor posture
        self.poor_posture_frames = 0
        
        # 2 seconds (60 frames) of poor posture -> Stagnant
        # 5 seconds (150 frames) of poor posture -> Fatigued
        self.stagnant_threshold = 2 * fps
        self.fatigued_threshold = 5 * fps

    def update(self, metrics):
        """
        Takes metrics dict and returns state and strain level.
        """
        if not metrics or not metrics.get('is_valid', False):
            return self.current_state, 0.0

        self.history.append(metrics)

        # Evaluate if current metric is 'poor'
        is_poor = False
        strain_score = 0.0
        
        if metrics['neck_tilt'] > self.th_neck:
            is_poor = True
            strain_score += (metrics['neck_tilt'] - self.th_neck)
        if metrics['spine_inclination'] > self.th_spine:
            is_poor = True
            strain_score += (metrics['spine_inclination'] - self.th_spine)
        if metrics['shoulder_imbalance'] > self.th_shoulder:
            is_poor = True
            strain_score += (metrics['shoulder_imbalance'] - self.th_shoulder)
            
        if metrics.get('neck_ratio', 1.0) < self.th_neck_ratio_min:
            is_poor = True
            # The lower it goes, the worse the strain
            strain_score += (self.th_neck_ratio_min - metrics['neck_ratio']) * 50
            
        if is_poor:
            self.poor_posture_frames += 1
        else:
            # Recover gradually or reset? Let's reset quickly if good posture is assumed
            self.poor_posture_frames = max(0, self.poor_posture_frames - 2)

        if self.poor_posture_frames > self.fatigued_threshold:
            self.current_state = "Fatigued"
        elif self.poor_posture_frames > self.stagnant_threshold:
            self.current_state = "Stagnant"
        else:
            self.current_state = "Active"
            
        # Normalize strain score for visual feedback (0.0 to 1.0 roughly)
        strain_intensity = min(1.0, strain_score / 30.0)

        return self.current_state, strain_intensity

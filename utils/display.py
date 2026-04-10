import cv2

class Visualizer:
    def __init__(self):
        # MediaPipe standard connections
        self.POSE_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), 
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), 
            (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), 
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), 
            (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
        ]
        
    def draw(self, frame, pose_results, metrics, state, fps=0.0, calibration_state=None, lean_warning=False):
        """
        Draws pose skeleton, text overlays, and metrics on the frame.
        Includes ghost skeleton and dynamic stress skeleton.
        """
        h, w, _ = frame.shape
        # --- 1. Draw Ghost Skeleton (if calibrated) ---
        if calibration_state and calibration_state.is_calibrated and calibration_state.ghost_landmarks:
            ghost = calibration_state.ghost_landmarks
            for connection in self.POSE_CONNECTIONS:
                idx1, idx2 = connection
                lm1 = ghost.get(idx1)
                lm2 = ghost.get(idx2)
                if lm1 and lm2 and lm1['visibility'] > 0.5 and lm2['visibility'] > 0.5:
                    pt1 = (int(lm1['x'] * w), int(lm1['y'] * h))
                    pt2 = (int(lm2['x'] * w), int(lm2['y'] * h))
                    cv2.line(frame, pt1, pt2, (100, 255, 100), 2)  # Faint green
            
            for idx, lm in ghost.items():
                if lm['visibility'] > 0.5:
                    pt = (int(lm['x'] * w), int(lm['y'] * h))
                    cv2.circle(frame, pt, 2, (100, 255, 100), -1)

        # --- 2. Draw Live Dynamic Skeleton ---
        if pose_results and pose_results.pose_landmarks and len(pose_results.pose_landmarks) > 0:
            landmarks = pose_results.pose_landmarks[0]
            
            # Determine line color based on state
            if state == "Active":
                line_color = (255, 255, 255) # White
            elif state == "Stagnant":
                line_color = (0, 200, 255) # Yellow/Orange
            else:
                line_color = (0, 0, 255) # Red showing strain!
            
            # Draw connections
            for connection in self.POSE_CONNECTIONS:
                idx1, idx2 = connection
                lm1 = landmarks[idx1]
                lm2 = landmarks[idx2]
                
                vis1 = getattr(lm1, 'visibility', getattr(lm1, 'presence', 0.0))
                vis2 = getattr(lm2, 'visibility', getattr(lm2, 'presence', 0.0))
                
                if vis1 > 0.5 and vis2 > 0.5:
                    pt1 = (int(lm1.x * w), int(lm1.y * h))
                    pt2 = (int(lm2.x * w), int(lm2.y * h))
                    cv2.line(frame, pt1, pt2, line_color, 2)
            
            # Draw points
            for lm in landmarks:
                vis = getattr(lm, 'visibility', getattr(lm, 'presence', 0.0))
                if vis > 0.5:
                    pt = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(frame, pt, 4, (0, 255, 0), -1)

        # Base properties for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # State color logic
        if state == "Active":
            state_color = (0, 255, 0) # Green
        elif state == "Stagnant":
            state_color = (0, 165, 255) # Orange
        else:
            state_color = (0, 0, 255) # Red

        # Overlay text background panel
        cv2.rectangle(frame, (5, 5), (320, 180), (0, 0, 0), -1)
        
        cv2.putText(frame, f"State: {state}", (15, 30), font, 0.8, state_color, 2)
        
        if metrics and metrics.get('is_valid', False):
            cv2.putText(frame, f"Neck Tilt: {metrics['neck_tilt']} deg", (15, 60), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(frame, f"Spine Inc: {metrics['spine_inclination']} deg", (15, 85), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(frame, f"Shoulder Imb: {metrics['shoulder_imbalance']} deg", (15, 110), font, font_scale, (255, 255, 255), thickness)
            neck_ratio_val = metrics.get('neck_ratio', 1.0)
            warning_color = (0, 0, 255) if neck_ratio_val < 0.50 else (255, 255, 255)
            cv2.putText(frame, f"Neck Ratio: {neck_ratio_val} (drop<0.5 is Slouch)", (15, 135), font, 0.5, warning_color, thickness)
        else:
             cv2.putText(frame, "No upper body detected.", (15, 60), font, font_scale, (100, 100, 100), thickness)

        # Distance/Lean Warning
        if lean_warning:
            cv2.putText(frame, "WARNING: TOO CLOSE!", (15, 165), font, 0.7, (0, 0, 255), 2)
        elif calibration_state and not calibration_state.is_calibrated:
            cv2.putText(frame, "Press 'c' to Calibrate", (15, 165), font, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Calibrated (c to reset)", (15, 165), font, 0.6, (0, 255, 0), 1)

        if fps > 0:
            cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 100, 30), font, font_scale, (0, 255, 0), thickness)

        return frame

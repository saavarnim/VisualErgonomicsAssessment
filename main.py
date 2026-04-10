import cv2
import time

from core.pose import PoseEstimator
from core.biomechanics import PostureAnalyzer
from core.temporal import FatigueClassifier
from core.calibration import CalibrationState
from advanced.heatmap import HeatmapRenderer
from utils.display import Visualizer
from utils.notifications import PostureNotifier

def main():
    # Initialize Modules
    pose_estimator = PoseEstimator()
    posture_analyzer = PostureAnalyzer()
    fatigue_classifier = FatigueClassifier()
    calibration = CalibrationState()
    heatmap_renderer = HeatmapRenderer()
    visualizer = Visualizer()
    notifier = PostureNotifier(cooldown_seconds=60)
    
    # Open Webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # FPS and Timestamp tracking
    prev_time = time.time()
    system_start_time = time.time()
    last_timestamp_ms = -1
    
    print("Starting Visual Ergonomics System. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Optional: flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / max((current_time - prev_time), 0.001)
        prev_time = current_time
        
        timestamp_ms = int((current_time - system_start_time) * 1000)
        timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
        last_timestamp_ms = timestamp_ms

        # 1. Pose Estimation
        landmarks_dict, raw_results = pose_estimator.process(frame, timestamp_ms)
        keypoints = pose_estimator.extract_upper_body_keypoints(landmarks_dict)

        # 2. Biomechanics / Posture Analysis
        metrics = posture_analyzer.analyze(keypoints)

        # 3. Temporal Buffer & Classification
        state, strain_intensity = fatigue_classifier.update(metrics)

        # 4. Lean-in Warning
        lean_warning = calibration.check_lean_in(keypoints)

        # 5. Heatmap Rendering (Overlay on original frame)
        frame_with_heatmap = heatmap_renderer.draw_heatmap_overlay(frame.copy(), keypoints, strain_intensity)

        # 6. Desktop Notifications
        notifier.notify_bad_posture(state)

        # 7. Display / Visualization
        final_frame = visualizer.draw(frame_with_heatmap, raw_results, metrics, state, fps, calibration, lean_warning)

        # Show Output
        cv2.imshow("Visual Ergonomics Assessment", final_frame)

        # Quit on 'q', Calibrate on 'c'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            calibration.calibrate(landmarks_dict, keypoints)
            print(">>> Calibrated Ghost Posture")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import time

from core.pose import PoseEstimator
from core.biomechanics import PostureAnalyzer
from core.temporal import FatigueClassifier
from advanced.heatmap import HeatmapRenderer
from utils.display import Visualizer

def main():
    # Initialize Modules
    pose_estimator = PoseEstimator()
    posture_analyzer = PostureAnalyzer()
    fatigue_classifier = FatigueClassifier()
    heatmap_renderer = HeatmapRenderer()
    visualizer = Visualizer()
    
    # Open Webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # FPS tracking
    prev_time = time.time()
    system_start_time = time.time()
    
    print("Starting Visual Ergonomics System. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Optional: flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        
        timestamp_ms = int((current_time - system_start_time) * 1000)

        # 1. Pose Estimation
        landmarks_dict, raw_results = pose_estimator.process(frame, timestamp_ms)
        keypoints = pose_estimator.extract_upper_body_keypoints(landmarks_dict)

        # 2. Biomechanics / Posture Analysis
        metrics = posture_analyzer.analyze(keypoints)

        # 3. Temporal Buffer & Classification
        state, strain_intensity = fatigue_classifier.update(metrics)

        # 4. Heatmap Rendering (Overlay on original frame)
        frame_with_heatmap = heatmap_renderer.draw_heatmap_overlay(frame.copy(), keypoints, strain_intensity)

        # 5. Display / Visualization
        final_frame = visualizer.draw(frame_with_heatmap, raw_results, metrics, state, fps)

        # Show Output
        cv2.imshow("Visual Ergonomics Assessment", final_frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

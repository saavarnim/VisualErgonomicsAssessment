import cv2
import numpy as np

class HeatmapRenderer:
    def __init__(self, width=640, height=480):
        # We can keep an accumulating blank canvas if we want motion trails, 
        # but stateless rendering is simpler and fully responsive.
        self.width = width
        self.height = height

    def draw_heatmap_overlay(self, frame, keypoints, strain_intensity):
        """
        Renders a glowing heatmap over neck and shoulders representing strain.
        strain_intensity: [0.0, 1.0] scale.
        """
        if strain_intensity <= 0.01 or not keypoints:
            return frame

        # Create a black image
        heatmap_canvas = np.zeros_like(frame, dtype=np.uint8)

        # Decide color based on intensity (yellow to red)
        # BGR format: Red is (0, 0, 255), Yellow is (0, 255, 255)
        # If intensity is high, red dominates. Else yellow.
        b = 0
        g = int(255 * (1 - strain_intensity))
        r = 255
        color = (b, g, r)

        # Keypoints to highlight: Neck (between nose and shoulders), and shoulders themselves
        pts_to_highlight = []
        
        nose = keypoints.get('nose')
        l_sh = keypoints.get('left_shoulder')
        r_sh = keypoints.get('right_shoulder')

        h, w, _ = frame.shape

        if l_sh and l_sh['visibility'] > 0.5:
            pts_to_highlight.append((int(l_sh['x'] * w), int(l_sh['y'] * h)))
        if r_sh and r_sh['visibility'] > 0.5:
            pts_to_highlight.append((int(r_sh['x'] * w), int(r_sh['y'] * h)))
            
        if nose and l_sh and r_sh:
             # approximation of upper neck/cervical spine
             mid_x = int(((l_sh['x'] + r_sh['x']) / 2.0 * w + nose['x'] * w) / 2.0)
             mid_y = int(((l_sh['y'] + r_sh['y']) / 2.0 * h + nose['y'] * h) / 2.0)
             pts_to_highlight.append((mid_x, mid_y))

        # Size of the blobs
        radius = int(50 + (30 * strain_intensity))

        for pt in pts_to_highlight:
            cv2.circle(heatmap_canvas, pt, radius, color, -1)

        # Blur heavily to create heatmap "glow" effect
        # The kernel size must be odd
        ksize = 101
        heatmap_canvas = cv2.GaussianBlur(heatmap_canvas, (ksize, ksize), 0)

        # Alpha blend the heatmap over the original frame
        # Make alpha proportional to strain intensity (max 0.6 opacity)
        alpha = 0.6 * strain_intensity
        blended = cv2.addWeighted(heatmap_canvas, alpha, frame, 1.0, 0)
        
        return blended

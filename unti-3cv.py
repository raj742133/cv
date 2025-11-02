# ============================================================
# ADVANCED REAL-TIME COIN DETECTION WITH VIEWFINDER (OpenCV)
# ============================================================
# Detects and counts coins with enhanced preprocessing,
# duplicate filtering, and adaptive tuning.
# ============================================================

import cv2
import numpy as np
import time


def main():
    # --- PARAMETERS ---
    dp = 1.2
    min_dist = 80
    min_radius = 55
    max_radius = 130
    param1 = 120
    param2 = 45  # Higher = stricter detection (less noise)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("🎥 Starting enhanced coin detection... Press 'q' to quit.")

    total_detected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # --- PREPROCESSING ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 2)
        gray = cv2.equalizeHist(gray)  # Contrast normalization

        # Apply adaptive thresholding to improve edge detection
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
        )

        # Combine edges and threshold
        edges = cv2.Canny(gray, 100, 180)
        combined = cv2.bitwise_or(edges, adaptive)

        # --- CIRCLE DETECTION ---
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        count = 0
        filtered_circles = []

        if circles is not None:
            circles = np.uint16(np.around(circles[0]))

            # Filter duplicates (if circles overlap heavily)
            for (x, y, r) in circles:
                if all(np.hypot(x - cx, y - cy) > r * 0.75 for (cx, cy, cr) in filtered_circles):
                    filtered_circles.append((x, y, r))

            count = len(filtered_circles)
            total_detected = count

            # Draw circles
            for (x, y, r) in filtered_circles:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 3)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

        # --- VIEWFINDER OVERLAY ---
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.line(overlay, (w//2 - 20, h//2), (w//2 + 20, h//2), (255, 255, 255), 2)
        cv2.line(overlay, (w//2, h//2 - 20), (w//2, h//2 + 20), (255, 255, 255), 2)
        cv2.rectangle(overlay, (10, 10), (w - 10, h - 10), (0, 255, 0), 2)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

        # --- INFO DISPLAY ---
        fps = 1.0 / (time.time() - start_time)
        color = (0, 255, 0) if count > 0 else (0, 0, 255)

        cv2.putText(frame, f"Coins Detected: {total_detected}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'Q' to Quit", (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # --- DISPLAY LIVE VIEW ---
        cv2.imshow("🪙 Enhanced Coin Detector | Live Viewfinder", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Stream ended. Final detected coins: {total_detected}")


if __name__ == "__main__":
    main()

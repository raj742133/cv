# ===============================================================
# REAL-TIME EDGE DETECTOR COMPARISON (OpenCV + NumPy)
# ===============================================================
# Live webcam version of your static comparison code.
# Shows:
# - Original frame
# - Sobel
# - Prewitt
# - Laplacian
# - Canny
# Each view labeled with processing time.
# ===============================================================

import cv2
import numpy as np
import time

def compute_edges(frame):
    """Compute multiple edge detections on a single frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    results = {}

    # --- Sobel ---
    start = time.time()
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    results["Sobel"] = (sobel, round(time.time() - start, 4))

    # --- Prewitt ---
    start = time.time()
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewittx = cv2.filter2D(blur, -1, kernelx)
    prewitty = cv2.filter2D(blur, -1, kernely)
    prewitt = cv2.magnitude(prewittx.astype(float), prewitty.astype(float))
    results["Prewitt"] = (prewitt, round(time.time() - start, 4))

    # --- Laplacian ---
    start = time.time()
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    results["Laplacian"] = (laplacian, round(time.time() - start, 4))

    # --- Canny ---
    start = time.time()
    canny = cv2.Canny(blur, 100, 200)
    results["Canny"] = (canny, round(time.time() - start, 4))

    return gray, results

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("🎥 Real-time Edge Detection Comparison started!")
    print("Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        gray, results = compute_edges(frame)

        # Normalize float images to 8-bit for display
        def normalize(img):
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            return img.astype(np.uint8)

        # Create panels
        target_w, target_h = 480, 360
        views = [
            cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (target_w, target_h))
        ]
        names = ["Original"]

        for name, (output, t) in results.items():
            view = normalize(output)
            view = cv2.resize(view, (target_w, target_h))
            view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
            cv2.putText(view, f"{name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 3)
            cv2.putText(view, f"Time: {t}s", (20, target_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            views.append(view)
            names.append(name)

        # Combine evenly into 2 rows
        top = np.hstack((views[0], views[1]))
        bottom = np.hstack((views[2], views[3]))
        # Canny as extra column (centered if fewer columns)
        canny_view = views[4]
        blank = np.zeros_like(canny_view)
        bottom_row = np.hstack((canny_view, blank))

        combined = np.vstack((top, bottom_row))

        cv2.putText(combined, "🧠 Real-Time Edge Detector Comparison",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(combined, "Press 'Q' to Quit",
                    (20, combined.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("🧠 Real-Time Edge Detector Comparison", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Visualization ended.")

if __name__ == "__main__":
    main()

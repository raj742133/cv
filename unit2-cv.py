# ===============================================================
# REAL-TIME DISTANCE TRANSFORM VISUALIZATION (OpenCV + NumPy)
# ===============================================================
# Displays 4 evenly sized panels:
# - Original Frame
# - Binary Mask
# - Distance Transform
# - Skeleton View
# Each labeled clearly with total maxima count.
# ===============================================================

import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology

# ---------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------

def find_local_maxima(dist_transform):
    """Find local maxima points in the distance map."""
    local_max = ndimage.maximum_filter(dist_transform, size=5)
    maxima = (dist_transform == local_max) & (dist_transform > 0)
    coords = np.argwhere(maxima)
    return coords

def compute_skeleton(binary_img):
    """Compute morphological skeleton."""
    skeleton = morphology.skeletonize(binary_img > 0)
    return skeleton.astype(np.uint8) * 255

def compute_distance_transforms(binary_img):
    """Compute different types of distance transforms."""
    dist_euclidean = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    dist_manhattan = cv2.distanceTransform(binary_img, cv2.DIST_L1, 3)
    dist_chessboard = cv2.distanceTransform(binary_img, cv2.DIST_C, 3)
    return {
        'Euclidean': dist_euclidean,
        'Manhattan': dist_manhattan,
        'Chessboard': dist_chessboard
    }

# ---------------------------------------------------------------
# Real-Time Processing Loop
# ---------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("🎥 Real-time Distance Transform Visualization started!")
    print("Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        # --- Step 1: Preprocess ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        binary_img = cv2.adaptiveThreshold(
            gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 5
        )

        # --- Step 2: Distance Transform ---
        dist_transforms = compute_distance_transforms(binary_img)
        dist_chessboard = dist_transforms["Chessboard"]

        dist_display = cv2.normalize(dist_chessboard, None, 0, 255, cv2.NORM_MINMAX)
        dist_display = dist_display.astype(np.uint8)
        dist_colored = cv2.applyColorMap(dist_display, cv2.COLORMAP_JET)

        # --- Step 3: Skeleton + Local Maxima ---
        skeleton = compute_skeleton(binary_img)
        coords = find_local_maxima(dist_chessboard)
        count = len(coords)
        for (y, x) in coords:
            cv2.circle(dist_colored, (x, y), 3, (255, 255, 255), -1)

        # --- Step 4: Resize each view evenly ---
        target_w, target_h = 480, 360  # each quadrant same size
        views = [
            cv2.resize(frame, (target_w, target_h)),
            cv2.resize(cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR), (target_w, target_h)),
            cv2.resize(dist_colored, (target_w, target_h)),
            cv2.resize(cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR), (target_w, target_h))
        ]

        # --- Step 5: Combine evenly ---
        top = np.hstack((views[0], views[1]))
        bottom = np.hstack((views[2], views[3]))
        combined = np.vstack((top, bottom))

        # --- Step 6: Labels ---
        cv2.putText(combined, f"Local Maxima (Centers): {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(combined, "Press 'Q' to Quit", (20, combined.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        h, w = target_h, target_w
        # Top row labels
        cv2.putText(combined, "ORIGINAL FRAME", (int(w * 0.25) - 80, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        cv2.putText(combined, "BINARY MASK", (int(w * 1.25) - 80, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        # Bottom row labels
        cv2.putText(combined, "DISTANCE TRANSFORM", (int(w * 0.25) - 120, h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        cv2.putText(combined, "SKELETON VIEW", (int(w * 1.25) - 100, h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

        # --- Step 7: Show ---
        cv2.imshow("🧠 Real-Time Distance Transform Visualization", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Visualization ended.")


if __name__ == "__main__":
    main()

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------
# Configuration
# ----------------------------
# Load your YOLO model
model = YOLO('Model/fall-detection.pt')

# To use an IP camera, set IP_CAM to your RTSP URL.
# Example: CAM = "rtsp://username:password@192.168.x.x:554/stream"
#
# To use a local webcam, set CAM = 0 (or another integer for different cameras).
CAM = 0

# Define class names
class_names = ["falling", "sitting", "standing", "walking"]

# Initial polygon zone (x, y) coordinates of vertices
polygon_zone = np.array([
    [8, 7],
    [29, 446],
    [282, 444],
    [277, 3]
], dtype=np.int32)

# Interaction variables
dragging = False
selected_vertex = -1
radius = 10  # Radius for selecting vertices by mouse click


def mouse_callback(event, x, y, flags, param):
    global polygon_zone, dragging, selected_vertex

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check which vertex is closest to the click point
        distances = [np.linalg.norm((x - vx, y - vy)) for (vx, vy) in polygon_zone]
        closest_idx = np.argmin(distances)

        # If the click is close enough to a vertex, start dragging it
        if distances[closest_idx] < radius:
            dragging = True
            selected_vertex = closest_idx

    elif event == cv2.EVENT_MOUSEMOVE:
        # If currently dragging a vertex, update its coordinates
        if dragging and selected_vertex != -1:
            polygon_zone[selected_vertex] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        # Stop dragging when mouse button is released
        dragging = False
        selected_vertex = -1


def is_inside_polygon(polygon, bbox):
    """
    Check if the object defined by a bounding box is inside the polygon.
    We test the four corners and the center of the bounding box.
    If any point lies inside (pointPolygonTest >= 0), return True.
    """
    x1, y1, x2, y2 = bbox
    points_to_check = [
        (x1, y1),  # top-left
        (x2, y1),  # top-right
        (x1, y2),  # bottom-left
        (x2, y2),  # bottom-right
        ((x1 + x2) // 2, (y1 + y2) // 2)  # center
    ]

    polygon_contour = polygon.reshape((-1, 1, 2)).astype(np.int32)

    for (px, py) in points_to_check:
        if cv2.pointPolygonTest(polygon_contour, (px, py), False) >= 0:
            return True
    return False


def detect_fall(frame):
    # Perform inference on the frame
    results = model(frame)
    fall_detected = False

    # Loop through detections
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            score = box.conf.item()

            # Filter out low-confidence detections
            if score < 0.50:
                continue

            # Get bounding box coordinates
            bbox = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, bbox)

            # Check if inside polygon
            if is_inside_polygon(polygon_zone, (x1, y1, x2, y2)):
                # If inside polygon, draw differently
                if class_id == 0:
                    class_name = "Sleeping"
                    color = (255, 0, 255)
                else:
                    class_name = class_names[class_id]
                    color = (0, 255, 0) if class_id == 2 else (255, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} ({score:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, color, 2)
                continue

            # If outside polygon, handle normally
            class_name = class_names[class_id]
            if class_id == 0:  # Falling
                color = (0, 0, 255)
                fall_detected = True
            elif class_id == 1:  # Sitting
                color = (255, 255, 0)
            elif class_id == 2:  # Standing
                color = (0, 255, 0)
            else:  # Walking
                color = (255, 0, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} ({score:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2)

    # Alert if fall detected outside polygon
    if fall_detected:
        cv2.putText(frame, "Fall detected!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame


def main():
    cap = cv2.VideoCapture(CAM)
    if not cap.isOpened():
        print("Error: Unable to connect to the camera")
        return

    cv2.namedWindow("Fall Detection")
    cv2.setMouseCallback("Fall Detection", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read the frame")
            break

        # Detect falls in the frame
        annotated_frame = detect_fall(frame)

        # Draw the polygon zone
        cv2.polylines(annotated_frame, [polygon_zone.reshape((-1, 1, 2))],
                      isClosed=True, color=(51, 255, 51), thickness=2)

        # Draw vertices for easier grabbing
        for (vx, vy) in polygon_zone:
            cv2.circle(annotated_frame, (vx, vy), radius, (0, 0, 255), -1)

        cv2.imshow("Fall Detection", annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

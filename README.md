# Interactive-Polygon-ROI-Editor-for-Real-Time-YOLO-Fall-Detection

This project demonstrates how to integrate a YOLO-based object detection model with an interactive polygonal region of interest (ROI) editor. Users can dynamically adjust a polygonal zone on a live video feed using simple click-and-drag mouse interactions. Objects inside or outside the polygon can be processed differently. 

**Key Features:**
- **Real-Time Detection:** Continuously processes frames from an RTSP camera feed or a local webcam using a YOLO model.
- **Interactive Polygon Editing:** Click and drag the polygon’s vertices directly on the displayed video to reshape the zone without stopping the system.
- **Customizable Behavior:** Easily modify the code to change the classes, colors, detection logic, or polygon shape.

This approach is particularly useful for tasks like fall detection in a specific area of interest. Users can fine-tune the polygonal region interactively to ensure accurate and context-relevant detections.

## How It Works

1. **Loading the Model and Video Stream:**
   - **RTSP Camera:** Set `CAM` to your camera’s RTSP URL.
   - **Local Webcam:** Set `CAM` to `0` (or another integer) to use your default webcam.

2. **Polygonal ROI Definition:**
   A polygon (`polygon_zone`) defines a region of interest on the video feed. Detected objects inside this region can be treated differently than those outside.

3. **Mouse Interaction:**
   - Move your mouse over a vertex of the polygon and click to "grab" it.
   - Drag the vertex to reposition it.
   - Release the mouse to fix the vertex at the new location.
   - The updated polygon is used immediately in subsequent detection frames.

4. **Object Detection and Polygon Testing:**
   The YOLO model detects objects. For each detection:
   - The bounding box is tested against the polygon to determine if it lies inside.
   - Objects inside can be highlighted or processed differently.
   - If the class is "falling" and detected outside the polygon, a "Fall detected!" alert is displayed.

## Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- [Ultralytics YOLO](https://docs.ultralytics.com/) (`pip install ultralytics`)
- A `fall-detection.pt` YOLO model file.
- An IP camera or a local webcam.

## Usage

1. Clone this repository 
2. Update the `CAM` variable in `run.py`:
   - Use your RTSP URL for an IP camera, e.g.:  
     ```python
     CAM = "rtsp://username:password@your_camera_ip:port/path"
     ```
   - Use an integer (e.g., `0`) for your local webcam:  
     ```python
     CAM = 0
     ```
3. Run the code:
   ```bash
   python main.py

4. A window will appear showing the live video with the polygon drawn.
5. Click and drag the polygon vertices to adjust the ROI in real-time.
6. Press q to quit

## Code Explanation

1. detect_fall(): Runs YOLO inference on the frame and draws bounding boxes and labels for detected objects.
2. is_inside_polygon(): Checks if a bounding box lies inside the polygon zone.
3. mouse_callback(): Handles mouse clicks and movements to allow vertex dragging.
4. main(): Captures frames from the camera (IP or webcam), calls detect_fall(), and updates the display.
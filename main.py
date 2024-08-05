import cv2
import torch
from imutils.video import VideoStream
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use 'yolov5s'

# Initialize video stream
vs = VideoStream(src=0).start()  # Change src to 1 or other if using an external webcam
time.sleep(2.0)  # Allow camera sensor to warm up

# Known reference values (example values)
KNOWN_HEIGHT = 170.0    # Known diagonal size of a cell phone (xioami 11T)

def calculate_height(box, reference_pixel):
    # box[1] is the top-left corner's y-coordinate, box[3] is the bottom-right corner's y-coordinate
    pixel_height = box[3] - box[1]
    height = (KNOWN_HEIGHT * pixel_height) / reference_pixel
    return height

def calculate_phone_pixel(box):
    # box[1] is the top-left corner's y-coordinate, box[3] is the bottom-right corner's y-coordinate
    # box[0] is the top-left corner's x-coordinate, box[2] is the bottom-right corner's x-coordinate
    return max(box[3] - box[1], box[2] - box[0])

while True:
    frame = vs.read()
    if frame is None:
        break

    reference_pixel = 0

    # Perform object detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Extract detections

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if cls == 67:   # Class 67 is 'cell phone' in COCO dataset
            reference_pixel = calculate_phone_pixel(detection)
        if cls == 0 and reference_pixel > 0:  # Class 0 is 'person' in COCO dataset
            # Calculate height
            height = calculate_height(detection, reference_pixel)
            label = f"Height: {height:.2f} mm"

            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream
vs.stop()
cv2.destroyAllWindows()

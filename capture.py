from ultralytics import YOLO

model = YOLO("yolo11n.pt")

import cv2

# Open webcam feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform object detection
    results = model(frame)

    # Render the results on the webcam feed
    annotated_frame = results.render()[0]

    # Show the processed frame
    cv2.imshow('YOLO Object Detection', annotated_frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()

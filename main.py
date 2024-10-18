import cv2
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the RTSP URL from environment variables
rtsp_url = os.getenv("RTSP_URL")

# Ensure RTSP_URL is provided
if not rtsp_url:
    print("RTSP URL not found. Please set RTSP_URL in your .env file.")
    exit()
    
# Get ROI coordinates from environment variables
roi_x1 = int(os.getenv("ROI_X1", 0))
roi_y1 = int(os.getenv("ROI_Y1", 0))
roi_x2 = int(os.getenv("ROI_X2", 0))
roi_y2 = int(os.getenv("ROI_Y2", 0))

# Ensure ROI is correctly defined
if roi_x1 == 0 and roi_y1 == 0 and roi_x2 == 0 and roi_y2 == 0:
    print("ROI coordinates not found. Please set ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 in your .env file.")
    exit()

""" 
Function to retrieve video feed from the specified URL and perform object detection using YOLO model to detect if a person is present within a specified Region of Interest (ROI).
"""
def get_stream(url):
    # Load YOLO model from weights and config file
    try:
        net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Load the class names from coco.names
    try:
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        print("Class names loaded successfully.")
    except FileNotFoundError:
        print("coco.names file not found.")
        return

    # Initialize video stream
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Error opening video stream")
        return 

    # Define the Region of Interest (ROI)
    roi_x1, roi_y1, roi_x2, roi_y2 = 118, 6, 218, 206

    frame_count = 0
    prev_person_detected = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from video stream")
            break

        frame_count += 1

        # Prepare the frame for YOLO model
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        # Get output layer names
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        # Perform forward pass to get detections
        detections = net.forward(output_layers)

        # Flag to check if a person is detected in the ROI
        person_detected = False

        # Process the detections
        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Filter for person class with confidence > 0.5
                if confidence > 0.5 and classes[class_id] == "person":
                    # Get bounding box coordinates
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    # Calculate the top-left and bottom-right coordinates of the bounding box
                    x1 = int(center_x - w / 2)
                    y1 = int(center_y - h / 2)
                    x2 = int(center_x + w / 2)
                    y2 = int(center_y + h / 2)

                    # Debugging: Print bounding box coordinates
                 
                    # Draw the bounding box on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"person: {round(confidence, 2)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Check if any part of the bounding box lies within the ROI (relaxed condition)
                    if (x2 > roi_x1 and x1 < roi_x2 and y2 > roi_y1 and y1 < roi_y2):
                        person_detected = True
                        break
            if person_detected:
                break

        # Draw the ROI rectangle on the original frame
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

        # Display whether a person is present or missing in the ROI only when the state changes
        if frame_count % 30 == 0:
            if person_detected:
                print("Person present in ROI")
            else:
                print("Person missing in ROI")


        # Stream display
        cv2.imshow('Camera Stream', frame)

        # Press 'q' to quit the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("quit video stream")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace this with your actual RTSP URL
    get_stream(rtsp_url)

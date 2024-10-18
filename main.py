import cv2
import numpy as np

""" 
Function to retrieve video feed from the specified URL and perform object detection using YOLO model.
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

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from video stream")
            break

        # Prepare the frame for YOLO model
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        # Get output layer names
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        # Perform forward pass to get detections
        detections = net.forward(output_layers)

        # Process the detections
        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Filter for objects with confidence > 0.5
                if confidence > 0.5:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    # Calculate coordinates for the bounding box
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Draw the bounding box and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{classes[class_id]}: {round(confidence, 2)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Stream display
        cv2.imshow('Camera Stream with YOLO Detection', frame)

        # Press 'q' to quit the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("quit video stream")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace this with your actual RTSP URL
    rtsp_url = "rtsp://admin:EIYTCA@192.168.1.5:554/Streaming/Channels/102"
    get_stream(rtsp_url)

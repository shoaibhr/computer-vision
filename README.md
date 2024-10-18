# YOLO Person Detection in RTSP Stream

This project uses the YOLOv4-tiny model to detect if a person is present within a specified Region of Interest (ROI) in an RTSP video stream. The YOLO (You Only Look Once) model is used for object detection, and the application runs on a video feed from an IP camera or another RTSP source.

## Features
- Detects persons in a video stream using the YOLOv4-tiny model.
- Checks if a detected person falls within a specified Region of Interest (ROI).
- Displays the video stream with bounding boxes drawn around detected persons.
- Uses a `.env` file to keep sensitive data, such as RTSP URLs, secure.

## Getting Started

### Prerequisites
- Python 3.6 or higher
- OpenCV (`cv2`) for video capture and processing
- `numpy` for numerical operations
- `python-dotenv` to manage environment variables

### Installation
1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required Python packages:
   ```sh
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root of your project folder and add the following configuration:
   ```env
   RTSP_URL=rtsp://username:password@192.168.x.x:554/Streaming/Channels/102
   ROI_X1=118
   ROI_Y1=6
   ROI_X2=218
   ROI_Y2=206
   ```
   - Replace the `RTSP_URL` with your camera's RTSP URL.
   - Set the ROI coordinates as per your requirements.

### Usage
1. Run the script to start detecting:
   ```sh
   python main.py
   ```

2. The script will display the camera stream with bounding boxes for detected persons. It will also indicate if a person is present in the specified ROI every 30 frames.

### Configuring the Region of Interest (ROI)
You can configure the **ROI** by modifying the values in the `.env` file. The ROI is defined by two points:
- **ROI_X1, ROI_Y1**: Top-left corner of the ROI
- **ROI_X2, ROI_Y2**: Bottom-right corner of the ROI

The coordinates should be adjusted to target the specific area in the video where you want to detect the presence of a person.

### Adding to `.gitignore`
To prevent sensitive information from being pushed to GitHub, make sure your `.env` file is included in `.gitignore`:
```
.env
```

## Notes
- This project uses the **YOLOv4-tiny** model for faster performance with acceptable accuracy.
- Make sure the `yolov4-tiny.weights`, `yolov4-tiny.cfg`, and `coco.names` files are available in the project directory for the model to work correctly.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- [YOLOv4-tiny](https://github.com/AlexeyAB/darknet) for the pre-trained object detection model.
- [OpenCV](https://opencv.org/) for image processing.
- [dotenv](https://pypi.org/project/python-dotenv/) for managing environment variables.

## Troubleshooting
- **Video Stream Not Opening**: Make sure your RTSP URL is correct and that your IP camera is accessible from the network.
- **Model Not Loading**: Ensure that `yolov4-tiny.weights` and `yolov4-tiny.cfg` are correctly placed in your project directory.
- **Error Reading .env File**: Make sure the `.env` file is correctly formatted and located in the root of your project folder.


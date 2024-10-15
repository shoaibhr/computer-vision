import cv2

""" 
Function to retrieve video feed from the specified URL. """
def get_stream(url):
    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        print("Error opening video stream")
        return 
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error reading frame from video stream")
            break
        
        #stream display
        cv2.imshow('Camera Stream', frame)
        
        #press 'q'to quit the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # Replace this with your actual RTSP URL
    rtsp_url = "rtsp://admin:EIYTCA@192.168.1.18:554/Streaming/Channels/102"

    get_stream(rtsp_url)
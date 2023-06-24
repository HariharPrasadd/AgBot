import cv2
import numpy as np

# Load the YOLOv4 model architecture and weights
net = cv2.dnn.readNetFromDarknet("C:\\Users\\Admin\\Desktop\\Object Detection\\yolov4.cfg", "C:\\Users\\Admin\\Desktop\\Object Detection\\yolov4.weights")
# Set backend and target to utilize OpenCV's DNN module
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # You can also use DNN_TARGET_CUDA if you have a compatible GPU
# Read an example frame from an image or video source
frame = cv2.imread("C:\\Users\\Admin\\Desktop\\yolotest2.jpg")

# Preprocess the frame
blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Set the input to the network
net.setInput(blob)

# Perform forward pass and get output
outs = net.forward()
# Define a confidence threshold
confidence_threshold = 0.5

# Iterate through each output detection
# Initialize variables to store the highest confidence detection
max_confidence = 0.0
best_detection = None

# Iterate through each output detection
for detection in outs:
    # Extract information from the detection
    scores = detection[5:]
    class_id = np.argmax(scores)
    confidence = scores[class_id]

    # Filter detections based on confidence threshold
    if confidence > confidence_threshold:
        # Check if this detection has higher confidence than the previous best detection
        if confidence > max_confidence:
            max_confidence = confidence
            best_detection = detection

# If a detection with sufficient confidence was found
if best_detection is not None:
    # Extract information from the best detection
    scores = best_detection[5:]
    class_id = np.argmax(scores)
    confidence = scores[class_id]

    # Calculate bounding box coordinates
    box = best_detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
    (center_x, center_y, width, height) = box.astype("int")
    x = int(center_x - (width / 2))
    y = int(center_y - (height / 2))

    # Draw bounding box and label on the frame
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(frame, "Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.imshow("Object Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


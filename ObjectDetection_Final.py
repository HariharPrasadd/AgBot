import cv2
import numpy as np

#declaring video urls
url = 'http://10.1.45.87:8081'
capture = cv2.VideoCapture(0)

#establishing backend connection and target device with conf file and weights
net = cv2.dnn.readNetFromDarknet("C:\\Users\\Admin\\Desktop\\Object Detection\\yolov4.cfg", "C:\\Users\\Admin\\Desktop\\Object Detection\\yolov4.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) 

while True:

    ret, frame = capture.read()

    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward()

    with open("C:\\Users\\Admin\\Documents\\AgBot\\coco.names.txt", "r") as f:
        class_labels = [line.strip() for line in f.readlines()]    

    confidence_threshold = 0.5
    max_confidence = 0.0
    best_detection = None

    for detection in outs:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > confidence_threshold:
            if confidence > max_confidence:
                max_confidence = confidence
                best_detection = detection

    if best_detection is not None:
        scores = best_detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        #best_detection[0:4] contains x_min, y_min, x_max, y_max of the best detection
        box = best_detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
        (center_x, center_y, width, height) = box.astype("int")
        x = int(center_x - (width / 2))
        y = int(center_y - (height / 2))

        #rectangle accepts image input, top left coordinates, bottom right coordinates, box color, line thickness
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        #text accepts image input, text to display, coordinates(y-10 goes up because in image
        # processing the top has the least y value), the font, the text size in relation to image
        #text color, and line thickness
        cv2.putText(frame, class_labels[class_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.line(frame, (center_x, 0), (center_x, frame.shape[0]), (255,255,0), 2)
        cv2.line(frame, (0, center_y), (frame.shape[1], center_y), (255,255,0), 2)
    cv2.imshow('Object Detection', frame)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
capture.release()
cv2.destroyAllWindows()


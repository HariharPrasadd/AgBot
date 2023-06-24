import cv2
import numpy as np

url = 'http://10.1.45.59:8081'
capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    if not ret:
        break
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    green_pixels = np.sum(green_mask == 255)
    total_pixels = green_mask.size
    percentage_green_hsv = green_pixels/total_pixels*100

    print(f"Percentage of hsv green pixels: {percentage_green_hsv:.1f}%")
    
    cv2.imshow('Mask', green_mask)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()



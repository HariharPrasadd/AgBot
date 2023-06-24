from PIL import Image
import numpy as np
import cv2

#image = Image.open("C:\\Users\\Admin\\Desktop\\Wallpaperbruh.jpg")
#image_array = np.array(image)
image = cv2.imread("C:\\Users\\Admin\\Desktop\\Wallpaperbruh.jpg")

#Normal version of numpy image processing
"""height, width, channels = image_array.shape
greenpixels = np.sum(image_array[:,:,1] > 200)
totalpixels = height * width"""

#Calculating normal green percentage
#greenpercentagenormal = (greenpixels/totalpixels)*100

#image processing using hsv
lower_green = np.array([30, 25, 25])
upper_green = np.array([80, 255, 255])
hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
green_pixels = np.sum(green_mask == 255)
total_pixels = green_mask.size

greenpercentagehsv = (green_pixels/total_pixels)*100

print(f"The percentage of green pixels is: {greenpercentagehsv:.1f}%")
#print(f"The percentage of normal green pixels is: {greenpercentagenormal:.1f}%")
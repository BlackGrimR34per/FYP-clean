import cv2
from PIL import Image

image = cv2.imread("thresholded_image.png")
reversed_image = cv2.bitwise_not(image)

cv2.imwrite("Reversed.png", reversed_image)
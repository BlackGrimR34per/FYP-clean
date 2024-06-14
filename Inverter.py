import cv2
from PIL import Image

image_path = "./cropped images/IC_0.png"
image = cv2.imread(image_path)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

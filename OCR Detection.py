import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

image = cv2.imread("Entropy 2.png")
reversed_image = cv2.bitwise_not(image)

result = pytesseract.image_to_string(image, config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7")

print(result)

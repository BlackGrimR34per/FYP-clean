from matplotlib import pyplot as plt
import cv2
import os
import pytesseract

# picture_dir = "./cropped_texts/"

# for filename in os.listdir(picture_dir):
#     if filename.endswith(".png"):
#         image_path = os.path.join(picture_dir, filename)
#
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#         blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)
#         text = pytesseract.image_to_string(blurred_image)
#         print(text)

image = cv2.imread('./cropped_texts/cropped_text_8.png')

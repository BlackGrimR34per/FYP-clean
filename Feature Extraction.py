import numpy as np
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

image = cv2.imread('./cropped_texts/cropped_text_12.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
blurred_binary_image = cv2.GaussianBlur(binary_image, (5, 5), 0)

plt.imshow(blurred_binary_image, cmap='gray')
plt.show()

vertical_projection = np.sum(blurred_binary_image, axis=0)

plt.plot(vertical_projection)
plt.show()

threshold = np.max(vertical_projection)
start = 0
characters = []

for i in range(len(vertical_projection)):
    if vertical_projection[i] < threshold:
        if start == 0:
            start = i
    else:
        if start != 0:
            end = i
            characters.append((start, end))
            start = 0
if start != 0:
    characters.append((start, len(vertical_projection)))

for i, (start, end) in enumerate(characters):
    character_image = binary_image[:, start:end]
    plt.imshow(character_image, cmap='gray')
    plt.show()



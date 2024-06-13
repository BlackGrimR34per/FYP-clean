import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from skimage.filters import threshold_sauvola

# Load the image
image = cv2.imread("Entropy.png", cv2.IMREAD_GRAYSCALE)

# Apply Sauvola thresholding
sauvola_thresh = threshold_sauvola(image, window_size=61)
sauvola_binary = image > sauvola_thresh

# # Plot the original and Sauvola thresholded images
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(image, cmap='gray')
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.title("Sauvola Thresholding")
# plt.imshow(sauvola_binary, cmap='gray')
# plt.axis('off')
#
# plt.show()

cv2.imwrite("sauvola_binary.png", image * sauvola_binary)
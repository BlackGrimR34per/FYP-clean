import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_niblack

# Load the image
image = cv2.imread("Entropy.png", cv2.IMREAD_GRAYSCALE)

# Apply Niblack thresholding
niblack_thresh = threshold_niblack(image, window_size=61, k=0.8)
niblack_binary = image > niblack_thresh

# Plot the original and Niblack thresholded images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Niblack Thresholding")
plt.imshow(niblack_binary, cmap='gray')
plt.axis('off')

plt.show()

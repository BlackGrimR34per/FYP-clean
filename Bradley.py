import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_local

# Load the image
image = cv2.imread("Entropy.png", cv2.IMREAD_GRAYSCALE)

# Apply Bradley adaptive thresholding
window_size = 71
bradley_thresh = threshold_local(image, window_size, offset=10, method='mean')
bradley_binary = image > bradley_thresh

# Plot the original and Bradley thresholded images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Bradley Thresholding")
plt.imshow(bradley_binary, cmap='gray')
plt.axis('off')

plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.io import imread, imshow

# Load the image
image = cv2.imread("./cropped images/IC_0.png", cv2.IMREAD_GRAYSCALE)

# Define a structuring element for entropy filtering
selem = disk(5)

# Apply entropy filtering
entropy_image = entropy(image, selem)
scaled_entropy = entropy_image / entropy_image.max()

mask = scaled_entropy > 0.3
plt.figure(num=None, figsize=(8, 6), dpi=80)
reverse_mask = 1 - mask
new_image = image * reverse_mask
cv2.imwrite("Entropy.png", new_image)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import threshold_otsu
from skimage.io import imread, imshow
from skimage.util import img_as_ubyte

# Load the image
image_path = "./cropped images/IC_0.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.bitwise_not(image)

# Define a structuring element for entropy filtering
selem = disk(5)

# Disk value can be adjusted
scaled_entropy = image / image.max()
entropy_image = entropy(scaled_entropy, selem)
scaled_entropy = entropy_image / entropy_image.max()
mask = scaled_entropy > 0.8

filtered_image = image * mask
cv2.imshow(filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

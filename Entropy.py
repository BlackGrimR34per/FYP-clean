import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.io import imread, imshow

# Load the image
image = cv2.imread("./cropped images/IC_0.png", cv2.IMREAD_GRAYSCALE)

# Define a structuring element for entropy filtering
# selem = disk(5)

# Apply entropy filtering
# entropy_image = entropy(image, selem)
# scaled_entropy = entropy_image / entropy_image.max()

def disk_iterations(image):
    f_size = 20
    radi = list(range(1,10))
    fig, ax = plt.subplots(3,3,figsize=(15,15))
    for n, ax in enumerate(ax.flatten()):
        ax.set_title(f'Radius at {radi[n]}', fontsize = f_size)
        ax.imshow(entropy(image, disk(radi[n])), cmap =
                  'magma');
        ax.set_axis_off()
    fig.tight_layout()
    plt.show()

disk_iterations(image)
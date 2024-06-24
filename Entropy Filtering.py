import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pytesseract

image_path = "./cropped images/IC.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image = cv2.blur(image,(5, 5))
cv2.imshow("Original", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

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

def threshold_checker(image):
    thresholds = np.arange(0.1, 1.1, 0.1)
    entropy_image = entropy(image, disk(4))
    scaled_entropy = entropy_image / entropy_image.max()
    fig, ax = plt.subplots(2, 5, figsize=(17, 10))
    for n, ax in enumerate(ax.flatten()):
        ax.set_title(f'Threshold  : {round(thresholds[n], 2)}',
                     fontsize=16)
        threshold = scaled_entropy > thresholds[n]
        ax.imshow(threshold, cmap='gist_stern_r');
        ax.axis('off')
    fig.tight_layout()
    plt.show()


# disk_iterations(image)
# threshold_checker(image)

# scaled_entropy = image / image.max()
# entropy_image = entropy(image, disk(4))
# scaled_entropy = entropy_image / entropy_image.max()
# mask = scaled_entropy > 0.8
#
# filtered_image = (1 - mask) * image
# plt.imshow(filtered_image, cmap='gray')
# plt.show()




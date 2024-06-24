import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from pytesseract import pytesseract
from skimage.filters.rank import entropy
from skimage.morphology import disk

# Function to display images
def display_image(image, title, cmap='gray'):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Step (a): Load the original input image
image_path = "./cropped images/IC_1.png"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
display_image(original_image, "Original Input Image")

original_image = cv2.blur(original_image, (4, 4))

# Step (b): Local entropy map
entropy_image = entropy(original_image, disk(5))
# entropy_image = img_as_ubyte(entropy_image)
display_image(entropy_image, "Local Entropy Map")

# Step (c): Normalized negative entropy image
normalized_entropy = cv2.normalize(entropy_image, None, 0, 255, cv2.NORM_MINMAX)
negative_entropy = 255 - normalized_entropy
display_image(negative_entropy, "Normalized Negative Entropy Image")

# Step (d): Binarized entropy image
ret, binarized_entropy = cv2.threshold(negative_entropy, 128, 255, cv2.THRESH_BINARY)
display_image(binarized_entropy, "Binarized Entropy Image")

# Step (e): Result of masking
masked_image = cv2.bitwise_and(original_image, original_image, mask=binarized_entropy.astype(np.uint8))
display_image(masked_image, "Result of Masking")

# Step (f): Dilated masked image (Full Background Estimate)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated_image = cv2.dilate(masked_image, kernel, iterations=2)
display_image(dilated_image, "Dilated Masked Image (Full Background Estimate)")

# Step (g): Result of background subtraction
background_subtracted = cv2.subtract(original_image, dilated_image)
display_image(background_subtracted, "Result of Background Subtraction")

# Step (h): Negative with eliminated background
negative_eliminated_background = 255 - background_subtracted
display_image(negative_eliminated_background, "Negative with Eliminated Background")

histogram = cv2.calcHist([negative_eliminated_background], [0], None, [256], [0, 256])
ret, new = cv2.threshold(negative_eliminated_background, 200, 255, cv2.THRESH_BINARY)
display_image(new, "Histogram of Eliminated Background")

detailed_data = pytesseract.image_to_data(new, output_type=pytesseract.Output.DICT, config="--psm 11")

fig, ax = plt.subplots(1, figsize=(12, 12))
ax.imshow(cv2.cvtColor(new, cv2.COLOR_BGR2RGB))

n_boxes = len(detailed_data['text'])
for i in range(n_boxes):
    if int(detailed_data['conf'][i]) > 30:  # Only consider positive confidence scores
        (x, y, w, h) = (detailed_data['left'][i], detailed_data['top'][i], detailed_data['width'][i], detailed_data['height'][i])
        conf = detailed_data['conf'][i]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y - 10, f'{detailed_data["text"][i]} ({conf}%)', color='blue', fontsize=12, weight='bold')

plt.title('Detected Characters with Bounding Boxes and Confidence Scores')
plt.axis('off')
plt.show()
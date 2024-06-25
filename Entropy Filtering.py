from scipy.ndimage import rotate
from skimage.filters.rank import entropy
from skimage.morphology import disk
from PIL import Image
import cv2
import PIL.Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import pytesseract


def display_image(image, title, cmap='gray'):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


def detect_key_features(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def match_key_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def estimate_orientation(matches, keypoints1, keypoints2):
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    angle = -np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    return 90 * round(angle / 90)


def correct_image_orientation(image, angle):
    corrected_image = rotate(image, angle, reshape=True)
    return corrected_image


image_path = "./cropped images/IC.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
reference_image = cv2.imread("./cropped images/IC.png")
cropped_texts_dir = "./cropped_texts"
template_path = "./Logo Template.jpeg"

os.makedirs(cropped_texts_dir, exist_ok=True)
image = cv2.medianBlur(image, 5)
template = cv2.imread(template_path, 0)
height, width = image.shape

keypoints1, descriptors1 = detect_key_features(image)
keypoints2, descriptors2 = detect_key_features(reference_image)

matches = match_key_features(descriptors1, descriptors2)
angle = estimate_orientation(matches, keypoints1, keypoints2)

corrected_image = correct_image_orientation(image, angle)
display_image(corrected_image, "Corrected Image")

res = cv2.matchTemplate(corrected_image, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8  # Adjust this value as needed
loc = np.where(res >= threshold)

# Create a mask to mark the locations of the template
mask = np.zeros_like(corrected_image)
w, h = template.shape

for pt in zip(*loc[::-1]):
    cv2.rectangle(mask, pt, (pt[0] + w, pt[1] + h), 255, -1)

inpainted_image = cv2.inpaint(corrected_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

display_image(inpainted_image, "Inpainted Image")


entropy_image = entropy(inpainted_image, disk(3))
scaled_entropy = entropy_image / entropy_image.max()
mask = entropy_image > 0.6

filtered_image = inpainted_image * mask
display_image(filtered_image, "Filtered Image")

# Apply morphological closing to fill in holes
kernel = np.ones((4, 4), np.uint8)
closed_image = cv2.morphologyEx(filtered_image, cv2.MORPH_CLOSE, kernel)

# Display the result after closing
closed_image_inverted = cv2.bitwise_not(closed_image)
display_image(closed_image_inverted, "Closed image inverted")

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_image, connectivity=8)
sizes = stats[1:, -1]
min_size = 500  # Minimum size of a component to keep (tune as needed)
filled_image = np.zeros((labels.shape), np.uint8)

for i in range(0, num_labels - 1):
    if sizes[i] >= min_size:
        filled_image[labels == i + 1] = 255

display_image(filled_image, "Connected Components Analysis")
filled_image = cv2.bitwise_not(filled_image)
threshold_value = 128  # Adjust threshold value as needed
_, mask = cv2.threshold(filled_image, threshold_value, 255, cv2.THRESH_BINARY)

inpainted_image = cv2.inpaint(closed_image_inverted, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
display_image(inpainted_image, "Inpainted Image")

detection_data = pytesseract.image_to_data(
    inpainted_image,
    config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    output_type=pytesseract.Output.DICT
)

fig, ax = plt.subplots(1, figsize=(12, 12))
ax.imshow(inpainted_image, cmap='gray')

n_boxes = len(detection_data['text'])
for i in range(n_boxes):
    if int(detection_data['conf'][i]) >= 0:  # Consider all detected boxes for cropping
        (x, y, w, h) = (
        detection_data['left'][i], detection_data['top'][i], detection_data['width'][i], detection_data['height'][i])
        conf = detection_data['conf'][i]

        # Draw bounding boxes
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y - 10, f'{detection_data["text"][i]} ({conf}%)', color='blue', fontsize=12, weight='bold')

        # Crop the detected region
        cropped_region = inpainted_image[y:y + h, x:x + w]

        cropped_region_image = Image.fromarray(cropped_region)
        cropped_region_image.save(os.path.join(cropped_texts_dir, f"cropped_text_{i}.png"))

plt.title('Detected Characters with Bounding Boxes and Confidence Scores')
plt.axis('off')
plt.show()


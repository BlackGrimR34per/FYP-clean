import cv2
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt

# Step 1: Read the image
image = cv2.imread("sauvola_binary.png")

# Step 2: Convert to grayscale (if needed, here assumed preprocessed image)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Tesseract OCR to get bounding boxes
d = pytesseract.image_to_data(gray_image, output_type=Output.DICT)

# Step 4: Draw bounding boxes
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 0:  # Filter out boxes with low confidence
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Step 5: Display the image with bounding boxes
plt.figure(figsize=(12, 12))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Text with Bounding Boxes")
plt.axis("off")
plt.show()

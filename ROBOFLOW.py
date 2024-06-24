import cv2
import inference
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Load the model
model = inference.get_model("inspection-y0kud/2")

# Read the image using cv2
image_path = "IC.png"
image = cv2.imread(image_path)

# Convert the image to RGB (cv2 loads images in BGR format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform inference
result = model.infer(image=image_rgb)

# Output directory
output_dir = "cropped images"
os.makedirs(output_dir, exist_ok=True)

# Assuming result is a list containing an ObjectDetectionInferenceResponse object
response = result[0]

# Access predictions from the response
predictions = response.predictions

for i, prediction in enumerate(predictions):
    # Extract bounding box coordinates and other details
    x_center = prediction.x
    y_center = prediction.y
    width = prediction.width
    height = prediction.height
    label = prediction.class_name

    # Calculate the top-left corner of the bounding box
    x = int(x_center - (width / 2))
    y = int(y_center - (height / 2))
    width = int(width)
    height = int(height)

    # Crop the image using the bounding box coordinates
    cropped_image = image[y:y + height, x:x + width]

    # Create a file name for the cropped image
    output_path = os.path.join(output_dir, image_path)

    # Save the cropped image
    cv2.imwrite(output_path, cropped_image)

print(f"Cropped images saved to {output_dir}")
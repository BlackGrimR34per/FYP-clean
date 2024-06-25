


# image_path = "./cropped images/IC.png"
# preprocessed_image = preprocess_image(image_path)
# detection_data = extract_text(preprocessed_image)
# print(detection_data)

# import numpy as np
# from skimage.filters.rank import entropy
# from skimage.morphology import disk
# import cv2
# import matplotlib.pyplot as plt
# import os
# import inference
#
# model = inference.get_model("inspection-y0kud/2")
#
#
# def display_image(image, title, cmap='gray'):
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image, cmap=cmap)
#     plt.title(title)
#     plt.axis('off')
#     plt.show()
#
#
# def apply_entropy_filter(image, radius=3):
#     entropy_image = entropy(image, disk(radius))
#     scaled_entropy = entropy_image / entropy_image.max()
#     return scaled_entropy
#
#
# def analyze_entropy_image(height):
#     # Calculate the histogram of the entropy image
#
#     # Decide based on histogram analysis
#     if height < 300:  # This threshold can be adjusted
#         return 'normal'
#     else:
#         return 'inverse'
#
#
# def apply_mask(image, entropy_image, mask_type):
#     mask = entropy_image > 0.6
#     if mask_type == 'normal':
#         mask = 1 - mask
#     filtered_image = mask * image
#     return filtered_image
#
#
# # Load and preprocess the image
# image_path = "./cropped images/IC_1.png"
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
# height, width = image.shape
#
# if image is None:
#     raise ValueError("Image not found or path is incorrect")
# if not isinstance(image, np.ndarray):
#     raise ValueError("Image is not in the correct format")
# if image.ndim != 2:
#     raise ValueError("Image is not grayscale")
#
# # Display original image
# display_image(image, "Original Image")
#
# # Apply entropy filtering
# entropy_image = apply_entropy_filter(image)
#
# # Analyze entropy image to decide mask type
# mask_type = analyze_entropy_image(height)
# print(f"Selected mask type: {mask_type}")
#
# # Apply the selected mask
# filtered_image = apply_mask(image, entropy_image, mask_type)
#
# # Display the filtered image
# display_image(filtered_image, "Filtered Image")

# For further processing like thresholding
# ret, thresholded_image = cv2.threshold(filtered_image, 70, 255, cv2.THRESH_BINARY_INV)
# display_image(thresholded_image, "Thresholded Image")

# import cv2
# import inference
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import os
#
# # Load the model
# model = inference.get_model("inspection-y0kud/2")
#
# # Define the directory containing images
# input_dir = "/Users/ysheraun/Documents/Vision Dataset/Long"
# output_dir = "/Users/ysheraun/Documents/Vision Dataset/Long/cropped"
# os.makedirs(output_dir, exist_ok=True)

# for filename in os.listdir(input_dir):
#     if filename.endswith(".png") or filename.endswith(".jpg"):  # Process only image files
#         image_path = os.path.join(input_dir, filename)
#
#         # Read the image using cv2
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Failed to load image: {image_path}")
#             continue
#
#         # Convert the image to RGB (cv2 loads images in BGR format)
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         # Perform inference
#         result = model.infer(image=image_rgb)
#
#         # Assuming result is a list containing an ObjectDetectionInferenceResponse object
#         response = result[0]
#
#         # Access predictions from the response
#         predictions = response.predictions
#
#         for i, prediction in enumerate(predictions):
#             # Extract bounding box coordinates and other details
#             x_center = prediction.x
#             y_center = prediction.y
#             width = prediction.width
#             height = prediction.height
#             label = prediction.class_name
#
#             # Calculate the top-left corner of the bounding box
#             x = int(x_center - (width / 2))
#             y = int(y_center - (height / 2))
#             width = int(width)
#             height = int(height)
#
#             # Crop the image using the bounding box coordinates
#             cropped_image = image[y:y + height, x:x + width]
#
#             # Create a file name for the cropped image
#             output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_{label}_{i}.png")
#
#             # Save the cropped image
#             cv2.imwrite(output_path, cropped_image)
#
# print(f"Cropped images saved to {output_dir}")

# def get_average_image_shape(directory):
#     total_height = 0
#     total_width = 0
#     total_images = 0
#
#     for filename in os.listdir(directory):
#         if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
#             image_path = os.path.join(directory, filename)
#             image = cv2.imread(image_path)
#             if image is not None:
#                 height, width = image.shape[:2]
#                 total_height += height
#                 total_width += width
#                 total_images += 1
#
#     if total_images == 0:
#         return None
#
#     avg_height = total_height / total_images
#     avg_width = total_width / total_images
#
#     return (avg_height, avg_width)
#
# # Define the directory containing images
# input_dir = "/Users/ysheraun/Documents/Vision Dataset/Small/cropped"
#
# average_shape = get_average_image_shape(input_dir)
#
# if average_shape:
#     print(f"Average image shape (height, width): {average_shape}")
# else:
#     print("No valid images found in the directory.")


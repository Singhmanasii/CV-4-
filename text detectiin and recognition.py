import cv2
import numpy as np
import os
from datetime import datetime

# Step 1: Load the input image
image_path = 'C:/Users/singh/Downloads/text detection.png'
print(f"Loading image from {image_path}...")
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Image not found or unable to load!")
    exit()

print("Image loaded successfully.\n")

# Step 2: Convert the image to grayscale
print("Converting image to grayscale...")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Image converted to grayscale.\n")

# Step 3: Apply MSER to detect text regions
print("Applying MSER to detect text regions...")
mser = cv2.MSER_create()
regions, _ = mser.detectRegions(gray_image)

# Step 4: Draw bounding boxes around detected regions
print("Drawing bounding boxes around detected regions...\n")
for region in regions:
    hull = cv2.convexHull(region.reshape(-1, 1, 2))
    x, y, w, h = cv2.boundingRect(hull)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

print("All detected text regions have been marked with bounding boxes.\n")

# Step 5: Save the output image
output_dir = 'C:/Users/singh/Downloads/'
output_filename = f"text_detection_mser_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
output_path = os.path.join(output_dir, output_filename)
cv2.imwrite(output_path, image)
print(f"Output image saved as {output_filename} in {output_dir}.\n")

# Step 6: Display the output image with bounding boxes
cv2.imshow("Text Detection using MSER", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Script execution completed successfully.")

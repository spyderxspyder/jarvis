import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to convert RGB to Grayscale using cv2
def rgb_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function for Zero Order Hold
def zero_order_hold(gray_image, scale_factor):
    h, w = gray_image.shape
    new_h, new_w = h * scale_factor, w * scale_factor
    zoh_image = np.zeros((new_h, new_w), dtype=np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            zoh_image[i, j] = gray_image[i // scale_factor, j // scale_factor]

    return zoh_image

# Function for First Order Hold
def first_order_hold(gray_image, scale_factor):
    h, w = gray_image.shape
    new_h, new_w = h * scale_factor, w * scale_factor
    foh_image = np.zeros((new_h, new_w), dtype=np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            x = i / scale_factor
            y = j / scale_factor
            x1, y1 = int(x), int(y)
            x2, y2 = min(x1 + 1, h - 1), min(y1 + 1, w - 1)

            # Linear interpolation
            foh_image[i, j] = (
                gray_image[x1, y1] * (x2 - x) * (y2 - y) +
                gray_image[x2, y1] * (x - x1) * (y2 - y) +
                gray_image[x1, y2] * (x2 - x) * (y - y1) +
                gray_image[x2, y2] * (x - x1) * (y - y1)
            )

    return foh_image

# Load the image using cv2
image_path = 'D:/sem9/cv/sunflower.jpeg'  # Replace with your image path
rgb_image = cv2.imread(image_path)

# Convert to grayscale
gray_image = rgb_to_grayscale(rgb_image)

# Define scale factor
scale_factor = 4

# Apply ZOH and FOH
zoh_image = zero_order_hold(gray_image, scale_factor)
foh_image = first_order_hold(gray_image, scale_factor)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original Grayscale")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Zero Order Hold")
plt.imshow(zoh_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("First Order Hold")
plt.imshow(foh_image, cmap='gray')
plt.axis('off')

plt.show()

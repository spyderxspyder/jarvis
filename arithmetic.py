import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load two images (convert to grayscale for simplicity)
img1 = Image.open("/content/drive/MyDrive/MSc SS/9th Sem/20XW97 - CV lab/labwork/images.jpeg").convert("L")
img2 = Image.open("/content/drive/MyDrive/MSc SS/9th Sem/20XW97 - CV lab/labwork/images2.jpeg").convert("L")

img1 = img1.resize(img2.size)

# Convert to NumPy arrays
arr1 = np.array(img1, dtype=np.int16)  # use int16 to avoid overflow during operations
arr2 = np.array(img2, dtype=np.int16)

# Arithmetic Operations
add_img = np.clip(arr1 + arr2, 0, 255).astype(np.uint8)
sub_img = np.clip(arr1 - arr2, 0, 255).astype(np.uint8)
mul_img = np.clip(arr1 * arr2 / 255, 0, 255).astype(np.uint8)
div_img = np.clip(arr1 / (arr2 + 1) * 255, 0, 255).astype(np.uint8)  # avoid divide by zero

# Logical Operations
and_img = np.bitwise_and(arr1, arr2).astype(np.uint8)
or_img  = np.bitwise_or(arr1, arr2).astype(np.uint8)
xor_img = np.bitwise_xor(arr1, arr2).astype(np.uint8)
not_img1 = np.bitwise_not(arr1).astype(np.uint8)

# Display results using matplotlib
images = [img1, img2, add_img, sub_img, mul_img, div_img, and_img, or_img, xor_img, not_img1]
titles = ["Image 1", "Image 2", "Addition", "Subtraction", "Multiplication", "Division",
          "AND", "OR", "XOR", "NOT (Image1)"]

plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# Function to check if region is homogeneous
# -----------------------------
def is_homogeneous(region, threshold):
    return (region.max() - region.min()) <= threshold

# -----------------------------
# Recursive function to split
# -----------------------------
def split(image, x, y, w, h, threshold):
    region = image[y:y+h, x:x+w]

    if w <= 1 or h <= 1 or is_homogeneous(region, threshold):
        return [(x, y, w, h)]

    w2, h2 = w // 2, h // 2
    regions = []
    regions += split(image, x, y, w2, h2, threshold)           # Top-left
    regions += split(image, x + w2, y, w - w2, h2, threshold) # Top-right
    regions += split(image, x, y + h2, w2, h - h2, threshold) # Bottom-left
    regions += split(image, x + w2, y + h2, w - w2, h - h2, threshold) # Bottom-right

    return regions

# -----------------------------
# Draw segmented image
# -----------------------------
def segment(image, regions):
    seg_img = np.zeros_like(image)
    for (x, y, w, h) in regions:
        mean_val = int(np.mean(image[y:y+h, x:x+w]))
        seg_img[y:y+h, x:x+w] = mean_val
    return seg_img

# -----------------------------
# Load image (grayscale)
# -----------------------------
img = Image.open("/content/drive/MyDrive/MSc SS/9th Sem/20XW97 - CV lab/labwork/images.jpeg").convert("L")
img_np = np.array(img)

# Threshold
threshold = 100

# Split into regions
regions = split(img_np, 0, 0, img_np.shape[1], img_np.shape[0], threshold)
print(region)
# Create segmented image
seg_img = segment(img_np, regions)

# -----------------------------
# Plot results
# -----------------------------
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_np, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(seg_img, cmap="gray")
plt.title("Region Splitting & Merging")
plt.axis("off")

plt.show()


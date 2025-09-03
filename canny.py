import numpy as np
import math

#img = np.array([
#    [10, 9, 9, 4, 0],
#    [0, 6, 6, 2, 2],
#    [5, 9, 8, 4, 3],
#    [7, 5, 5, 4, 3],
#    [8, 10, 8, 5, 0]
#])

img = Image.open("/content/drive/MyDrive/MSc SS/9th Sem/20XW97 - CV lab/labwork/images.jpeg").convert("L")
img = np.array(img)

gaussian_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]) / 16.0

x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

y = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

height, width = img.shape
padded_image = np.pad(img, pad_width=1, mode='constant', constant_values=0)

filtered_img = np.zeros_like(img)

# Step 1: Apply Gaussian Filtering

for i in range(height):
    for j in range(width):
        region = padded_image[i:i+3, j:j+3]
        filtered_img[i, j] = np.sum(region * gaussian_kernel)

padded_filtered_image = np.pad(filtered_img, pad_width=1, mode='constant', constant_values=0)

gx = np.zeros((height, width))
gy = np.zeros((height, width))
magnitude = np.zeros((height, width))
angle = np.zeros((height, width))

#gx = np.zeros((height - 2, width - 2))
#gy = np.zeros((height - 2, width - 2))
#magnitude = np.zeros((height - 2, width - 2))

padded_height, padded_width = padded_filtered_image.shape


# Step 2: Calculate Gx, Gy using Sobel Operation

#for i in range(height - 2):
#  for j in range(width - 2):

    #region = img[i : i + 3, j : j + 3]


for i in range(padded_height - 2):
  for j in range(padded_width - 2):

    region = padded_filtered_image[i : i + 3, j : j + 3]

    gx[i][j] = np.sum(region * x)
    gy[i][j] = np.sum(region * y)

    # Step 3: Calculate Magnitude and Orientation matrix
    magnitude[i, j] = np.sqrt(gx[i][j] ** 2 + gy[i][j] ** 2)
    angle[i, j] = np.arctan2(gy[i, j], gx[i, j])
    angle[i, j] = np.where(angle[i, j] < 0, angle[i, j] + np.pi, angle[i, j])
    angle[i, j] = angle[i, j] * 180 / np.pi

min_val = np.min(magnitude)
max_val = np.max(magnitude)

magnitude = 255 * (magnitude - min_val) / (max_val - min_val)

edge_map = np.zeros_like(magnitude)


# Step 4: Apply Non-maximum Suppression

for i in range(1, height - 1):
  for j in range(1, width - 1):

    angle_deg = angle[i, j]
    mag = magnitude[i, j]

    if (0 <= angle_deg < 22.5) or (157.5 <= angle_deg <= 180 ):
        neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
    elif (22.5 <= angle_deg < 67.5):
        neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
    elif (67.5 <= angle_deg < 112.5):
        neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
    else:
        neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]

    if mag >= max(neighbors):
        edge_map[i, j] = mag
    else:
        edge_map[i, j] = 0

# Step 5: Apply double thresholding

#avg_magnitude = np.mean(magnitude)
#high_threshold = avg_magnitude * 1.5
#low_threshold = avg_magnitude * 0.5

max_magnitude = np.max(magnitude)
high_threshold = max_magnitude * 0.8
low_threshold = max_magnitude * 0.25

strong_edges = (edge_map > high_threshold).astype(int)
weak_edges = ((edge_map > low_threshold) & (edge_map <= high_threshold)).astype(int)

# ------------------ Visualization ------------------
plt.figure(figsize=(12,10))

plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(filtered_img, cmap='gray')
plt.title("Gaussian Filtered")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(magnitude, cmap='gray')
plt.title("Gradient Magnitude")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(angle, cmap='hsv')
plt.title("Gradient Orientation")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(edge_map, cmap='gray')
plt.title("After Non-Max Suppression")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(strong_edges + weak_edges, cmap='gray')
plt.title("Double Thresholding")
plt.axis("off")

plt.tight_layout()
plt.show()

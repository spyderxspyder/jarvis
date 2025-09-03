from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load and convert to grayscale
img = Image.open("/content/drive/MyDrive/MSc SS/9th Sem/20XW97 - CV lab/labwork/images.jpeg").convert("L")
img_array = np.array(img)

# 1. Negative transformation
L = 256  # for 8-bit image
negative = (L - 1) - img_array

# 2. Log transformation
c = 255 / np.log(1 + np.max(img_array))  # normalization factor
log_transformed = c * np.log(1 + img_array)
log_transformed = np.array(log_transformed, dtype=np.uint8)

# 3. Power-law (Gamma) transformation
gamma = 0.5  # try 0.5, 1.5 etc.
c = 255 / (np.max(img_array) ** gamma)
gamma_transformed = c * (img_array ** gamma)
gamma_transformed = np.array(gamma_transformed, dtype=np.uint8)

# Plot results
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(img_array, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(negative, cmap="gray")
plt.title("Negative Transformation")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(log_transformed, cmap="gray")
plt.title("Log Transformation")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(gamma_transformed, cmap="gray")
plt.title("Gamma Transformation (γ=0.5)")
plt.axis("off")

plt.tight_layout()
plt.show()


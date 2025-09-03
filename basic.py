# prompt: Use opencv to convert one image to another type and basic operations as the image in the location ('/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/img.jpg')

!pip install opencv-python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img_path = '/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/img.jpg'
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not load image from {img_path}")
else:
    # Display the original image (OpenCV uses BGR color space)
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.subplot(2, 3, 2)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    # Convert image to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB))
    plt.title('HSV Image')
    plt.axis('off')

    # Basic operations: Resize
    resized_img = cv2.resize(img, (300, 200)) # width, height
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    plt.title('Resized Image (300x200)')
    plt.axis('off')

    # Basic operations: Flip
    flipped_img = cv2.flip(img, 1) # 0 for vertical, 1 for horizontal, -1 for both
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(flipped_img, cv2.COLOR_BGR2RGB))
    plt.title('Horizontally Flipped Image')
    plt.axis('off')

    # Basic operations: Rotate (using getRotationMatrix2D and warpAffine)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 90, 1.0) # rotate 90 degrees, scale 1.0
    rotated_img = cv2.warpAffine(img, M, (w, h))
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
    plt.title('Rotated Image (45 deg)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Saving converted images
    cv2.imwrite('grayscale_img.png', gray_img)
    cv2.imwrite('resized_img.jpg', resized_img)

    print("Image conversion and basic operations performed.")
    print("Grayscale image saved as 'grayscale_img.png'")
    print("Resized image saved as 'resized_img.jpg'")


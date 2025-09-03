import numpy as np

img = np.array([
    [0, 1, 2, 1, 0, 0],
    [0, 3, 4, 4, 1, 0],
    [2, 4, 5, 5, 4, 0],
    [1, 4, 5, 5, 4, 1],
    [0, 3, 4, 4, 3, 1],
    [0, 2, 3, 3, 2, 0]
])

freq = dict()

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i, j] in freq:
            freq[img[i, j]] += 1
        else:
            freq[img[i, j]] = 1


intensity_count = len(freq)
pixel_count = img.shape[0] * img.shape[1]
sig_b = np.zeros(intensity_count)

for i in range(len(freq)):

  wb = np.sum(list(freq.values())[:i]) / pixel_count
  wf = np.sum(list(freq.values())[i:]) / pixel_count

  if i == 0:
    mb = 0
    mf = 1
  else:
    mb = np.sum(np.array(list(freq.keys()))[:i] * np.array(list(freq.values()))[:i]) / np.sum(list(freq.values())[:i])
    mf = np.sum(np.array(list(freq.keys()))[i:] * np.array(list(freq.values()))[i:]) / np.sum(list(freq.values())[i:])

  var_b = wb * wf * (mb - mf) ** 2
  sig_b[i] = var_b


# Find the optimal threshold
optimal_threshold_index = np.argmax(sig_b)

print("\nBetween-class variance for each threshold:")
print(sig_b)

print("\nOptimal Threshold (Otsu's Method):")
print(list(freq.keys())[optimal_threshold_index])




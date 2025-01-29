import cv2
import numpy as np

# Read the image in grayscale
img = cv2.imread('test_image.jfif', cv2.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# Apply histogram equalization
equ = cv2.equalizeHist(img)

# Stack the original and equalized images side-by-side for comparison
res = np.hstack((img, equ))

# Save the result
cv2.imwrite('res.png', res)

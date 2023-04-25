import cv2
import os

fileName = "00000.tif"
file_path = "../../fullScrollData/" + fileName

# Load the images
image1 = cv2.imread(file_path)
image2 = cv2.imread("../../losslesslyCompressedScrollData/c" + fileName)
# Ensure the images have the same size and number of channels
if image1.shape != image2.shape:
    raise ValueError("The two images must have the same size and number of channels.")

# Compute the absolute difference between the two images
diff_image = cv2.absdiff(image1, image2) * 100

# Save the resulting image
outputFilePath = "../../comparisonScrollData/comp100x" + fileName
cv2.imwrite(
    outputFilePath,
    cv2.cvtColor(diff_image, cv2.COLOR_RGB2BGR),
)

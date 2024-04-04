import cv2
import numpy as np
from skimage import measure

def calculate_volume_length_width(segmented_image, voxel_size):
    # Calculate volume
    volume = np.sum(segmented_image) * voxel_size

    # Calculate length and width
    contours, _ = cv2.findContours(segmented_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # Find the contour with the maximum area (the largest tumor)
        max_contour = max(contours, key=cv2.contourArea)
        # Calculate the minimum area rectangle that encloses the contour
        rect = cv2.minAreaRect(max_contour)
        length = max(rect[1]) * voxel_size  # Length corresponds to the longer side of the rectangle
        width = min(rect[1]) * voxel_size  # Width corresponds to the shorter side of the rectangle
    else:
        length = width = 0

    return volume, length, width

# Example usage:
# Assuming segmented_image is the binary segmented image of the tumor region
# voxel_size is the size of a voxel in mm^3 (e.g., obtained from MRI metadata)

# Example voxel size (in mm^3)
voxel_size = 1.0

# Example segmented_image (binary mask of the tumor region)
segmented_image = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0]])

# Calculate volume, length, and width
volume, length, width = calculate_volume_length_width(segmented_image, voxel_size)

print("Volume:", volume, "mm^3")
print("Length:", length, "mm")
print("Width:", width, "mm")

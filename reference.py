import numpy as np
import cv2

# ATTENTION ////////////////////////////////////////////////////////////////////
# YOU HAVE TO RUN image_processing.py before running this file
#//////////////////////////////////////////////////////////////////////////////

####################################################################
#CREATE AN ARBITRARY REFERENCE IMAGE
####################################################################

####################################################################
#On this part of the code, an arbitrary reference image is created to represent different parts 
#of the foot (bones, tendons, muscles, inside). The putSquare function generates square coordinates 
#within a given region. The image_reference is initialized with a black background, and each part is 
#drawn using different colors. The resulting image is displayed using cv2.imshow.

#The reference image is then passed through the segmentation process to obtain labels and centroids. 
#The first centroid (corresponding to the black pixels) is removed from the list. Finally, 
#the labels of each centroid are computed using the label_centroid function.
# Function to generate square coordinates within a given region
####################################################################


def putSquare(top_left_corner, right_down_corner, gap=2, size=5):
    lx, ly = top_left_corner
    rx, ry = right_down_corner
    PART = []
    for x in range(lx, rx - size, size + gap):
        for y in range(ly, ry - size, size + gap):
            PART.append([(xx, yy) for xx in range(x, x + size) for yy in range(y, y + size)])
    return PART

# Create an image with a black background
# Note: dtype=np.uint8 is specified to allow grayscale
image_reference = np.zeros(shape=(630, 630), dtype=np.uint8)

# Define the parts of the foot and their corresponding colors
# b for bones, t for tendons, m for muscles, i for inside
part_colors = {"b": 101, "t": 251, "m": 151, "i": 51}

# Define the coordinates for each part of the foot
TENDONS = putSquare((70, 490), (470, 570))
BONES = putSquare((70, 410), (380, 490))
MUSCLES = putSquare((270, 120), (420, 400)) + putSquare((420, 70), (470, 430))

parts = {"t": TENDONS, "b": BONES, "m": MUSCLES}

# Draw each part of the foot with its corresponding color
for part, list_coords in parts.items():
    color = part_colors[part]
    for coords in list_coords:
        for x, y in coords:
            image_reference[x, y] = color

# Perform segmentation on the reference image
n_img, labels_reference, stats_reference, centroids_reference = segmentation(image_reference, show=False, process=False)

# Delete the first centroid which corresponds to the center of the picture
# (corresponds to pixels in black)
centroids_reference = centroids_reference[1:]
stats_reference = stats_reference[1:]

# Get label of each centroid
labels_reference = label_centroid(stats_reference, image_reference, show=False)

# Display the image
cv2.imshow('IMAGE REFERENCE', image_reference)
cv2.waitKey(0)
cv2.destroyAllWindows()




























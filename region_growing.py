import cv2
import numpy as np
import matplotlib.pyplot as plt


# ATTENTION ////////////////////////////////////////////////////////////////////
# YOU HAVE TO RUN image_processing.py before running this file
#//////////////////////////////////////////////////////////////////////////////

def find_surface(img, seed_point, fill_color=100, show=False):
    """
    Performs region growing on the image.

    Args:
        img: 2D array containing only 0 and 255.
        seed_point: Origin of the region growing.
        fill_color: Color to fill the region with.
        show: Flag to show the result.

    Returns:
        img: Image with the region filled.
    """
    # Set the threshold for region growing
    threshold = 1
    # Set the connectivity (4 or 8) for region growing
    connectivity = 4
    # Initialize the mask for region growing
    mask = np.zeros((img.shape[0]+2, img.shape[1]+2), np.uint8)
    # Perform region growing only on white regions
    if img[seed_point[1], seed_point[0]] == 255:
        cv2.floodFill(img, mask, seed_point, fill_color, threshold, threshold, connectivity)
    if show:
        cv2.imshow('Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img



def segmentation(img, show=True, process=True, er_kern=7, di_kern=5):
    """
    Performs region growing on the image.

    Args:
        img: 2D array containing only 0 and 255.
        show: Flag to show the result.
        process: Flag to perform additional image processing steps.
        er_kern: Kernel size for erosion.
        di_kern: Kernel size for dilation.

    Returns:
        img: Processed image.
        labels: Labels of the connected components.
        stats: Statistics of the connected components.
        centroids: Centroids of the connected components.
    """
    if process:
        img = erode(img, kern=er_kern, iteration=1)  # Add erosion step
        img = dilation(img, kern=di_kern, iteration=1)  # Add dilation step
    # Perform region growing
    connectivity = 4  # 4 or 8: impact the definition of two nearby pixels
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
    fill_color = 250
    # COLOR THE IMAGE
    for X in centroids:
        img = find_surface(img, (int(X[0]), int(X[1])), fill_color)
        fill_color -= 5
    # Display the result
    if show:
        cv2.imshow('Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img, labels, stats, centroids


###############################################################################
###############################################################################

#path of the image
path = "C:/Users/grego/github/EIT_GT_ENPC/6.jpeg"

# Load image as grayscale
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Put a filter
img = threshold(img, 110)

plt.imshow(img)

# Segmentation
n_img, labels, stats, centroids = segmentation(img)


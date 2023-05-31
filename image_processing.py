import cv2
import numpy as np


##############################################################################
#RUN BEFORE RUNING OTHERS FILES
##############################################################################


def threshold(img, seuil):
    """
    Applies thresholding to the input image to create a binary image.

    Args:
        img: Input image to be thresholded.
        seuil: Threshold value.

    Returns:
        binary_image: Thresholded binary image.
    """
    thresh_value, binary_image = cv2.threshold(img, seuil, 255, cv2.THRESH_BINARY)
    return binary_image


def erode(img, kern=7, iteration=1, show=False):
    """
    Applies erosion to the input image.

    Args:
        img: Input image to be eroded.
        kern: Size of the kernel for erosion.
        iteration: Number of iterations for erosion.
        show: Flag to display the result.

    Returns:
        img: Eroded image.
    """
    kernel = np.ones((kern, kern), np.uint8)
    img = cv2.erode(img, kernel, iterations=iteration)
    if show:
        cv2.imshow('Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img


def dilation(img, kern=6, iteration=2, show=False):
    """
    Applies dilation to the input image.

    Args:
        img: Input image to be dilated.
        kern: Size of the kernel for dilation.
        iteration: Number of iterations for dilation.
        show: Flag to display the result.

    Returns:
        img: Dilated image.
    """
    kernel = np.ones((kern, kern), np.uint8)
    img = cv2.dilate(img, kernel, iterations=iteration)
    if show:
        cv2.imshow('Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img



def label_centroid(stats, labels, show=False):
    """
    Associates a label to each centroid.

    Args:
        stats: Statistics of connected components.
        labels: Connected components labeled image.
        show: Flag to display the result.

    Returns:
        centroids_label: List of labels associated with each centroid.
    """
    centroids_label = []
    for i in range(0, len(stats)):
        x, y, width, height, _ = stats[i]
        sub_labels = labels[y:y+height, x:x+width] #define a sub image (a rectangle containing the object)
        sub_labels = sub_labels[sub_labels != 0]  # Remove zero values
        unique_labels, label_counts = np.unique(sub_labels, return_counts=True)
        most_common_label = unique_labels[np.argmax(label_counts)]  # Return the most common label
        centroids_label.append(most_common_label)
        if show:
            print(y, x)
            new_image = np.copy(labels)
            new_image[y:y+height, x:x+width] = most_common_label
            cv2.imshow('sub_label', new_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return centroids_label


def snell_foot(img0, show=False):
    """
    Performs image processing to detect the foot.

    Args:
        img0: Input image without any processing.
        show: Flag to display the result.

    Returns:
        body: Processed image with the foot in white.
    """
    # Load image as grayscale
    # Put a filter
    body = threshold(img0, 20)
    body = erode(body, kern=10, iteration=1)
    body = dilation(body, kern=10, iteration=2)

    if show:
        cv2.imshow('Result', body)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return body

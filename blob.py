import cv2
import numpy as np



#ATTENTION ////////////////////////////////////////////////////////////////////
#YOU HAVE TO RUN image_processing.py before runing this file
#////////////////////////////////////////////////////////////////////////////// 

def blob_algo(binary_image):
    
    # Set up SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    
    # Set thresholds
    params.minThreshold = 50
    params.maxThreshold = 200
    
    # Filter by area
    params.filterByArea = True
    params.minArea = 250
    params.maxArea = 10000
    
    # Filter by circularity
    params.filterByCircularity = False
    params.minCircularity = 0.01
    
    # Filter by convexity
    params.filterByConvexity = False
    #between 0 and 1
    params.minConvexity = 0.8
    params.maxConvexity = 1
    
    # Filter by inertia
    #between 0 and 1 
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    params.maxInertiaRatio = 1
    
    params.blobColor = 255
    
    # Create detector object
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(binary_image)
    
    # Draw blobs on image
    img_with_keypoints = cv2.drawKeypoints(binary_image, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Display image with keypoints
    cv2.imshow('Blobs', img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
# Load image as grayscale
img = cv2.imread("C:/Users/grego/github/EIT_GT_ENPC/6.jpeg", cv2.IMREAD_GRAYSCALE) 

binary_image = threshold(img,110)

binary_image = erode(binary_image,kern = 7,  iteration=1) 
binary_image = dilation(binary_image, kern = 5, iteration = 1) 

blob_algo(binary_image)





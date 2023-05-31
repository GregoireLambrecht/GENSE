import cv2


# ATTENTION ////////////////////////////////////////////////////////////////////
# YOU HAVE TO RUN image_processing.py before running this file
#//////////////////////////////////////////////////////////////////////////////

####################################################################
# Knn on the reference image 
####################################################################


####################################################################
# In this part of the code, a loop is used to iterate through a series of input images. 
#For each image, it is read and converted to grayscale using cv2.imread function. 
#Then, the labelize function is called with the reference image, centroids, labels, 
#and the current input image. 
#The labelize function performs the labeling process using K-Nearest Neighbors 
#algorithm and returns the labeled image, labels, centroids, and results. Finally, 
#the labeled image is saved using cv2.imwrite function.

# Please make sure to update the file paths according to your specific directory structure before running the code.
####################################################################

def labelize(image_reference, centroids_reference, labels_reference, img, show=True, name="", show_centroids=False):
    """
    Labels the input image using the K-Nearest Neighbors algorithm based on a reference image.

    Args:
        image_reference: Reference image used for training.
        centroids_reference: Centroids of the reference image.
        labels_reference: Labels of the reference image.
        img: Input image to be labeled.
        show: Flag to show the result.
        name: Name of the input image.
        show_centroids: Flag to show centroids in the result image.

    Returns:
        n_img: Labeled image.
        labels: Labels of the segmented objects.
        centroids: Centroids of the segmented objects.
        results: Predicted labels for the segmented objects.
    """
    # Put a filter
    body = snell_foot(img)
    
    img = threshold(img, 110)
       
    # Segmentation
    n_img, labels, stats, centroids = segmentation(img, show=False, er_kern=10, di_kern=5)
    
    centroids = centroids[1:]
    stats = stats[1:]
    
    labels = label_centroid(stats, n_img, show=False)  # At one, to avoid zero (the black part)
    
    # Training set
    centroids_train = np.array(centroids_reference, dtype=np.float32)
    
    # Training label
    labels_train = np.array(labels_reference, dtype=np.float32)
    
    # Create a K-NN model
    model = cv2.ml.KNearest_create()
    
    # Train the model on the train data
    model.train(centroids_train, cv2.ml.ROW_SAMPLE, labels_train)
    k = 5
    
    ret, results, neighbours, dist = model.findNearest(np.array(centroids, dtype=np.float32), k)
    i_dist = np.argmax(dist)
    for i in range(len(labels)):
        n_img[n_img == labels[i]] = results[i]
          
    n_img[n_img == 255] = 0
    
    X, Y = n_img.shape
    for x in range(X):
        for y in range(Y): 
            if n_img[x, y] == 0 and body[x, y] == 255:
                n_img[x, y] = part_colors["i"]
                
    # Display centroids
    if show_centroids:
        for center in centroids:
            y = int(center[0])
            x = int(center[1])
            n_img[x-3:x+3, y-3:y+3] = 255
            n_img[x-3:x+3, y-3] = 0
            n_img[x-3:x+3, y+3] = 0
            n_img[x-3, y-3:y+3] = 0
            n_img[x+3, y-3:y+3] = 0
         
    if show: 
        cv2.imshow('Result for image ' + name, n_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return n_img, labels, centroids, results


for k in range(0, 15):
    img = cv2.imread("C:/Users/grego/github/EIT_GT_ENPC/raw/achilles tendon rupture/" + str(k) + ".jpeg", cv2.IMREAD_GRAYSCALE) 
    n_img, _, _, results = labelize(image_reference, centroids_reference, labels_reference, img, show_centroids=False)
    
    cv2.imwrite("C:/Users/grego/github/EIT_GT_ENPC/achilles_tendon_rupture_reference_TIFF/"+str(k) + ".tif", n_img)



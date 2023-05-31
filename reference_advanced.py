import numpy as np
import cv2


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


def putSquare(top_left_corner,right_down_corner, gap = 1, size = 5):
    lx,ly = top_left_corner
    rx,ry = right_down_corner
    PART = []
    for x in range(lx,rx-size,size+gap):
        for y in range(ly, ry-size, size+gap): 
            PART.append([(xx, yy) for xx in range(x,x+size) for yy in range(y,y+size)])
    return PART
        
        
# Create an image with a black background
# Note: dtype=np.uint8 is specified to allow grayscale
image_reference = np.zeros(shape=(630,630), dtype=np.uint8)

# Define the parts of the foot and their corresponding colors
# b for bones, t for tendons, m for muscles, i for inside
part_colors = {"b": 101, "t": 251, "m": 151, "i" : 51}


TENDONS = putSquare((70,520),(420,570))
BONES = putSquare((70,420),(200,470)) + putSquare((210,400),(320,500)) + putSquare((280,320),(320,390)) + putSquare((330,260),(400,500)) + putSquare((370,170),(400,250))

MUSCLES =  putSquare((420,100),(470,500)) + putSquare((230,300),(270,390)) + putSquare((280,240),(320,320)) + putSquare((330,150),(360,250)) + putSquare((370,130),(400,160))


# Define the coordinates for each part of the foot
parts = {"t":TENDONS, "b": BONES, "m": MUSCLES}
    
# Draw each part of the foot with its corresponding color
for part, list_coords in parts.items():
    color = part_colors[part]
    for coords in list_coords:
        for x, y in coords:
            image_reference[x, y] = color

#Segmentation on our reference image
n_img,labels_reference, stats_reference, centroids_reference = segmentation(image_reference, show = False, process = False)

#Delete the first centroid which corresponds to the center of the picture
#Correspond to pixels in black 
centroids_reference = centroids_reference[1:]
stats_reference = stats_reference[1:]

#Get label of each centroid
labels_reference = label_centroid(stats_reference, image_reference, show = False)


# Display the image
cv2.imshow('IMAGE REFERENCE', image_reference)
cv2.waitKey(0)
cv2.destroyAllWindows()



def annotate(image, parts):
    image_annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1


    for label, organ in parts.items():
        for sub_organ in organ :
            x,y = sub_organ[len(sub_organ)//2]
            cv2.putText(image_annotated, label, (y, x),
                            font, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)
    return image_annotated

# Annotate the image reference with the labels
image_annotated = annotate(image_reference, parts)





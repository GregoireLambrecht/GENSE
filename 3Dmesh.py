import numpy as np
import skimage.io as io
import skimage.measure as measure
import trimesh
from trimesh import util
import pyvista as pv
import tetgen
from matplotlib import pyplot as plt

###############################################################################
#PUT THESE LINES IN A JUPYTER NOTEBOOK
#JUPYTER NOTEBOOKS ARE TO HEAVY TO USE GITHUB
###############################################################################

#Path of 2D labelized images
path = "C:/Users/grego/github/EIT_GT_ENPC/achilles_tendon_rupture_reference_TIFF/"

image_paths = []
for k in range(15): 
    image_paths.append(path + str(k) + ".tif")
    
masks = [io.imread(path) for path in image_paths]


def generate_surface_from_masks(masks):
    """
    Generates a 3D surface mesh from a list of binary masks.
    
    Args:
        masks (list): A list of binary masks representing different slices or layers.
        
    Returns:
        mesh (trimesh.Trimesh): A 3D surface mesh generated from the input masks.
    """
    # Combine the binary masks into a single volume
    volume = np.zeros((*masks[0].shape, len(masks)), dtype=np.uint8)
    for i, mask in enumerate(masks):
        volume[:,:,i] = masks[i]

    # Create a 3D surface mesh from the volume using marching cubes algorithm
    vertices, faces, normals, values = measure.marching_cubes(volume, level=0)
    vertices[:,2]*=12
    mesh = trimesh.Trimesh(vertices, faces)
    return mesh

mesh = generate_surface_from_masks(masks)

mesh.show()


# Define the color mapping
color_mapping = {
    51 : np.array([255, 0, 0],dtype=np.uint8),   #Red tissues
    101: np.array([0, 0, 0],dtype=np.uint8),     # Black Bones
    151: np.array([0, 0, 253],dtype=np.uint8),   # Blue  Muscles
    251: np.array([0, 255, 0],dtype=np.uint8),   # Green Tendons
}


def generate_surface_organ(masks,label): 
    """
    Generates a 3D surface mesh of a specific organ from a list of binary masks.
    
    Args:
        masks (list): A list of binary masks representing different slices or layers.
        label (int): The label value corresponding to the desired organ.
        
    Returns:
        mesh (trimesh.Trimesh): A 3D surface mesh of the specified organ generated from the input masks.
    """
    # Combine the binary masks into a single volume
    volume = np.zeros((*masks[0].shape, len(masks)), dtype=np.uint8)
    for i, mask in enumerate(masks):
        organ = np.copy(masks[i])
        organ[organ != label] = 0 
        volume[:,:,i] = organ

    # Create a 3D surface mesh from the volume using marching cubes algorithm
    vertices, faces, normals, values = measure.marching_cubes(volume, level=0)
    vertices[:,2]*=12
    mesh = trimesh.Trimesh(vertices, faces)
    num_vertices = len(mesh.vertices)
    color_array = np.array([color_mapping[value] for value in values], dtype=np.uint8)[:num_vertices,:]
    mesh.visual.vertex_colors = color_array.T
    mesh.visual.vertex_colors = color_array
    return mesh


muscles = generate_surface_organ(masks,151)

muscles.show()

bones = generate_surface_organ(masks,101)

bones.show()

tendons = generate_surface_organ(masks,251)

tendons.show()

tissues = generate_surface_organ(masks,51)

tissues.show()


foot_mesh = util.concatenate([bones, muscles,tendons])

foot_mesh.show()


# Assuming you have two meshes named 'mesh1' and 'mesh2'
foot_mesh_fool = util.concatenate([foot_mesh,tissues])

foot_mesh_fool.show()


#If you are not using jupyter dnon't run these lines
# file_path = "C:/Users/grego/github/EIT_GT_ENPC/mesh.obj"
# foot_mesh.export(file_path, file_type='obj')
#Sometimes doesn't work with foot_mesh_fool. Don't know why ???

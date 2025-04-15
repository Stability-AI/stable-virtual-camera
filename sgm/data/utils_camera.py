import os
import numpy as np
import json
import sys
from sgm.data.read_write_model import qvec2rotmat, read_cameras_text, read_images_text


def read_extrinsics_colmap(image_meta, mode="c2w"):
    """
    Converts image metadata from COLMAP format to a world-to-camera (w2c) transformation matrix.
    Args:
        image_meta: An object containing the following attributes:
            - tvec (numpy.ndarray): A 3-element translation vector.
            - qvec (numpy.ndarray): A 4-element quaternion representing rotation.
    Returns:
        numpy.ndarray: A 4x4 world-to-camera transformation matrix.
    """
    
    translation = image_meta.tvec.reshape(3, 1)
    rotation = qvec2rotmat(image_meta.qvec)
    w2c = np.concatenate([rotation, translation], 1)
    w2c = np.concatenate([w2c, np.array([[0.0, 0.0, 0.0, 1.0]])], 0)
    if mode == "w2c":
        return w2c
    elif mode == "c2w":
        return np.linalg.inv(w2c)
    else:
        raise ValueError

def read_intrinsics_colmap(camera_meta, normalize=False):
    assert camera_meta.model in {"PINHOLE", "OPENCV"}
    K = np.eye(3, dtype=np.float32)
    K[0,0] = camera_meta.params[0] # f_x
    K[1,1] = camera_meta.params[1] # f_y
    K[0,2] = camera_meta.params[2] # c_x
    K[1,2] = camera_meta.params[3] # c_y

    if normalize:
        K[0,0] /= camera_meta.width
        K[1,1] /= camera_meta.height
        K[0,2] /= camera_meta.width
        K[1,2] /= camera_meta.height
    return K

def read_extrinsics_nerfstudio(transforms_dict, mode="c2w"):
    """
    Reads camera extrinsics from a dictionary and returns a list of all 4x4 transformation matrices.

    Args:
        transforms_dict (dict): A dictionary containing camera extrinsics with the following keys:
            - "frames" (list): A list of dictionaries, each containing:
                - "transform_matrix" (list): A 4x4 transformation matrix.

    Returns:
        list: A list of numpy.ndarray objects, each representing a 4x4 transformation matrix.
    """
    num_frames = len(transforms_dict["frames"])
    c2ws = np.empty((num_frames, 4, 4), dtype=np.float32)
    for i, frame in enumerate(transforms_dict["frames"]):
        c2ws[i] = np.array(frame["transform_matrix"], dtype=np.float32)
    return c2ws


def read_intrinsics_nerfstudio(transforms_dict, normalize=False):
    """
    Reads camera intrinsics from a dictionary and returns the intrinsic matrix.

        transforms_dict (dict): A dictionary containing camera intrinsics with the following keys:
            - "w" (int): Image width.
            - "h" (int): Image height.
            - "fl_x" (float): Focal length in the x direction.
            - "fl_y" (float): Focal length in the y direction.
            - "cx" (float): Principal point x-coordinate.
            - "cy" (float): Principal point y-coordinate.
            principal point coordinates by the image width and height. Defaults to False.

    Returns:
        numpy.ndarray: A 3x3 camera intrinsic matrix (K) with the following structure:
            [[fl_x,  0,    cx],
             [  0,  fl_y,  cy],
             [  0,   0,    1]]
            If `normalize` is True, `fl_x`, `fl_y`, `cx`, and `cy` are normalized
            by the image width and height.
    """

    w = transforms_dict["w"]
    h = transforms_dict["h"]
    fl_x = transforms_dict["fl_x"]
    fl_y = transforms_dict["fl_y"]
    cx = transforms_dict["cx"]
    cy = transforms_dict["cy"]

    if normalize:
        fl_x /= float(w) 
        fl_y /= float(h)
        cx /= float(w)
        cy /= float(h)

    K= np.array([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    return K


def subsample_images_metas(images_metas, n):
    """
    Subsamples a dictionary of image metadata by selecting `n` evenly spaced entries.

    Args:
        images_metas (dict): A dictionary where keys represent metadata identifiers 
            and values represent the corresponding metadata.
        n (int): The number of evenly spaced entries to select from the dictionary.

    Returns:
        dict: A new dictionary containing `n` evenly spaced entries from the input dictionary.
    """
    indices = np.linspace(0, len(images_metas) - 1, n, dtype=int)
    sorted_keys = sorted(images_metas.keys())
    return {key: images_metas[key] for i, key in enumerate(sorted_keys) if i in indices}


def opencv_to_opengl(c2w):
    """
    Converts a camera-to-world transformation matrix (or a batch of matrices) from OpenCV convention to OpenGL convention.
    In the OpenCV convention:
    - The camera-to-world matrix's rotation is represented as:
        `R = [ u_1 | u_2 | u_3 ]`
    where `u_1`, `u_2`, and `u_3` are the column vectors of the matrix.
    
    - In the OpenGL convention, the camera-to-world matrix's rotation is represented as:
    `R = [ v_1 | v_2 | v_3 ]`
    where `v_1`, `v_2`, and `v_3` are the column vectors of the matrix. 
    
    The conversion is performed as follows:
    - `v_1 = u_1`
    - `v_2 = -u_2`
    - `v_3 = -u_3`

    Note:
    - The translation vector (last column of the matrix) remains unchanged, as it represents the position of the camera in world coordinates.
    Args:
        c2w (np.ndarray or torch.Tensor): A 4x4 camera-to-world transformation matrix or a batch of Nx4x4 matrices.
            - If the input is a single 4x4 matrix, it is treated as a single transformation.
            - If the input is a batch of matrices, it should have the shape (N, 4, 4), where N is the batch size.
    Returns:
        (np.ndarray or torch.Tensor): The transformed camera-to-world matrix (or batch of matrices) in OpenGL convention.
            - The shape of the output matches the shape of the input (4x4 or Nx4x4).

    See:
    - https://www.songho.ca/opengl/gl_camera.html
    - https://colmap.github.io/format.html
    """
    dummy_batch = (c2w.ndim == 2)
    if dummy_batch:
        c2w = c2w[None, ...]

    assert c2w.ndim == 3 and c2w.shape[1:] == (4, 4), "Input must have shape Nx4x4"

    c2w[:, :3, 1:3] *= -1
    if dummy_batch:
        c2w = c2w[0]
    return c2w


def nerfstudio_to_colmap(c2w, applied_transform):
    """
    applied_transform comes from the transforms.json
    """
    applied_transform = np.concatenate([applied_transform, [[0, 0, 0, 1]]])
    dummy_batch = (c2w.ndim == 2)
    if dummy_batch:
        c2w = c2w[None, ...]

    assert c2w.ndim == 3 and c2w.shape[1:] == (4, 4), "Input must have shape Nx4x4"

    c2w = applied_transform @ c2w  # swap x/y coordinates (or y/z) and negate z is the same operation
    c2w = opencv_to_opengl(c2w)  # negating second and third element of the basis is the same operation

    if dummy_batch:
        c2w = c2w[0]
    return c2w


def colmap_to_nerfstudio(c2w):
    '''
    See:
     - https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/process_data/colmap_utils.py#L409
     - https://github.com/nerfstudio-project/nerfstudio/commit/25a00a09abed042dbe9b7d5078ec30efec6fd7e0
     - https://github.com/nerfstudio-project/nerfstudio/issues/1504?utm_source=chatgpt.com

    This code matches the DL3DV transforms.json
    '''
    dummy_batch = (c2w.ndim == 2)
    if dummy_batch:
        c2w = c2w[None, ...]

    assert c2w.ndim == 3 and c2w.shape[1:] == (4, 4), "Input must have shape Nx4x4"

    c2w = opencv_to_opengl(c2w)

    # Match Nerfstudio viewer convention (+z) from COLMAP (-y)
    # TODO: what if the applied transform is np.array([0, 2, 1, 3])
    c2w[:] = c2w[:, np.array([1, 0, 2, 3]), :] # x <-> y
    c2w[:, 2, :] *= -1                         # z -> -z

    if dummy_batch:
        c2w = c2w[0]
    return c2w

if __name__ == "__main__":
    colmap_dir="/home/nviolant/TrajectoryCrafter/test/colmap_traj"
    cameras = read_cameras_text(os.path.join(colmap_dir, "cameras.txt"))
    images_metas = read_images_text(os.path.join(colmap_dir, "images.txt"))
    
    print(type(cameras[1]))
    print(read_intrinsics_colmap(cameras[1]))
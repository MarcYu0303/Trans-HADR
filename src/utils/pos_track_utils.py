import numpy as np
from scipy.spatial.transform import Rotation as R

def rotate_translate_hand_keypoints(keypoints, rotation_matrix, translation_vector):
    """
    Apply rotation and translation to 21 hand keypoints.
    
    Parameters:
    keypoints (numpy.ndarray): A (21, 3) array of hand keypoints.
    rotation_matrix (numpy.ndarray): A (3, 3) rotation matrix.
    translation_vector (numpy.ndarray): A (3,) translation vector.
    
    Returns:
    numpy.ndarray: The transformed keypoints as a (21, 3) array.
    """
    # Ensure keypoints are in the correct shape
    assert keypoints.shape == (21, 3), "Keypoints must be a (21, 3) array."
    assert rotation_matrix.shape == (3, 3), "Rotation matrix must be a (3, 3) array."
    assert translation_vector.shape == (3,), "Translation vector must be a (3,) array."

    # Translate keypoints so that the wrist (point 0) is at the origin
    wrist_position = keypoints[0]
    translated_keypoints = keypoints - wrist_position

    # Apply the rotation
    rotated_keypoints = np.dot(translated_keypoints, rotation_matrix.T)

    # Translate keypoints back to their original position relative to the wrist
    rotated_keypoints += wrist_position

    # Apply the final translation
    final_keypoints = rotated_keypoints + translation_vector

    return final_keypoints

def find_translation_rotation(initial_keypoints, final_keypoints):
    """
    Find the translation and rotation that transform initial keypoints to final keypoints.
    
    Parameters:
    initial_keypoints (numpy.ndarray): A (21, 3) array of initial hand keypoints.
    final_keypoints (numpy.ndarray): A (21, 3) array of final hand keypoints.
    
    Returns:
    tuple: (translation_vector, rotation_matrix) where translation_vector is a (3,) array
           and rotation_matrix is a (3, 3) array.
    """
    # Ensure keypoints are in the correct shape
    assert initial_keypoints.shape == (21, 3), "Initial keypoints must be a (21, 3) array."
    assert final_keypoints.shape == (21, 3), "Final keypoints must be a (21, 3) array."
    
    # Compute the translation vector
    initial_wrist_position = initial_keypoints[0]
    final_wrist_position = final_keypoints[0]
    translation_vector = final_wrist_position - initial_wrist_position
    
    # Translate the keypoints so that the wrist (point 0) is at the origin
    initial_translated = initial_keypoints - initial_wrist_position
    final_translated = final_keypoints - final_wrist_position
    
    # Use only the keypoints excluding the wrist to find the rotation
    initial_points = initial_translated[1:]
    final_points = final_translated[1:]
    
    # Compute the covariance matrix
    H = np.dot(initial_points.T, final_points)
    
    # Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)
    
    # Compute the rotation matrix
    rotation_matrix = np.dot(Vt.T, U.T)
    
    # Ensure the rotation matrix is a proper rotation (det(R) = 1)
    if np.linalg.det(rotation_matrix) < 0:
        Vt[2, :] *= -1
        rotation_matrix = np.dot(Vt.T, U.T)
    
    return translation_vector, rotation_matrix


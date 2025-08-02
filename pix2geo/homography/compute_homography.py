# pix2geo/homography/compute_homography.py

import numpy as np
import cv2
import utm


def latlon_to_utm(latlon_points):
    """
    Converts a list of (lat, lon) tuples to UTM coordinates.
    Returns a numpy array of shape (N, 2)
    """
    utm_points = [utm.from_latlon(lat, lon)[:2] for lat, lon in latlon_points]
    return np.array(utm_points, dtype=np.float32)


def compute_homography(pixel_points, latlon_points):
    """
    Computes the homography matrix H such that:
        [X_utm, Y_utm, 1]^T = H * [x_pix, y_pix, 1]^T

    Args:
        pixel_points: List of (x, y) pixel coordinates
        latlon_points: Corresponding list of (lat, lon) coordinates

    Returns:
        homography_matrix: 3x3 NumPy array
    """
    if len(pixel_points) < 4 or len(latlon_points) < 4:
        raise ValueError("Need at least 4 point correspondences for homography.")

    pixels = np.array(pixel_points, dtype=np.float32)
    utm_coords = latlon_to_utm(latlon_points)

    H, status = cv2.findHomography(pixels, utm_coords, method=0)
    return H


def apply_homography(pixel_point, H):
    """
    Applies a homography matrix to a single pixel coordinate.

    Args:
        pixel_point: (x, y) pixel
        H: 3x3 homography matrix

    Returns:
        (lat, lon) tuple
    """
    point = np.array([[pixel_point]], dtype=np.float32)  # Shape (1, 1, 2)
    transformed = cv2.perspectiveTransform(point, H)[0][0]  # Shape (2,)
    lat, lon = utm.to_latlon(transformed[0], transformed[1], 32, 'U')  # Adjust zone as needed
    return lat, lon


def save_homography(H, path):
    np.save(path, H)


def load_homography(path):
    return np.load(path)

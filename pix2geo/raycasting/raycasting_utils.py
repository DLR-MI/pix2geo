# pix2geo/raycasting/raycasting_utils.py
# Author: Borja Carrillo-Perez (carr_br) & Felix Sattler (sat_fe)
# Description: Core raycasting utilities for pixel-to-world georeferencing using camera pose.

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Args:
        v: A NumPy array of shape (3,) representing a 3D vector.

    Returns:
        A NumPy array of shape (3,) representing the normalized vector.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Zero-length vector cannot be normalized.")
    return v / norm


def calcRotationMatrix(theta: float, phi: float, psi: float) -> np.ndarray:
    """
    Calculate a 3D rotation matrix from Euler angles (ZYX convention).

    Args:
        theta: Rotation around X-axis (pitch) in radians.
        phi:   Rotation around Y-axis (yaw) in radians.
        psi:   Rotation around Z-axis (roll) in radians.

    Returns:
        A 3x3 NumPy array representing the rotation matrix.
    """
    # Rotation around X-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

    # Rotation around Y-axis
    Ry = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])

    # Rotation around Z-axis
    Rz = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])

    # ZYX order: roll (Z), pitch (Y), yaw (X)
    return Rz @ Ry @ Rx


def calcCameraRay(u: float, v: float, fov: float, w: int, h: int, Rt: np.ndarray) -> tuple:
    """
    Compute a world-space ray direction and origin from a pixel coordinate.

    Args:
        u: Horizontal pixel coordinate.
        v: Vertical pixel coordinate.
        fov: Field of view (vertical) of the camera in radians.
        w: Image width in pixels.
        h: Image height in pixels.
        Rt: 4x4 camera extrinsic matrix (rotation and translation).

    Returns:
        A tuple (ray_direction, ray_origin):
          - ray_direction: 3-element NumPy array, unit direction vector in world coordinates.
          - ray_origin: 3-element NumPy array, camera position in world coordinates.
    """
    # Aspect ratio
    aspect_ratio = w / float(h)
    # Tangent of half fov
    t = np.tan(fov / 2) / aspect_ratio

    # Normalize pixel to NDC space [-1, 1]
    x_ndc = (2 * ((u + 0.5) / w) - 1) * t * aspect_ratio
    y_ndc = (1 - 2 * ((v + 0.5) / h)) * t

    # Camera origin in world coords (homogeneous to cartesian)
    cam_origin_h = Rt @ np.array([0, 0, 0, 1], dtype=float)
    ray_origin = cam_origin_h[:3]

    # Point on near plane in camera coordinates
    cam_point_h = Rt @ np.array([x_ndc, y_ndc, -1.0, 1], dtype=float)
    point_world = cam_point_h[:3]

    # Direction is vector from origin to point
    ray_dir = normalize(point_world - ray_origin)
    return ray_dir, ray_origin


def intersectPlane(n: np.ndarray, p0: np.ndarray, rO: np.ndarray, rD: np.ndarray) -> tuple:
    """
    Compute the intersection of a ray with a plane.

    Plane defined by normal n and a point p0 on the plane.
    Ray defined by origin rO and direction rD.

    Args:
        n: 3-element normal vector of the plane.
        p0: 3-element point on the plane.
        rO: 3-element ray origin.
        rD: 3-element ray direction (unit vector).

    Returns:
        (has_intersection, distance):
          - has_intersection: True if ray intersects plane in front of origin.
          - distance: distance along the ray to intersection point.
    """
    denom = np.dot(n, rD)
    if abs(denom) < 1e-6:
        return False, 0.0
    d = np.dot(p0 - rO, n) / denom
    return (d >= 0), d

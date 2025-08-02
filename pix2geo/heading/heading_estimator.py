# pix2geo/heading/heading_estimator.py
# Author: Borja Carrillo-Perez (carr_br)
# Description: Functions to estimate heading of moving objects using optical flow and georeferencing, supporting both bounding boxes and masks.

import numpy as np
import cv2
import utm


def compute_optical_flow(prev_gray: np.ndarray, curr_gray: np.ndarray, **fb_params) -> np.ndarray:
    """
    Compute dense optical flow between two grayscale frames using Farneback's method.
    """
    params = {
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 15,
        'iterations': 3,
        'poly_n': 5,
        'poly_sigma': 1.2,
        'flags': 0,
    }
    params.update(fb_params)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        params['pyr_scale'], params['levels'],
                                        params['winsize'], params['iterations'],
                                        params['poly_n'], params['poly_sigma'],
                                        params['flags'])
    return flow


def extract_flow_in_bbox(flow: np.ndarray, bbox: tuple) -> tuple:
    """
    Extract flow vectors inside a bounding box (x_center, y_center, width, height).
    """
    xc, yc, w, h = bbox
    x1 = int(xc - w/2);
    y1 = int(yc - h/2);
    x2 = int(xc + w/2);
    y2 = int(yc + h/2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(flow.shape[1]-1, x2), min(flow.shape[0]-1, y2)
    fx = flow[y1:y2, x1:x2, 0].flatten()
    fy = flow[y1:y2, x1:x2, 1].flatten()
    angles = np.arctan2(fy, fx)
    magnitudes = np.hypot(fx, fy)
    valid = magnitudes > 1e-3
    return angles[valid], magnitudes[valid]


def extract_flow_in_mask(flow: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Extract flow vectors inside a binary mask region.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Mask is empty; no flow to extract.")
    fx = flow[ys, xs, 0]
    fy = flow[ys, xs, 1]
    angles = np.arctan2(fy, fx)
    magnitudes = np.hypot(fx, fy)
    valid = magnitudes > 1e-3
    return angles[valid], magnitudes[valid]


def median_flow_direction(angles: np.ndarray, magnitudes: np.ndarray) -> float:
    """
    Compute the magnitude-weighted median flow direction.
    """
    if len(angles) == 0:
        raise ValueError("No valid flow vectors to compute direction.")
    idx = np.argsort(angles)
    angles_s = angles[idx]
    mags_s = magnitudes[idx]
    cum = np.cumsum(mags_s)
    half = cum[-1] / 2.0
    i = np.searchsorted(cum, half)
    return angles_s[i]


def get_heading_from_homography(center_px: tuple, tip_px: tuple, H: np.ndarray,
                                zone_number: int, zone_letter: str) -> float:
    """
    Transform two pixels via homography, convert to lat/lon, and calculate bearing.
    """
    pts = np.array([[center_px, tip_px]], dtype=np.float32)
    utm_pts = cv2.perspectiveTransform(pts, H)[0]
    lat1, lon1 = utm.to_latlon(utm_pts[0,0], utm_pts[0,1], zone_number, zone_letter)
    lat2, lon2 = utm.to_latlon(utm_pts[1,0], utm_pts[1,1], zone_number, zone_letter)
    dLon = np.radians(lon2 - lon1)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    X = np.sin(dLon) * np.cos(phi2)
    Y = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dLon)
    bearing = (np.degrees(np.arctan2(X, Y)) + 360) % 360
    return bearing


def estimate_heading(prev_gray: np.ndarray,
                     curr_gray: np.ndarray,
                     region,
                     H: np.ndarray,
                     zone_number: int,
                     zone_letter: str,
                     region_type: str = 'bbox',
                     fb_params: dict = None) -> float:
    """
    Estimate heading for a region (bbox or mask) between two frames.

    Args:
        prev_gray, curr_gray: Grayscale frames.
        region: tuple for bbox or 2D binary mask.
        region_type: 'bbox' or 'mask'.
        H: pixel->UTM homography matrix.
        zone_number, zone_letter: UTM zone info.
        fb_params: Farneback params.

    Returns:
        heading_deg: float bearing in degrees (0-360).
    """
    flow = compute_optical_flow(prev_gray, curr_gray, **(fb_params or {}))
    if region_type == 'bbox':
        angles, mags = extract_flow_in_bbox(flow, region)
        xc, yc, w, h = region
        center = (int(xc), int(yc))
        tip = (int(xc + np.cos(median_flow_direction(angles, mags)) * (w/2)),
               int(yc + np.sin(median_flow_direction(angles, mags)) * (h/2)))
    else:
        angles, mags = extract_flow_in_mask(flow, region)
        xs, ys = np.nonzero(region)
        center = (int(xs.mean()), int(ys.mean()))
        angle = median_flow_direction(angles, mags)
        # Tip offset arbitrary length (e.g., 50px)
        tip = (int(center[0] + np.cos(angle)*50),
               int(center[1] + np.sin(angle)*50))
    return get_heading_from_homography(center, tip, H, zone_number, zone_letter)

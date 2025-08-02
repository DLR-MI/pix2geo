# pix2geo/homography/reference_selector.py
# Author: Borja Carrillo-Perez (carr_br)
# Description: Pick the reference pixel from a bbox or mask for homography georeferencing.

import numpy as np


def select_bottom_center_bbox(bbox):
    """
    Given a bounding box, returns the (x, y) pixel at its bottom-center.

    Args:
        bbox: tuple/list of (x, y, w, h) or (x1, y1, x2, y2)

    Returns:
        (x_ref, y_ref): ints
    """
    x, y, w, h = bbox
    # if bbox is (x1, y1, x2, y2), detect and convert:
    if len(bbox) == 4 and w < 1:  # heuristically detect xyxy vs xywh
        x1, y1, x2, y2 = bbox
        x_ref = int((x1 + x2) / 2)
        y_ref = int(y2)
    else:
        x_ref = int(x + w / 2)
        y_ref = int(y + h)
    return x_ref, y_ref


def select_bottom_mode_pixel(mask):
    """
    From a binary mask, finds the column (X) that occurs most often
    and returns the pixel at that X with the greatest Y.

    Args:
        mask: 2D numpy array of bool/int (True or 1 = object)

    Returns:
        (x_ref, y_ref): ints
    """
    # Find all object pixels
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Mask is empty; no pixels to select.")

    # Find mode of xs
    values, counts = np.unique(xs, return_counts=True)
    x_mode = values[np.argmax(counts)]

    # Among pixels in that column, pick the max y
    ys_in_mode = ys[xs == x_mode]
    y_ref = int(ys_in_mode.max())
    x_ref = int(x_mode)
    return x_ref, y_ref

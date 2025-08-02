# pix2geo/pix2geo/__init__.py

"""
pix2geo
———
A dual-method pixel-to-geocoordinate georeferencing toolkit for maritime situational awareness.
"""

# Package version
__version__ = "0.1.0"

# Make subpackages importable
from . import homography
from . import raycasting
from . import models
from . import heading

# expose the config loader
from .config import load_config
from .reference_selector import select_bottom_center_bbox, select_bottom_mode_pixel
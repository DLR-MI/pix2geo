# pix2geo/config.py
# Author: Borja Carrillo-Perez (carr_br)
# Description: Shared helper to load YAML/JSON config files.

import json
import yaml

def load_config(path: str) -> dict:
    """
    Load a configuration file (YAML or JSON) and return its contents as a dict.

    Args:
        path: Path to the .yaml, .yml, or .json config file.

    Returns:
        A dictionary of configuration parameters.
    """
    if path.lower().endswith(('.yaml', '.yml')):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    elif path.lower().endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError("Config file must be .yaml, .yml, or .json")
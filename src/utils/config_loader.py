"""
Configuration loader utility for YAML files.
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file
        
    Returns
    -------
    dict
        Parsed configuration dictionary
        
    Raises
    ------
    FileNotFoundError
        If the configuration file doesn't exist
    yaml.YAMLError
        If the YAML file is malformed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save a configuration dictionary to a YAML file.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary to save
    output_path : str
        Path where to save the YAML file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Parameters
    ----------
    *configs : dict
        Variable number of configuration dictionaries
        
    Returns
    -------
    dict
        Merged configuration dictionary
    """
    merged = {}
    for config in configs:
        _deep_merge(merged, config)
    return merged


def _deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """
    Recursively merge dict2 into dict1.
    
    Parameters
    ----------
    dict1 : dict
        Base dictionary
    dict2 : dict
        Dictionary to merge into dict1
        
    Returns
    -------
    dict
        Merged dictionary
    """
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            _deep_merge(dict1[key], value)
        else:
            dict1[key] = value
    return dict1

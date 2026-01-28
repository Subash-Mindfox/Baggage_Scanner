"""
Common bounding box operations.
"""

import numpy as np
from typing import List, Tuple, Dict


def transform_bbox_after_flip(
    bbox: List[int],
    flip_type: str,
    img_width: int,
    img_height: int
) -> List[int]:
    """
    Transform bounding box coordinates after flip operation.
    
    Parameters
    ----------
    bbox : list
        Bounding box [xmin, ymin, xmax, ymax]
    flip_type : str
        Type of flip: 'hflip', 'vflip', or 'hvflip'
    img_width : int
        Image width
    img_height : int
        Image height
        
    Returns
    -------
    list
        Transformed bounding box [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, xmax, ymax = bbox
    
    if flip_type == 'hflip':
        # Horizontal flip: adjust x-coordinates
        new_xmin = img_width - xmax
        new_xmax = img_width - xmin
        new_ymin, new_ymax = ymin, ymax
    elif flip_type == 'vflip':
        # Vertical flip: adjust y-coordinates
        new_xmin, new_xmax = xmin, xmax
        new_ymin = img_height - ymax
        new_ymax = img_height - ymin
    elif flip_type == 'hvflip':
        # Both horizontal and vertical flip
        new_xmin = img_width - xmax
        new_xmax = img_width - xmin
        new_ymin = img_height - ymax
        new_ymax = img_height - ymin
    else:
        raise ValueError(f"Unknown flip type: {flip_type}")
    
    # Ensure coordinates are in correct order
    new_xmin, new_xmax = sorted([new_xmin, new_xmax])
    new_ymin, new_ymax = sorted([new_ymin, new_ymax])
    
    return [new_xmin, new_ymin, new_xmax, new_ymax]


def transform_bbox_after_rotation(
    bbox: List[int],
    angle: float,
    img_width: int,
    img_height: int
) -> List[int]:
    """
    Transform bounding box coordinates after rotation.
    
    Parameters
    ----------
    bbox : list
        Bounding box [xmin, ymin, xmax, ymax]
    angle : float
        Rotation angle in degrees
    img_width : int
        Image width
    img_height : int
        Image height
        
    Returns
    -------
    list
        Transformed bounding box [xmin, ymin, xmax, ymax]
    """
    # This is a simplified version - for accurate bbox transformation after rotation,
    # you should rotate all 4 corners and compute the new bounding box
    import cv2
    
    xmin, ymin, xmax, ymax = bbox
    
    # Get rotation matrix
    center = (img_width / 2, img_height / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate all 4 corners
    corners = np.array([
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax]
    ])
    
    ones = np.ones(shape=(len(corners), 1))
    corners_homogeneous = np.hstack([corners, ones])
    
    # Apply rotation
    rotated_corners = M.dot(corners_homogeneous.T).T
    
    # Get new bounding box
    new_xmin = int(np.min(rotated_corners[:, 0]))
    new_xmax = int(np.max(rotated_corners[:, 0]))
    new_ymin = int(np.min(rotated_corners[:, 1]))
    new_ymax = int(np.max(rotated_corners[:, 1]))
    
    return [new_xmin, new_ymin, new_xmax, new_ymax]


def clip_bbox_to_image(
    bbox: List[int],
    img_width: int,
    img_height: int
) -> List[int]:
    """
    Clip bounding box coordinates to image boundaries.
    
    Parameters
    ----------
    bbox : list
        Bounding box [xmin, ymin, xmax, ymax]
    img_width : int
        Image width
    img_height : int
        Image height
        
    Returns
    -------
    list
        Clipped bounding box [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, xmax, ymax = bbox
    
    xmin = max(0, min(xmin, img_width))
    xmax = max(0, min(xmax, img_width))
    ymin = max(0, min(ymin, img_height))
    ymax = max(0, min(ymax, img_height))
    
    return [xmin, ymin, xmax, ymax]


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Parameters
    ----------
    bbox1 : list
        First bounding box [xmin, ymin, xmax, ymax]
    bbox2 : list
        Second bounding box [xmin, ymin, xmax, ymax]
        
    Returns
    -------
    float
        IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection area
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou


def scale_bbox(
    bbox: List[int],
    scale_x: float,
    scale_y: float
) -> List[int]:
    """
    Scale bounding box coordinates.
    
    Parameters
    ----------
    bbox : list
        Bounding box [xmin, ymin, xmax, ymax]
    scale_x : float
        Scaling factor for x coordinates
    scale_y : float
        Scaling factor for y coordinates
        
    Returns
    -------
    list
        Scaled bounding box [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, xmax, ymax = bbox
    
    new_xmin = int(xmin * scale_x)
    new_xmax = int(xmax * scale_x)
    new_ymin = int(ymin * scale_y)
    new_ymax = int(ymax * scale_y)
    
    return [new_xmin, new_ymin, new_xmax, new_ymax]


def bbox_area(bbox: List[int]) -> int:
    """
    Calculate the area of a bounding box.
    
    Parameters
    ----------
    bbox : list
        Bounding box [xmin, ymin, xmax, ymax]
        
    Returns
    -------
    int
        Area of the bounding box
    """
    xmin, ymin, xmax, ymax = bbox
    return (xmax - xmin) * (ymax - ymin)


def is_valid_bbox(bbox: List[int], min_size: int = 1) -> bool:
    """
    Check if a bounding box is valid.
    
    Parameters
    ----------
    bbox : list
        Bounding box [xmin, ymin, xmax, ymax]
    min_size : int
        Minimum width/height for a valid bbox
        
    Returns
    -------
    bool
        True if bbox is valid, False otherwise
    """
    xmin, ymin, xmax, ymax = bbox
    
    width = xmax - xmin
    height = ymax - ymin
    
    return width >= min_size and height >= min_size

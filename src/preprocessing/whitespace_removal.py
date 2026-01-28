"""
Whitespace removal preprocessing for X-ray images.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from src.common.image_utils import get_whiteness


class WhitespaceRemover:
    """
    Configurable whitespace removal from X-ray images.
    
    This class handles the removal of white borders/backgrounds from X-ray images
    and provides the transformation maps for adjusting bounding boxes.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize whitespace remover with configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary with preprocessing parameters
        """
        self.enabled = config.get('enabled', True)
        params = config.get('params', {})
        
        self.do_vertical = params.get('do_vertical', True)
        self.do_horizontal = params.get('do_horizontal', True)
        self.v_threshold = params.get('v_whiteness_threshold', 99.9)
        self.h_threshold = params.get('h_whiteness_threshold', 99.9)
        self.seg_width = params.get('seg_width', 1)
        self.tolerance = params.get('whiteness_tolerance', 33)
    
    def process(
        self,
        image_np: np.ndarray
    ) -> Tuple[Optional[int], Optional[Dict], Optional[Dict], np.ndarray]:
        """
        Remove whitespace if enabled, otherwise return original.
        
        Parameters
        ----------
        image_np : np.ndarray
            Input image as numpy array
            
        Returns
        -------
        tuple
            (seg_size, v_transform_dict, h_transform_dict, processed_image)
            If disabled, returns (None, None, None, original_image)
        """
        if not self.enabled:
            return None, None, None, image_np
        
        return remove_white(
            im=image_np,
            do_vertical=self.do_vertical,
            do_horizontal=self.do_horizontal,
            v_whiteness_threshold=self.v_threshold,
            h_whiteness_threshold=self.h_threshold,
            seg_wid=self.seg_width,
            whiteness_tolerance=self.tolerance
        )


def reflect(old_w: int, w_max: int) -> int:
    """
    Reflect a coordinate around the center of the image.
    
    Parameters
    ----------
    old_w : int
        Original coordinate
    w_max : int
        Maximum coordinate (image dimension)
        
    Returns
    -------
    int
        Reflected coordinate
    """
    mid_w = w_max // 2
    if mid_w < old_w:
        new_w = old_w - 2 * (old_w - mid_w)
    else:
        new_w = old_w + 2 * (mid_w - old_w)
    return new_w


def get_nearest_obj(tran_dic: Dict[int, int], seg_X: int) -> int:
    """
    Find the nearest non-removed segment.
    
    Parameters
    ----------
    tran_dic : dict
        Transformation dictionary mapping old segments to new segments
    seg_X : int
        Segment index to find nearest neighbor for
        
    Returns
    -------
    int
        Index of nearest non-removed segment
    """
    seg_X = int(seg_X)
    
    if tran_dic[seg_X] != -1:
        return seg_X
    
    # Search backwards
    bfl = list(tran_dic.keys())[:seg_X]
    bfl = bfl[::-1]
    bf_counter = 0
    bf_seg = -1
    
    for bf in bfl:
        bf_counter += 1
        if tran_dic[bf] != -1:
            bf_seg = bf
            break
    
    if bf_seg == -1:
        bf_counter = 9999
    
    # Search forwards
    afl = list(tran_dic.keys())[seg_X + 1:]
    af_counter = 0
    af_seg = -1
    
    for af in afl:
        af_counter += 1
        if tran_dic[af] != -1:
            af_seg = af
            break
    
    if af_seg == -1:
        af_counter = 9999
    
    # Return nearest
    if bf_counter <= af_counter:
        return bf_seg
    else:
        return af_seg


def rotated_this(old_x: int, old_y: int, xmax: int, ymax: int) -> Tuple[int, int]:
    """
    Calculate rotated coordinates (90-degree rotation).
    
    Parameters
    ----------
    old_x : int
        Original x coordinate
    old_y : int
        Original y coordinate
    xmax : int
        Maximum x value
    ymax : int
        Maximum y value
        
    Returns
    -------
    tuple
        (new_x, new_y) rotated coordinates
    """
    new_org_x, new_org_y = (0, xmax)
    return (new_org_x + old_y, new_org_y - old_x)


def remove_white(
    im: np.ndarray,
    do_vertical: bool = True,
    do_horizontal: bool = True,
    v_whiteness_threshold: float = 80,
    h_whiteness_threshold: float = 80,
    seg_wid: int = 10,
    whiteness_tolerance: int = 33
) -> Tuple[int, Dict[int, int], Dict[int, int], np.ndarray]:
    """
    Remove white borders from X-ray images.
    
    Parameters
    ----------
    im : np.ndarray
        Input image
    do_vertical : bool
        Whether to remove vertical white space
    do_horizontal : bool
        Whether to remove horizontal white space
    v_whiteness_threshold : float
        Threshold for vertical whiteness (0-100)
    h_whiteness_threshold : float
        Threshold for horizontal whiteness (0-100)
    seg_wid : int
        Segment width for analysis
    whiteness_tolerance : int
        Tolerance for considering a pixel as white
        
    Returns
    -------
    tuple
        (seg_width, v_transform_dict, h_transform_dict, processed_image)
    """
    v_tran = {}
    h_tran = {}
    
    # Vertical whitespace removal
    if do_vertical:
        img_hig = im.shape[0]
        num_of_seg = int(img_hig / seg_wid)
        seg_num, iter_count, br, all_where = 0, 0, 0, 0
        v_dic = dict(zip(range(num_of_seg), [-1] * num_of_seg))
        
        while seg_num <= num_of_seg - br + 2:
            if all_where >= img_hig:
                break
            
            this_seg = im[seg_num * seg_wid:(seg_num + 1) * seg_wid, :]
            
            if get_whiteness(this_seg, whiteness_tolerance) > v_whiteness_threshold:
                br += 1
            else:
                v_dic[seg_num] = iter_count
                iter_count += 1
                all_where += seg_wid
            
            seg_num += 1
        
        # Crop vertically
        no_white_v_rows = []
        for k, v in v_dic.items():
            if v != -1:
                no_white_v_rows.extend(range(k * seg_wid, (k + 1) * seg_wid))
        
        im = im[no_white_v_rows, :]
        v_tran = v_dic
    else:
        # No vertical removal
        num_of_seg = int(im.shape[0] / seg_wid)
        v_tran = dict(zip(range(num_of_seg), range(num_of_seg)))
    
    # Horizontal whitespace removal
    if do_horizontal:
        img_wid = im.shape[1]
        num_of_seg = int(img_wid / seg_wid)
        seg_num, iter_count, br, all_where = 0, 0, 0, 0
        h_dic = dict(zip(range(num_of_seg), [-1] * num_of_seg))
        
        while seg_num <= num_of_seg - br + 2:
            if all_where >= img_wid:
                break
            
            this_seg = im[:, seg_num * seg_wid:(seg_num + 1) * seg_wid]
            
            if get_whiteness(this_seg, whiteness_tolerance) > h_whiteness_threshold:
                br += 1
            else:
                h_dic[seg_num] = iter_count
                iter_count += 1
                all_where += seg_wid
            
            seg_num += 1
        
        # Crop horizontally
        no_white_h_cols = []
        for k, v in h_dic.items():
            if v != -1:
                no_white_h_cols.extend(range(k * seg_wid, (k + 1) * seg_wid))
        
        im = im[:, no_white_h_cols]
        h_tran = h_dic
    else:
        # No horizontal removal
        num_of_seg = int(im.shape[1] / seg_wid)
        h_tran = dict(zip(range(num_of_seg), range(num_of_seg)))
    
    return seg_wid, v_tran, h_tran, im

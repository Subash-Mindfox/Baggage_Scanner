"""
Preprocessing script - Remove whitespace and transform images.

This script loads raw data, removes whitespace from X-ray images,
transforms bounding boxes, and saves preprocessed data.
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import csv
import xml.etree.ElementTree as ET
from torchvision.io import read_image
import torch.nn.functional as F
import torch

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.preprocessing.whitespace_removal import WhitespaceRemover, get_nearest_obj, rotated_this, reflect
from src.common.image_utils import torch_to_np, np_to_torch, save_tensor_as_image
from src.data.statistics import get_image_dataframe_lazy, compute_csv_stats


class BaseTransform:
    """Base image transformation (resize and color adjustment)."""
    
    def __init__(self, resize=(584, 688), rgb_means=(104, 117, 123), transform_type="medium"):
        """
        Initialize base transform.
        
        Parameters
        ----------
        resize : tuple
            Target size (height, width)
        rgb_means : tuple
            RGB mean values for normalization
        transform_type : str
            Type of transform: "light", "medium", or "dark"
        """
        self.resize = resize
        self.rgb_means = torch.tensor(rgb_means).view(3, 1, 1)
        self.transform_type = transform_type
    
    def __call__(self, img):
        """Apply transformation to image."""
        img = self.resize_image(img)
        
        if self.transform_type == "light":
            img = img + self.rgb_means
        elif self.transform_type == "dark":
            img = img - self.rgb_means
        
        return img
    
    def resize_image(self, img):
        """Resize image to target size."""
        img = F.interpolate(
            img.unsqueeze(0),
            size=self.resize,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        return img


def create_csv_if_not_exists(csv_path):
    """
    Create CSV file with headers if it doesn't exist.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file
    """
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename", "width", "height", "depth",
                "class",
                "xmin", "ymin", "xmax", "ymax"
            ])


def process_dataset(
    df_images,
    csv_path,
    dataset_name,
    whitespace_remover,
    base_transform,
    output_dir,
    annotations_dir,
    logger
):
    """
    Process dataset: remove whitespace, transform images, adjust bboxes.
    
    Parameters
    ----------
    df_images : pd.DataFrame
        DataFrame containing image metadata
    csv_path : str
        Path to save annotations CSV
    dataset_name : str
        Name of dataset (Train/Test)
    whitespace_remover : WhitespaceRemover
        Whitespace removal processor
    base_transform : BaseTransform
        Image transformation
    output_dir : str
        Directory to save processed images
    annotations_dir : str
        Directory containing XML annotations
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    tuple
        (new_image_metadata, filenames_list, object_locations_list)
    """
    count = len(df_images)
    
    new_image_crop = []
    filenames_list = []
    object_locations_list = []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure CSV exists
    create_csv_if_not_exists(csv_path)
    
    logger.info(f"Processing {count} images from {dataset_name}...")
    
    for fi in range(count):
        try:
            # Read image
            ann_pic_path = df_images["path"].iloc[fi]
            ann_pic_name = df_images["filename"].iloc[fi]
            
            ann_pic = read_image(ann_pic_path)
            filenames_list.append(ann_pic_name)
            
            # Convert to numpy for whitespace removal
            ann_pic_np = torch_to_np(ann_pic)
            
            # Remove whitespace
            seg_size, v_tran, h_tran, no_whi_np = whitespace_remover.process(ann_pic_np)
            
            # Read XML annotations
            xml_path = os.path.join(annotations_dir, ann_pic_name[:-4] + ".xml")
            
            if not os.path.exists(xml_path):
                logger.warning(f"XML not found for {ann_pic_name}, skipping...")
                continue
            
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract bounding boxes
            dim_dic = {}
            count_obj = 0
            
            for child in root:
                if child.tag == "object":
                    bbox = child.find("bndbox")
                    dim_dic[count_obj] = {
                        "xmin": int(bbox.find("xmin").text),
                        "ymin": int(bbox.find("ymin").text),
                        "xmax": int(bbox.find("xmax").text),
                        "ymax": int(bbox.find("ymax").text),
                        "name": child.find("name").text
                    }
                    count_obj += 1
            
            # Original bounding boxes
            bbox0 = np.array([
                [v["ymin"], v["xmin"], v["ymax"], v["xmax"]]
                for v in dim_dic.values()
            ])
            
            # Transform bounding boxes if whitespace was removed
            w_bbox = []
            for box in bbox0:
                y_min, x_min, y_max, x_max = box
                
                # Map to segments
                y_min_seg = max(y_min // seg_size, min(v_tran.keys()))
                x_min_seg = max(x_min // seg_size, min(h_tran.keys()))
                y_max_seg = min(y_max // seg_size, max(v_tran.keys()))
                x_max_seg = min(x_max // seg_size, max(h_tran.keys()))
                
                # Find nearest non-removed segments
                y_min_seg = resolve_seg(v_tran, y_min_seg)
                y_max_seg = resolve_seg(v_tran, y_max_seg)
                x_min_seg = resolve_seg(h_tran, x_min_seg)
                x_max_seg = resolve_seg(h_tran, x_max_seg)
                
                # Calculate new coordinates
                w_ymin = seg_size * v_tran[y_min_seg] + y_min % seg_size
                w_xmin = seg_size * h_tran[x_min_seg] + x_min % seg_size
                w_ymax = seg_size * v_tran[y_max_seg] + y_max % seg_size
                w_xmax = seg_size * h_tran[x_max_seg] + x_max % seg_size
                
                w_bbox.append([w_xmin, w_ymin, w_xmax, w_ymax])
            
            # Rotate boxes (90-degree rotation)
            rot_boxes = []
            for box in w_bbox:
                x1, y1, x2, y2 = box
                rx1, ry1 = rotated_this(x1, y1, no_whi_np.shape[1], no_whi_np.shape[0])
                rx2, ry2 = rotated_this(x2, y2, no_whi_np.shape[1], no_whi_np.shape[0])
                rot_boxes.append([ry2, rx1, ry1, rx2])
            
            # Reflect boxes
            h_max, w_max, _ = no_whi_np.shape
            W_boxes = []
            for bx in rot_boxes:
                    wmin_, hmin_, wmax_, hmax_ = bx
                    W_boxes.append([
                        reflect(wmax_, w_max),
                        hmin_,
                        reflect(wmin_, w_max),
                        hmax_
                    ])
           
            
            # Apply base transform
            transformed_pic = base_transform(np_to_torch(no_whi_np))
            transformed_pic_np = torch_to_np(transformed_pic)
            
            # Resize bounding boxes to match transformed image
            Hratio = transformed_pic_np.shape[1] / no_whi_np.shape[1]
            Wratio = transformed_pic_np.shape[0] / no_whi_np.shape[0]
            ratioLst = [Wratio, Hratio, Wratio, Hratio]
            
            w_bbox_r = [
                [int(abs(a * b)) for a, b in zip(box, ratioLst)]
                for box in W_boxes
            ]
            
            # Create final object location dictionary
            labels = [v["name"] for v in dim_dic.values()]
            final_obj_location_dict = {
                i: [labels[i], w_bbox_r[i]]
                for i in range(len(labels))
            }
            
            # Store output
            object_locations_list.append(final_obj_location_dict)
            
            # Save transformed image
            file_path = os.path.join(output_dir, ann_pic_name)
            shape, mode = save_tensor_as_image(transformed_pic, file_path)
            
            new_image_crop.append([
                ann_pic_name,
                file_path,
                shape[0],  # width
                shape[1],  # height
                mode
            ])
            
            # Write annotations to CSV
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                for k in final_obj_location_dict:
                    label, box = final_obj_location_dict[k]
                    xmin, ymin, xmax, ymax = box
                    
                    writer.writerow([
                        ann_pic_name,
                        shape[0],  # width
                        shape[1],  # height
                        3,  # depth (RGB)
                        label,
                        xmin,
                        ymin,
                        xmax,
                        ymax
                    ])
            
            logger.info(f"[{dataset_name.upper()}] {fi+1}/{count} - {ann_pic_name}")
        
        except Exception as e:
            logger.error(f"[{dataset_name.upper()} ERROR] {fi}/{count} - {e}")
    
    return new_image_crop, filenames_list, object_locations_list

def resolve_seg(tran, seg):
    seg = min(seg, len(tran) - 1)   # clamp to last index
    return get_nearest_obj(tran, seg) if tran[seg] == -1 else seg

def main():
    """Main preprocessing function."""
    
    # Setup logger
    logger = setup_logger('preprocessing', 'outputs/logs/preprocessing.log')
    logger.info("Starting preprocessing pipeline")
    
    # Load configurations
    paths_config = load_config('configs/paths.yaml')
    preprocessing_config = load_config('configs/preprocessing.yaml')
    
    # Create whitespace remover
    whitespace_remover = WhitespaceRemover(
        preprocessing_config['preprocessing']['whitespace_removal']
    )
    logger.info(f"Whitespace removal enabled: {whitespace_remover.enabled}")
    
    # Create base transform
    transform_config = preprocessing_config['preprocessing']['base_transform']
    base_transform = BaseTransform(
        resize=tuple(transform_config['resize']),
        rgb_means=tuple(transform_config['rgb_means']),
        transform_type=transform_config['transform_type']
    )
    logger.info(f"Base transform: {transform_config['transform_type']}")
    
    # Load raw data statistics
    logger.info("\n=== Loading Training Data ===")
    df_train_images = get_image_dataframe_lazy(
        paths_config['paths']['trainingImages'],
        "Training Images"
    )
    df_train_annotations = compute_csv_stats(
        paths_config['paths']['trainningAnnot_file'],
        'Training Annotations'
    )
    
    logger.info("\n=== Loading Test Data ===")
    df_test_images = get_image_dataframe_lazy(
        paths_config['paths']['testImages'],
        "Testing Images"
    )
    df_test_annotations = compute_csv_stats(
        paths_config['paths']['testAnnot_file'],
        'Testing Annotations'
    )
    
    # Process training data
    logger.info("\n" + "="*80)
    logger.info("PROCESSING TRAINING DATA")
    logger.info("="*80)
    
    train_output_dir = paths_config['paths']['preprocessed_train']
    train_csv_path = os.path.join(train_output_dir, "_annotations.csv")
    
    train_images, train_files, train_objs = process_dataset(
        df_images=df_train_images,
        csv_path=train_csv_path,
        dataset_name="Train",
        whitespace_remover=whitespace_remover,
        base_transform=base_transform,
        output_dir=train_output_dir,
        annotations_dir=paths_config['paths']['trainingAnnotations'],
        logger=logger
    )
    
    # Process test data
    logger.info("\n" + "="*80)
    logger.info("PROCESSING TEST DATA")
    logger.info("="*80)
    
    test_output_dir = paths_config['paths']['preprocessed_test']
    test_csv_path = os.path.join(test_output_dir, "_annotations.csv")
    
    test_images, test_files, test_objs = process_dataset(
        df_images=df_test_images,
        csv_path=test_csv_path,
        dataset_name="Test",
        whitespace_remover=whitespace_remover,
        base_transform=base_transform,
        output_dir=test_output_dir,
        annotations_dir=paths_config['paths']['testAnnotations'],
        logger=logger
    )
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("="*80)
    logger.info(f"Training images processed: {len(train_images)}")
    logger.info(f"Test images processed: {len(test_images)}")
    logger.info(f"Training annotations: {train_csv_path}")
    logger.info(f"Test annotations: {test_csv_path}")
    logger.info("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()

"""
Model evaluation script.

This script evaluates a trained model on the test dataset.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from src.utils.config_loader import load_config
from src.utils.logger import get_experiment_logger
from src.models.faster_rcnn import FasterRCNNModel
from src.common.bbox_utils import calculate_iou


def calculate_precision_recall(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate precision and recall for object detection.
    
    Parameters
    ----------
    predictions : list
        List of predicted bounding boxes with scores
    ground_truths : list
        List of ground truth bounding boxes
    iou_threshold : float
        IoU threshold for considering a detection as correct
        
    Returns
    -------
    tuple
        (precision, recall)
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    matched_gt = set()
    
    # Sort predictions by score (descending)
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    for pred in predictions:
        pred_bbox = pred['bbox']
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching ground truth
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            
            gt_bbox = gt['bbox']
            iou = calculate_iou(pred_bbox, gt_bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if detection is correct
        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)
        else:
            false_positives += 1
    
    # Count missed ground truths
    false_negatives = len(ground_truths) - len(matched_gt)
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall


def calculate_map(predictions_list, ground_truths_list, iou_thresholds=[0.5, 0.75]):
    """
    Calculate mean Average Precision (mAP).
    
    Parameters
    ----------
    predictions_list : list
        List of predictions for all images
    ground_truths_list : list
        List of ground truths for all images
    iou_thresholds : list
        List of IoU thresholds to evaluate
        
    Returns
    -------
    dict
        Dictionary with mAP values for each threshold
    """
    results = {}
    
    for iou_thresh in iou_thresholds:
        precisions = []
        recalls = []
        
        for preds, gts in zip(predictions_list, ground_truths_list):
            precision, recall = calculate_precision_recall(preds, gts, iou_thresh)
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate average precision
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        
        results[f'mAP@{iou_thresh}'] = avg_precision
        results[f'recall@{iou_thresh}'] = avg_recall
    
    return results


def evaluate_model(model, dataloader, device, logger, iou_thresholds=[0.5, 0.75]):
    """
    Evaluate model on test dataset.
    
    Parameters
    ----------
    model : BaseDetectionModel
        Model to evaluate
    dataloader : DataLoader
        Test data loader
    device : torch.device
        Device to run evaluation on
    logger : logging.Logger
        Logger instance
    iou_thresholds : list
        IoU thresholds for evaluation
        
    Returns
    -------
    dict
        Evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    logger.info("Running inference on test set...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Move to device
            images = [img.to(device) for img in images]
            
            # Get predictions
            predictions = model.predict(images)
            
            # Collect predictions and ground truths
            for pred, target in zip(predictions, targets):
                # Convert predictions to standard format
                pred_list = []
                for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
                    pred_list.append({
                        'bbox': box.cpu().numpy().tolist(),
                        'score': score.item(),
                        'label': label.item()
                    })
                
                # Convert ground truths to standard format
                gt_list = []
                for box, label in zip(target['boxes'], target['labels']):
                    gt_list.append({
                        'bbox': box.cpu().numpy().tolist(),
                        'label': label.item()
                    })
                
                all_predictions.append(pred_list)
                all_ground_truths.append(gt_list)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = calculate_map(all_predictions, all_ground_truths, iou_thresholds)
    
    return metrics


def main():
    """Main evaluation function."""
    
    # Setup experiment logger
    logger = get_experiment_logger('evaluation', 'outputs/logs')
    logger.info("Starting model evaluation")
    
    # Load configurations
    paths_config = load_config('configs/paths.yaml')
    model_config = load_config('configs/model_config.yaml')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Loading model...")
    model = FasterRCNNModel(model_config['model'])
    model.to_device(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(
        paths_config['paths']['checkpoints'],
        'best_model.pth'  # Adjust this to your checkpoint name
    )
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        logger.error("Please train a model first using 04_train.py")
        return
    
    checkpoint = model.load_checkpoint(checkpoint_path)
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # TODO: Create test DataLoader
    # test_loader = create_dataloader(df_test_images, df_test_annotations,
    #                                 batch_size=model_config['validation']['val_batch_size'])
    
    # Evaluation
    logger.info("\n" + "="*80)
    logger.info("RUNNING EVALUATION")
    logger.info("="*80)
    
    logger.info("Note: Implement proper Dataset and DataLoader before evaluation")
    
    # iou_thresholds = model_config['validation']['iou_thresholds']
    # metrics = evaluate_model(model, test_loader, device, logger, iou_thresholds)
    # 
    # # Print results
    # logger.info("\n" + "="*80)
    # logger.info("EVALUATION RESULTS")
    # logger.info("="*80)
    # 
    # for metric_name, value in metrics.items():
    #     logger.info(f"{metric_name}: {value:.4f}")
    # 
    # # Save metrics
    # metrics_file = os.path.join(
    #     paths_config['paths']['metrics'],
    #     'evaluation_metrics.txt'
    # )
    # os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    # 
    # with open(metrics_file, 'w') as f:
    #     for metric_name, value in metrics.items():
    #         f.write(f"{metric_name}: {value:.4f}\n")
    # 
    # logger.info(f"\nMetrics saved to {metrics_file}")
    
    logger.info("\nEvaluation script template created. Implement Dataset class to begin evaluation.")


if __name__ == "__main__":
    main()

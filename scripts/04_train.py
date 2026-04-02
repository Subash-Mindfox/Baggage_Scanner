import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import json
from tqdm import tqdm
import numpy as np
from functools import partial
import traceback

from src.models.faster_rcnn_with_stn import create_model
from src.data.dataset import create_dataloaders
from src.data.statistics import get_image_dataframe_lazy
from src.utils.config_loader import load_config
from src.utils.logger import get_experiment_logger
from sklearn.utils.class_weight import compute_class_weight
from src.data.dataset import BaggageXRayDataset, collate_fn
import pandas as pd


def filter_invalid_images_and_boxes(images, targets):
    """
    Filter out images with invalid or empty bounding boxes.
    
    Parameters
    ----------
    images : list of torch.Tensor
        List of images
    targets : list of dict
        List of targets with 'boxes' and 'labels'
        
    Returns
    -------
    tuple
        (filtered_images, filtered_targets)
    """
    valid_images = []
    valid_targets = []
    
    for img, tgt in zip(images, targets):
        # Check if boxes exist and are valid
        if len(tgt['boxes']) > 0:
            boxes = tgt['boxes']
            
            # Filter boxes where xmax > xmin and ymax > ymin
            valid_box_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            
            if valid_box_mask.sum() > 0:
                valid_images.append(img)
                valid_targets.append({
                    'boxes': boxes[valid_box_mask],
                    'labels': tgt['labels'][valid_box_mask]
                })
    
    return valid_images, valid_targets


def custom_fastrcnn_loss(class_logits, box_regression, labels, regression_targets, class_weights_tensor, device):
    """
    Custom Faster R-CNN loss with class weights for imbalanced datasets.
    
    Parameters
    ----------
    class_logits : torch.Tensor
        Predicted class logits
    box_regression : torch.Tensor
        Predicted box regressions
    labels : torch.Tensor
        Ground truth labels
    regression_targets : torch.Tensor
        Ground truth box regression targets
    class_weights_tensor : torch.Tensor
        Weights for each class
    device : torch.device
        Device to run on
        
    Returns
    -------
    tuple
        (classification_loss, box_regression_loss)
    """
    import torch.nn.functional as F
    from torchvision.ops import boxes as box_ops
    
    # Classification loss with class weights
    classification_loss = F.cross_entropy(
        class_logits,
        labels,
        weight=class_weights_tensor.to(device)
    )
    
    # Box regression loss (only for positive examples)
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N = class_logits.shape[0]
    box_regression = box_regression.reshape(N, -1, 4)
    
    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction='sum'
    )
    box_loss = box_loss / labels.numel()
    
    return classification_loss, box_loss


def get_num_classes_from_annotations(df_annotations, logger):
    """
    Automatically get number of classes from annotations.
    
    Returns
    -------
    tuple
        (num_classes, class_to_idx, idx_to_class)
    """
    # Get unique classes from annotations
    unique_classes = sorted(df_annotations['class'].unique())
    
    # Create mapping (background is class 0)
    class_to_idx = {'background': 0}
    for idx, cls in enumerate(unique_classes, start=1):
        class_to_idx[cls] = idx
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Total number of classes (including background)
    num_classes = len(class_to_idx)
    
    logger.info(f"📊 Detected {num_classes} classes (including background):")
    for cls, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        logger.info(f"     {idx}: {cls}")
    
    return num_classes, class_to_idx, idx_to_class

def calculate_class_weights(df_annotations, class_to_idx, logger):
    """
    Calculate class weights for imbalanced datasets.
    
    Returns
    -------
    torch.Tensor
        Class weights tensor
    """
    logger.info("Calculating class weights for imbalanced data...")
    
    # Get all labels
    labels = df_annotations['class'].map(class_to_idx).values
    
    # Get unique classes (excluding background)
    classes = np.array([class_to_idx[cls] for cls in class_to_idx.keys() if cls != 'background'])
    
    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    
    # Add background weight (usually 1.0)
    weights_dict = {0: 1.0}  # Background
    for cls, weight in zip(classes, class_weights):
        weights_dict[cls] = weight
    
    # Create tensor
    num_classes = len(class_to_idx)
    weights_tensor = torch.zeros(num_classes)
    for cls_idx, weight in weights_dict.items():
        weights_tensor[cls_idx] = weight
    
    logger.info("Class weights calculated:")
    for cls_idx, weight in weights_dict.items():
        cls_name = {v: k for k, v in class_to_idx.items()}[cls_idx]
        logger.info(f"     {cls_name}: {weight:.4f}")
    
    return weights_tensor

def normalize_images(images):
    """
    Normalize images to [0, 1] range.
    
    Parameters
    ----------
    images : list of torch.Tensor
        List of image tensors
        
    Returns
    -------
    list of torch.Tensor
        Normalized images
    """
    normalized = []
    for img in images:
        if img.max() > 1.0:
            img = img.float() / 255.0
        normalized.append(img)
    return normalized

def save_transformed_tensors(tensors, transformation_names, save_dir, batch_idx):
    """
    Save transformed tensors for visualization.
    
    Useful for seeing what transformations the STN is learning.
    
    Parameters
    ----------
    tensors : list of torch.Tensor
        Transformed image tensors
    transformation_names : list of str
        Names of transformations
    save_dir : str
        Directory to save tensors
    batch_idx : int
        Batch index
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for idx, (tensor, transform_name) in enumerate(zip(tensors, transformation_names)):
        tensor_filename = f"{batch_idx}_tensor_{idx}_{transform_name}.pt"
        save_path = os.path.join(save_dir, tensor_filename)
        torch.save(tensor.cpu(), save_path)

def load_all_training_data(paths_config, logger):
    """
    Load ALL training data: preprocessed + augmented.
    
    Returns
    -------
    tuple
        (df_train_images, df_train_annotations)
    """
    all_train_images = []
    all_train_annotations = []
    
    # Load preprocessed training data
    logger.info("Loading preprocessed training data...")
    preprocessed_train = paths_config['paths']['preprocessed_train']
    preprocessed_csv = os.path.join(preprocessed_train, "_annotations.csv")
    
    if os.path.exists(preprocessed_csv):
        df_prep_images = get_image_dataframe_lazy(preprocessed_train, "Preprocessed Train")
        df_prep_annotations = pd.read_csv(preprocessed_csv)
        all_train_images.append(df_prep_images)
        all_train_annotations.append(df_prep_annotations)
        logger.info(f"  Loaded {len(df_prep_images)} preprocessed images")
    
    # Load ALL augmented data
    augmented_root = paths_config['paths']['augmented_root']
    
    if os.path.exists(augmented_root):
        logger.info("Loading augmented data...")
        
        # Get all augmented images from ALL subdirectories
        augmented_subdirs = [d for d in os.listdir(augmented_root) 
                           if os.path.isdir(os.path.join(augmented_root, d))]
        
        for subdir in augmented_subdirs:
            subdir_path = os.path.join(augmented_root, subdir)
            df_aug_images = get_image_dataframe_lazy(subdir_path, f"Augmented {subdir}")
            if len(df_aug_images) > 0:
                all_train_images.append(df_aug_images)
                logger.info(f"  Loaded {len(df_aug_images)} images from {subdir}")
        
        # Load augmented annotations
        augmented_csv = os.path.join(augmented_root, 'augmented_annotations.csv')
        if os.path.exists(augmented_csv):
            df_aug_annotations = pd.read_csv(augmented_csv)
            all_train_annotations.append(df_aug_annotations)
            logger.info(f"  Total augmented annotations: {len(df_aug_annotations)}")
    
    # Combine ALL training data (preprocessed + augmented)
    df_train_images = pd.concat(all_train_images, ignore_index=True) if all_train_images else pd.DataFrame()
    df_train_annotations = pd.concat(all_train_annotations, ignore_index=True) if all_train_annotations else pd.DataFrame()
    
    logger.info(f"✅ Total training data: {len(df_train_images)} images, {len(df_train_annotations)} annotations")
    
    return df_train_images, df_train_annotations

def load_test_data(paths_config, logger):
    """
    Load test data for validation.
    
    Returns
    -------
    tuple
        (df_test_images, df_test_annotations)
    """
    logger.info("Loading test data for validation...")
    preprocessed_test = paths_config['paths']['preprocessed_test']
    test_csv = os.path.join(preprocessed_test, "_annotations.csv")
    
    if os.path.exists(test_csv):
        df_test_images = get_image_dataframe_lazy(preprocessed_test, "Test")
        df_test_annotations = pd.read_csv(test_csv)
        logger.info(f"✅ Loaded {len(df_test_images)} test images for validation")
    else:
        df_test_images = pd.DataFrame()
        df_test_annotations = pd.DataFrame()
        logger.warning("⚠️  No test data found!")
    
    return df_test_images, df_test_annotations

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    num_epochs,
    class_weights_tensor=None,
    fold_idx=0,
    device=None,
    save_path='checkpoints/best_model_fold_{fold_idx}_epoch_{epoch}.pth',
    save_dir="outputs/transformed_tensors"
):
    """
    Train Faster R-CNN model with parallel STNs.
    
    Parameters
    ----------
    model : FasterRCNNWithParallelSTN
        Model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    optimizer : torch.optim.Optimizer
        Optimizer
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler
    num_epochs : int
        Number of training epochs
    class_weights_tensor : torch.Tensor, optional
        Class weights for imbalanced datasets
    fold_idx : int
        Fold index for k-fold cross-validation
    device : torch.device
        Device to train on
    save_path : str
        Path template for saving checkpoints
    save_dir : str
        Directory to save transformed tensors
        
    Returns
    -------
    tuple
        (model, best_val_loss, best_epoch, total_time, state_dict)
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f'outputs/tensorboard/fold_{fold_idx}')
    
    # Apply custom loss if class weights provided
    if class_weights_tensor is not None:
        model.faster_rcnn.roi_heads.fastrcnn_loss = partial(
            custom_fastrcnn_loss,
            class_weights_tensor=class_weights_tensor,
            device=device
        )
    
    # Early stopping variables
    best_fold_val_loss = float('inf')
    epochs_without_improvement = 0
    patience = 10
    best_epoch = 0
    total_training_time = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_loss_classifier = 0.0
        epoch_loss_box_reg = 0.0
        epoch_loss_objectness = 0.0
        epoch_loss_rpn_box_reg = 0.0
        
        start_time = time.time()
        
        # Progress bar
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            dynamic_ncols=True
        )
        
        for i, (images, targets) in pbar:
            # Filter invalid images
            images, targets = filter_invalid_images_and_boxes(images, targets)
            if not images:
                continue
            
            # Normalize images to [0, 1]
            images = normalize_images(images)
            
            # Move to device
            images = [image.to(device) for image in images]
            targets = [
                {
                    'boxes': t['boxes'].to(device),
                    'labels': t['labels'].to(device)
                }
                for t in targets
            ]
            
            optimizer.zero_grad()
            
            try:
                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                
                # Backward pass
                losses.backward()
                optimizer.step()
                
                # Accumulate losses
                epoch_loss += loss_value
                epoch_loss_classifier += loss_dict['loss_classifier'].item()
                epoch_loss_box_reg += loss_dict['loss_box_reg'].item()
                epoch_loss_objectness += loss_dict['loss_objectness'].item()
                epoch_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
                
                # Update progress bar
                pbar.set_postfix({'Total Loss': f'{loss_value:.4f}'})
                
                # Log to TensorBoard
                global_step = epoch * len(train_loader) + i
                writer.add_scalar('Loss/Total', loss_value, global_step)
                
            except Exception as e:
                print(f"Error during training: {e}")
                traceback.print_exc()
                continue
        
        # Scheduler step
        scheduler.step()
        
        # Calculate average losses
        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches
        avg_loss_classifier = epoch_loss_classifier / num_batches
        avg_loss_box_reg = epoch_loss_box_reg / num_batches
        avg_loss_objectness = epoch_loss_objectness / num_batches
        avg_loss_rpn_box_reg = epoch_loss_rpn_box_reg / num_batches
        
        # Log to TensorBoard
        writer.add_scalar('Loss_epoch/Total', avg_loss, epoch)
        writer.add_scalar('Loss_epoch/Classifier', avg_loss_classifier, epoch)
        writer.add_scalar('Loss_epoch/BoxReg', avg_loss_box_reg, epoch)
        writer.add_scalar('Loss_epoch/Objectness', avg_loss_objectness, epoch)
        writer.add_scalar('Loss_epoch/RPNBoxReg', avg_loss_rpn_box_reg, epoch)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        total_training_time += elapsed_time
        
        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Classifier Loss: {avg_loss_classifier:.4f}")
        print(f"  Box Reg Loss: {avg_loss_box_reg:.4f}")
        print(f"  Objectness Loss: {avg_loss_objectness:.4f}")
        print(f"  RPN Box Reg Loss: {avg_loss_rpn_box_reg:.4f}")
        print(f"  Time: {elapsed_time:.2f}s")
        
        # Validation step
        model.train()  # Keep in train mode to get losses
        val_loss = 0.0
        
        with torch.no_grad():
            for i, (images, targets) in enumerate(val_loader):
                images, targets = filter_invalid_images_and_boxes(images, targets)
                if not images:
                    continue
                
                images = [image.to(device) for image in images]
                targets = [
                    {
                        'boxes': t['boxes'].to(device),
                        'labels': t['labels'].to(device)
                    }
                    for t in targets
                ]
                
                try:
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()
                except Exception as e:
                    print(f"Error during validation: {e}")
                    continue
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_fold_val_loss:
            print(f"  Validation loss improved from {best_fold_val_loss:.4f} to {avg_val_loss:.4f}. Saving model.")
            best_fold_val_loss = avg_val_loss
            best_epoch = epoch + 1
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(
                model.state_dict(),
                save_path.format(fold_idx=fold_idx, epoch=epoch+1)
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break
    
    writer.close()
    
    return model, best_fold_val_loss, best_epoch, total_training_time, model.state_dict()

def main():
    """Main training function."""
    
    # Setup logger
    logger = get_experiment_logger('stn_training', 'outputs/logs')
    logger.info("Starting training with STN")
    
    # Load configurations
    paths_config = load_config('configs/paths.yaml')
    model_config = load_config('configs/model_config.yaml')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    # Load ALL training data (preprocessed + augmented merged together)
    logger.info("\n📂 Loading training data (preprocessed + augmented)...")
    df_train_images, df_train_annotations = load_all_training_data(paths_config, logger)
    
    if len(df_train_images) == 0:
        logger.error("❌ No training data found! Please run preprocessing and/or augmentation first.")
        return
    
    # Load test data for validation
    logger.info("\n📂 Loading test data for validation...")
    df_test_images, df_test_annotations = load_test_data(paths_config, logger)
    
    if len(df_test_images) == 0:
        logger.warning("⚠️  No test data! Will use 20% of training data for validation.")
        # Will split later
    
    # Get number of classes automatically from annotations
    logger.info("\n📊 Analyzing dataset...")
    num_classes, class_to_idx, idx_to_class = get_num_classes_from_annotations(
        df_train_annotations, logger
    )
        
    # Calculate class weights
    class_weights = calculate_class_weights(df_train_annotations, class_to_idx, logger)
    # Save class mapping
    os.makedirs('outputs/models', exist_ok=True)
    with open('outputs/models/class_mapping.json', 'w') as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class,
            'num_classes': num_classes
        }, f, indent=2)
    
    logger.info(f"✅ Class mapping saved to outputs/models/class_mapping.json")
    
    # Update annotations with numeric labels
    df_train_annotations['label_idx'] = df_train_annotations['class'].map(class_to_idx)
    if len(df_test_annotations) > 0:
        df_test_annotations['label_idx'] = df_test_annotations['class'].map(class_to_idx)
    
    # Create datasets
    logger.info("\n🔨 Creating datasets...")
    train_dataset = BaggageXRayDataset(df_train_images, df_train_annotations)
    
    if len(df_test_images) > 0:
        # Use test data for validation
        val_dataset = BaggageXRayDataset(df_test_images, df_test_annotations)
        logger.info(f"✅ Using separate test set for validation")
    else:
        # Split training data
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        logger.info(f"✅ Split training data: 80% train, 20% validation")
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=model_config['training']['batch_size'],
        shuffle=True,
        num_workers=model_config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=model_config['training']['batch_size'],
        shuffle=False,
        num_workers=model_config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    model_config['model']['num_classes'] = num_classes
    
    # Create model with STN
    logger.info("Creating model with STN...")
    model = create_model(
        num_classes=model_config['model']['num_classes'],
        stn_type='affine',  # Options: 'affine', 'projective', 'tps'
        device=device
    )
    logger.info(f"Model created with {model.stn_type} STN")
    
    # Setup optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=model_config['training']['learning_rate'],
        momentum=model_config['training']['momentum'],
        weight_decay=model_config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=model_config['training']['lr_scheduler']['step_size'],
        gamma=model_config['training']['lr_scheduler']['gamma']
    )
    
    # Train model
    logger.info("Starting training...")
    model, best_val_loss, best_epoch, total_time, final_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=model_config['training']['num_epochs'],
        class_weights_tensor=class_weights,
        device=device
    )
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Total training time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
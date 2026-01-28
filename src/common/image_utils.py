"""
Common image operations used across the project.
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Union


def torch_to_np(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Image tensor in (C, H, W) format
        
    Returns
    -------
    np.ndarray
        Image array in (H, W, C) format
    """
    return tensor.permute(1, 2, 0).numpy()


def np_to_torch(array: np.ndarray) -> torch.Tensor:
    """
    Convert NumPy array to PyTorch tensor.
    
    Parameters
    ----------
    array : np.ndarray
        Image array in (H, W, C) format
        
    Returns
    -------
    torch.Tensor
        Image tensor in (C, H, W) format
    """
    return torch.from_numpy(array).permute(2, 0, 1)


def save_tensor_as_image(
    tensor: torch.Tensor,
    filepath: str,
    quality: int = 95
) -> Tuple[Tuple[int, int, int], str]:
    """
    Save a tensor as an image file (avoids memory consumption).
    
    Parameters
    ----------
    tensor : torch.Tensor
        Image tensor to save
    filepath : str
        Path where to save the image
    quality : int
        JPEG quality (1-100)
        
    Returns
    -------
    tuple
        ((width, height, channels), mode) - Image dimensions and mode
    """
    # Convert to numpy if needed
    if isinstance(tensor, torch.Tensor):
        img_np = torch_to_np(tensor)
    else:
        img_np = tensor
    
    # Convert to uint8 if needed
    if img_np.dtype != np.uint8:
        # Normalize to 0-255 range
        img_np = np.clip(img_np, 0, 255)
        img_np = img_np.astype(np.uint8)
    
    # Create PIL Image
    img = Image.fromarray(img_np)
    
    # Save with quality parameter
    if filepath.lower().endswith('.jpg') or filepath.lower().endswith('.jpeg'):
        img.save(filepath, quality=quality, optimize=True)
    else:
        img.save(filepath)
    
    # Return shape and mode
    shape = (img.size[0], img.size[1], len(img.getbands()))
    return shape, img.mode


def load_image_as_tensor(filepath: str) -> torch.Tensor:
    """
    Load an image file as a PyTorch tensor.
    
    Parameters
    ----------
    filepath : str
        Path to the image file
        
    Returns
    -------
    torch.Tensor
        Image tensor in (C, H, W) format
    """
    from torchvision.io import read_image
    return read_image(filepath)


def show_image(
    image: Union[torch.Tensor, np.ndarray],
    title: str = "",
    figsize: Tuple[int, int] = (10, 10)
) -> None:
    """
    Display an image using matplotlib.
    
    Parameters
    ----------
    image : torch.Tensor or np.ndarray
        Image to display
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    
    # Convert to numpy if needed
    if isinstance(image, torch.Tensor):
        img_np = torch_to_np(image)
    else:
        img_np = image
    
    # Display info
    print(f"dtype: {img_np.dtype}")
    print(f"Shape: {img_np.shape}")
    if len(img_np.shape) >= 2:
        print(f"Aspect Ratio (h/w): {round(img_np.shape[0]/img_np.shape[1], 2)}")
    
    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(img_np)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def resize_image(
    image: Union[torch.Tensor, np.ndarray],
    size: Tuple[int, int],
    interpolation: str = 'bilinear'
) -> Union[torch.Tensor, np.ndarray]:
    """
    Resize an image to the specified size.
    
    Parameters
    ----------
    image : torch.Tensor or np.ndarray
        Image to resize
    size : tuple
        Target size (height, width)
    interpolation : str
        Interpolation method ('bilinear', 'nearest', 'bicubic')
        
    Returns
    -------
    torch.Tensor or np.ndarray
        Resized image (same type as input)
    """
    if isinstance(image, torch.Tensor):
        import torch.nn.functional as F
        return F.interpolate(
            image.unsqueeze(0),
            size=size,
            mode=interpolation,
            align_corners=False if interpolation != 'nearest' else None
        ).squeeze(0)
    else:
        # Use OpenCV for numpy arrays
        interpolation_map = {
            'bilinear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST,
            'bicubic': cv2.INTER_CUBIC
        }
        return cv2.resize(
            image,
            (size[1], size[0]),  # OpenCV expects (width, height)
            interpolation=interpolation_map.get(interpolation, cv2.INTER_LINEAR)
        )


def get_whiteness(
    image: Union[torch.Tensor, np.ndarray],
    whiteness_tolerance: int = 30,
    mean: Tuple[int, int, int] = (104, 117, 123)
) -> float:
    """
    Calculate the percentage of white pixels in an image.
    
    Parameters
    ----------
    image : torch.Tensor or np.ndarray
        Input image
    whiteness_tolerance : int
        Tolerance for considering a pixel as white
    mean : tuple
        Mean values to add back to normalized images
        
    Returns
    -------
    float
        Percentage of white pixels (0-100)
    """
    mn_np = np.array(mean)
    
    # Convert to numpy if needed
    if isinstance(image, torch.Tensor):
        img = image.numpy()
    else:
        img = image
    
    # Add mean back
    img = img + mn_np
    
    # Reshape to (N, 3) where N is number of pixels
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    
    # Count white pixels
    white_count = 0
    for pixel in img:
        if (255 - sum(pixel) / 3) <= whiteness_tolerance:
            white_count += 1
    
    white_percentage = 100 * white_count / len(img)
    return round(white_percentage, 2)

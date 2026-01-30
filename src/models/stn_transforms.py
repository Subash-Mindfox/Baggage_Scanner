"""
Spatial Transformer Networks (STN) for dynamic geometric transformations.

This module implements three types of spatial transformations:
1. Affine - Standard geometric transforms (rotation, scaling, translation)
2. Projective - Perspective transformations
3. TPS (Thin Plate Spline) - Non-rigid deformations

STNs learn to automatically apply the best geometric transformations to improve detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools


def compute_projective_grid(transform_params, input_image):
    """
    Compute grid coordinates for projective (perspective) transformation.
    
    Projective transformations can change the symmetry and perspective of images,
    useful for simulating different camera angles in X-ray images.
    
    Parameters
    ----------
    transform_params : torch.Tensor
        Transformation parameters of shape (batch_size, 3, 3)
    input_image : torch.Tensor
        Input images of shape (batch_size, C, H, W)
        
    Returns
    -------
    torch.Tensor
        Transformed grid coordinates of shape (batch_size, H, W, 2)
    
    Step-by-Step:
    1. Create a mesh grid of coordinates covering the entire image
    2. Convert to homogeneous coordinates (add a 1 as the third coordinate)
    3. Apply the 3x3 projective transformation matrix
    4. Divide by the third coordinate to get back to 2D (perspective divide)
    """
    batch_size, _, _ = transform_params.size()
    height, width = input_image.size(2), input_image.size(3)
    
    # Step 1: Create a grid of (x, y) coordinates from -1 to 1
    # This creates a mesh covering the entire image
    grid_coords = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, height),
        torch.linspace(-1, 1, width),
        indexing='ij'
    ), dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1).to(input_image.device)
    
    # Step 2: Reshape to (batch_size, num_pixels, 2)
    grid_coords = grid_coords.view(batch_size, -1, 2)
    
    # Step 3: Add homogeneous coordinate (add 1 as third column)
    # Shape becomes (batch_size, num_pixels, 3)
    grid_coords = torch.cat(
        [grid_coords, torch.ones_like(grid_coords[:, :, :1])],
        dim=-1
    )
    
    # Step 4: Apply transformation matrix
    # Multiply each pixel coordinate by the 3x3 transformation matrix
    grid_transformed = torch.bmm(grid_coords, transform_params.transpose(1, 2))
    
    # Step 5: Perspective divide
    # Divide x and y by the third coordinate (z) to get final 2D coordinates
    grid_transformed = grid_transformed[:, :, :2] / grid_transformed[:, :, 2:].clamp(min=1e-6)
    
    # Step 6: Reshape back to image dimensions
    grid_transformed = grid_transformed.view(batch_size, height, width, 2)
    
    return grid_transformed


def limit_affine_params(transform_params, scale_limit=(0.8, 1.1), translate_limit=(-0.2, 0.2)):
    """
    Limit affine transformation parameters to prevent extreme distortions.
    
    This ensures transformations are reasonable and won't completely
    distort the X-ray images beyond recognition.
    
    Parameters
    ----------
    transform_params : torch.Tensor
        Affine transformation matrix of shape (batch_size, 2, 3)
        Format: [[scale_x, shear, translate_x],
                 [shear, scale_y, translate_y]]
    scale_limit : tuple
        (min, max) allowed scaling factors
    translate_limit : tuple
        (min, max) allowed translation values
        
    Returns
    -------
    torch.Tensor
        Clamped transformation parameters
    
    Explanation:
    - transform_params[:, 0, 0] = scale in x direction
    - transform_params[:, 1, 1] = scale in y direction  
    - transform_params[:, 0, 2] = translation in x direction
    - transform_params[:, 1, 2] = translation in y direction
    """
    theta_clamped = transform_params.clone()
    
    # Limit scaling in x direction (0.8 to 1.1 means 80% to 110% of original size)
    theta_clamped[:, 0, 0] = transform_params[:, 0, 0].clamp(*scale_limit)
    
    # Limit scaling in y direction
    theta_clamped[:, 1, 1] = transform_params[:, 1, 1].clamp(*scale_limit)
    
    # Limit translation in x direction (-0.2 to 0.2 means max 20% shift)
    theta_clamped[:, 0, 2] = transform_params[:, 0, 2].clamp(*translate_limit)
    
    # Limit translation in y direction
    theta_clamped[:, 1, 2] = transform_params[:, 1, 2].clamp(*translate_limit)
    
    return theta_clamped


def radial_basis_fn(input_pts, base_pts):
    """
    Compute Thin Plate Spline (TPS) radial basis function.
    
    TPS is used for smooth, non-rigid deformations. Think of it like
    placing control points on a thin metal plate and bending the plate.
    
    Parameters
    ----------
    input_pts : torch.Tensor
        Input points of shape (num_inputs, 2)
    base_pts : torch.Tensor
        Base control points of shape (num_base, 2)
        
    Returns
    -------
    torch.Tensor
        Radial basis function values of shape (num_inputs, num_base)
    
    Mathematical Formula:
    - For each pair of points, compute distance squared: d²
    - Apply TPS kernel: 0.5 * d² * log(d²)
    - This creates smooth interpolation between control points
    """
    num_inputs = input_pts.size(0)
    num_base = base_pts.size(0)
    
    # Step 1: Compute differences between all input and base points
    # Shape: (num_inputs, num_base, 2)
    diffs = input_pts.view(num_inputs, 1, 2) - base_pts.view(1, num_base, 2)
    
    # Step 2: Compute squared distances
    # Sum across the last dimension (x and y)
    dist_squared = (diffs ** 2).sum(dim=2)
    
    # Step 3: Apply TPS kernel function
    # Formula: 0.5 * r² * log(r²)
    # Add small epsilon (1e-6) to avoid log(0)
    rbf_output = 0.5 * dist_squared * torch.log(dist_squared + 1e-6)
    
    # Step 4: Replace any NaN values with 0
    return rbf_output.masked_fill(rbf_output != rbf_output, 0)


class FlexibleTPSGrid(nn.Module):
    """
    Thin Plate Spline (TPS) grid generator for non-rigid transformations.
    
    TPS allows smooth, non-rigid deformations by placing control points
    on the image and learning how to move them. This is more flexible
    than affine or projective transforms.
    
    Think of it like: You place pushpins (control points) on a rubber sheet
    (the image), and the network learns where to move each pushpin to 
    improve detection.
    """
    
    def __init__(self, output_height, output_width, control_points):
        """
        Initialize TPS grid generator.
        
        Parameters
        ----------
        output_height : int
            Height of output image
        output_width : int
            Width of output image
        control_points : torch.Tensor
            Initial control point positions of shape (num_points, 2)
        """
        super(FlexibleTPSGrid, self).__init__()
        
        num_points = control_points.size(0)
        
        # Step 1: Create kernel matrix (the mathematical foundation of TPS)
        # This matrix size is (num_points + 3) x (num_points + 3)
        # The +3 accounts for affine part of the transformation
        kernel_matrix = torch.zeros(num_points + 3, num_points + 3)
        
        # Step 2: Fill in the radial basis function part
        # This captures the non-rigid deformation between control points
        rbf_representation = radial_basis_fn(control_points, control_points)
        kernel_matrix[:num_points, :num_points].copy_(rbf_representation)
        
        # Step 3: Fill in the affine part (for basic translation/rotation)
        kernel_matrix[:num_points, -3].fill_(1)  # Ones column
        kernel_matrix[-3, :num_points].fill_(1)  # Ones row
        kernel_matrix[:num_points, -2:].copy_(control_points)  # x, y coordinates
        kernel_matrix[-2:, :num_points].copy_(control_points.transpose(0, 1))
        
        # Step 4: Invert the kernel matrix
        # This will be used to solve for transformation weights
        self.inv_kernel_matrix = torch.inverse(kernel_matrix)
        
        # Step 5: Create target grid (all pixel locations in output image)
        num_pixels = output_height * output_width
        
        # Create grid coordinates for every pixel
        grid_coords = torch.Tensor(
            list(itertools.product(range(output_height), range(output_width)))
        )
        
        # Normalize coordinates to [-1, 1] range
        y_coords, x_coords = grid_coords.split(1, dim=1)
        y_coords = y_coords * 2 / (output_height - 1) - 1
        x_coords = x_coords * 2 / (output_width - 1) - 1
        final_grid_coords = torch.cat([x_coords, y_coords], dim=1)
        
        # Step 6: Compute radial basis for all pixels relative to control points
        rbf_target = radial_basis_fn(final_grid_coords, control_points)
        
        # Step 7: Create final representation matrix
        # This combines RBF, affine, and coordinate information
        self.final_representation = torch.cat(
            [rbf_target, torch.ones(num_pixels, 1), final_grid_coords],
            dim=1
        )
        
        # Step 8: Create padding matrix for batch processing
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
    
    def forward(self, src_control_points):
        """
        Generate transformed grid based on source control points.
        
        Parameters
        ----------
        src_control_points : torch.Tensor
            Source control point positions of shape (batch_size, num_points, 2)
            These are the NEW positions the network wants to move the control points to
            
        Returns
        -------
        torch.Tensor
            Transformed grid of shape (batch_size, height*width, 2)
        """
        batch_size = src_control_points.size(0)
        
        # Step 1: Add padding for batch processing
        padding_matrix_expanded = self.padding_matrix.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        extended_control_pts = torch.cat(
            [src_control_points, padding_matrix_expanded],
            dim=1
        )
        
        # Step 2: Expand inverse kernel matrix for batch
        batch_inv_kernel_matrix = self.inv_kernel_matrix.expand(batch_size, -1, -1)
        
        # Step 3: Compute transformation matrix
        # This solves for the weights needed to achieve the desired control point movement
        transformation_matrix = torch.matmul(
            batch_inv_kernel_matrix,
            extended_control_pts
        )
        
        # Step 4: Apply transformation to get final grid
        # This computes where each pixel should sample from in the original image
        target_grid = torch.matmul(self.final_representation, transformation_matrix)
        
        return target_grid


class DynamicSTN(nn.Module):
    """
    Dynamic Spatial Transformer Network.
    
    This network learns to apply geometric transformations to images
    to improve object detection. It can use three types of transformations:
    
    1. Affine - Rotation, scaling, translation, shearing (most common)
    2. Projective - Perspective transformations (like changing camera angle)
    3. TPS - Non-rigid deformations (bending and warping)
    
    How it works:
    1. Localization Network: Looks at the image and predicts transformation parameters
    2. Grid Generator: Creates a sampling grid based on those parameters
    3. Sampler: Uses the grid to sample from the original image
    
    Think of it as: The network learns "how should I rotate/zoom/warp this
    X-ray image to make the objects easier to detect?"
    """
    
    def __init__(
        self,
        transformation_type='affine',
        control_pts=10,
        out_height=584,
        out_width=688
    ):
        """
        Initialize Dynamic STN.
        
        Parameters
        ----------
        transformation_type : str
            Type of transformation: 'affine', 'projective', or 'tps'
        control_pts : int
            Number of control points for TPS (must be perfect square)
        out_height : int
            Output image height
        out_width : int
            Output image width
        """
        super(DynamicSTN, self).__init__()
        
        self.transformation_type = transformation_type
        self.control_pts = control_pts
        self.out_height = out_height
        self.out_width = out_width
        
        # LOCALIZATION NETWORK
        # This CNN looks at the image and predicts transformation parameters
        # It uses convolutions to extract features, then predicts transform params
        self.localization = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=0),
            nn.MaxPool2d(2, stride=2),  # Reduce size for efficiency
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5, stride=1),
            nn.MaxPool2d(2, stride=2),  # Further reduce size
            nn.ReLU(True)
        )
        
        # TRANSFORMATION PREDICTOR
        # Different FC layers for different transformation types
        
        if self.transformation_type == 'affine':
            # Affine needs 6 parameters for 2x3 matrix:
            # [[scale_x, shear_x, translate_x],
            #  [shear_y, scale_y, translate_y]]
            self.fc_transform = nn.Sequential(
                nn.Linear(32 * 142 * 168, 128),  # Flatten features
                nn.ReLU(True),
                nn.Linear(128, 6)  # Output 6 affine parameters
            )
            
            # Initialize to identity transform (no change)
            self.fc_transform[2].weight.data.zero_()
            self.fc_transform[2].bias.data.copy_(torch.tensor(
                [1, 0, 0, 0, 1, 0],  # Identity: scale=1, no rotation, no translation
                dtype=torch.float
            ))
        
        elif self.transformation_type == 'projective':
            # Projective needs 9 parameters for 3x3 matrix
            self.fc_transform = nn.Sequential(
                nn.Linear(32 * 142 * 168, 128),
                nn.ReLU(True),
                nn.Linear(128, 9)  # Output 9 projective parameters
            )
            
            # Initialize to identity
            self.fc_transform[2].weight.data.zero_()
            self.fc_transform[2].bias.data.copy_(torch.tensor(
                [1, 0, 0, 0, 1, 0, 0, 0, 1],  # 3x3 identity matrix
                dtype=torch.float
            ))
        
        elif self.transformation_type == 'tps':
            # TPS needs 2 coordinates (x, y) for each control point
            control_pts = self.adjust_control_pts(control_pts)
            
            self.fc_transform = nn.Sequential(
                nn.Linear(32 * 142 * 168, 128),
                nn.ReLU(True),
                nn.Linear(128, control_pts * 2)  # x, y for each control point
            )
            
            # Initialize to grid positions (no deformation)
            self.fc_transform[2].weight.data.zero_()
            self.fc_transform[2].bias.data.copy_(
                self._initialize_control_pts(control_pts).view(-1)
            )
            
            # Create TPS grid generator
            self.tps_grid_gen = FlexibleTPSGrid(
                self.out_height,
                self.out_width,
                self._initialize_control_pts(control_pts)
            )
    
    def adjust_control_pts(self, control_pts):
        """
        Ensure control points is a perfect square (needed for grid arrangement).
        
        For example:
        - 10 → 9 (3x3 grid)
        - 15 → 16 (4x4 grid)
        - 25 → 25 (5x5 grid - already perfect)
        """
        grid_size = int(math.sqrt(control_pts))
        if grid_size ** 2 != control_pts:
            control_pts = grid_size ** 2
            print(f"Adjusted control_pts to {control_pts} (nearest perfect square).")
        return control_pts
    
    def _initialize_control_pts(self, control_pts):
        """
        Create initial control points in a regular grid pattern.
        
        Returns
        -------
        torch.Tensor
            Control points of shape (num_points, 2) arranged in a grid
        """
        grid_size = int(math.sqrt(control_pts))
        
        # Create grid coordinates from -1 to 1
        control_grid = torch.meshgrid(
            torch.linspace(-1, 1, grid_size),
            torch.linspace(-1, 1, grid_size),
            indexing='ij'
        )
        
        # Flatten and combine x, y coordinates
        control_x = control_grid[0].reshape(-1, 1)
        control_y = control_grid[1].reshape(-1, 1)
        
        return torch.cat([control_x, control_y], dim=1)
    
    def forward(self, input_image, targets=None):
        """
        Apply spatial transformation to input images.
        
        Parameters
        ----------
        input_image : torch.Tensor
            Input images of shape (batch_size, 3, H, W)
        targets : list of dict, optional
            Not used in STN, but kept for compatibility
            
        Returns
        -------
        torch.Tensor
            Transformed images of same shape as input
        
        Process:
        1. Extract features using localization network
        2. Predict transformation parameters
        3. Generate sampling grid
        4. Sample from original image using the grid
        """
        # Step 1: Extract features
        features = self.localization(input_image)
        features = features.view(-1, 32 * 142 * 168)
        
        # Step 2: Predict transformation parameters
        transform_params = self.fc_transform(features)
        
        # Step 3 & 4: Generate grid and sample based on transformation type
        if self.transformation_type == 'affine':
            # Reshape to 2x3 matrix
            transform_params = transform_params.view(-1, 2, 3)
            
            # Limit parameters to prevent extreme distortions
            transform_params = limit_affine_params(transform_params)
            
            # Generate affine grid
            grid = F.affine_grid(
                transform_params,
                input_image.size(),
                align_corners=False
            )
        
        elif self.transformation_type == 'projective':
            # Reshape to 3x3 matrix
            transform_params = transform_params.view(-1, 3, 3)
            
            # Generate projective grid
            grid = compute_projective_grid(transform_params, input_image)
        
        elif self.transformation_type == 'tps':
            batch_size = input_image.size(0)
            
            # Reshape to control points (num_points, 2)
            num_control_pts = int(math.sqrt(self.control_pts)) ** 2
            control_points = transform_params.view(batch_size, num_control_pts, 2)
            
            # Generate TPS grid
            grid = self.tps_grid_gen(control_points)
            
            # Reshape grid to match image dimensions
            height, width = input_image.shape[2], input_image.shape[3]
            grid = grid.view(batch_size, height, width, 2)
        
        # Step 5: Sample from original image using the grid
        output_image = F.grid_sample(input_image, grid, align_corners=False)
        
        return output_image

# AI Baggage Scanner Detection System

A modular, production-ready object detection system for X-ray baggage scanner images.

## 🎯 Features

- **Configurable Preprocessing**: Toggle whitespace removal on/off via YAML config
- **Modular Augmentation System**: Add new augmentations without touching existing code
- **Plugin Architecture**: Registry-based system for easy extensibility
- **Memory Efficient**: Images written to disk, not stored in memory
- **Multiple Model Support**: Easy to swap between Faster R-CNN, YOLO, etc.
- **Comprehensive Logging**: Track experiments with detailed logs
- **Clean Project Structure**: Separation of concerns, easy to maintain

## 📁 Project Structure

```
baggage_scanner_project/
├── configs/              # YAML configuration files
├── data/                 # Data storage
│   ├── raw/             # Original, untouched data
│   └── temp/            # Processed & augmented data
├── src/                 # Source code
│   ├── data/            # Dataset and data utilities
│   ├── preprocessing/   # Preprocessing modules
│   ├── augmentation/    # Augmentation modules
│   ├── models/          # Model implementations
│   ├── training/        # Training utilities
│   ├── utils/           # Common utilities
│   └── common/          # Shared functions
├── scripts/             # Executable scripts
├── notebooks/           # Jupyter notebooks
├── tests/               # Unit tests
├── checkpoints/         # Saved models
└── outputs/             # Logs, metrics, visualizations
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd baggage_scanner_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Prepare Your Data

1. Place your raw data in the `data/raw/` directory:
   ```
   data/raw/
   ├── Train/
   │   ├── Images/
   │   └── Annotations/
   └── Test/
       ├── Images/
       └── Annotations/
   ```

2. Update `configs/paths.yaml` if your paths differ

### Run the Pipeline

```bash
# 1. Collect and analyze data
python scripts/01_collect_data.py

# 2. Preprocess data (whitespace removal)
python scripts/02_preprocess.py

# 3. Apply augmentations
python scripts/03_augment.py

# 4. Train model
python scripts/04_train.py

# 5. Evaluate model
python scripts/05_evaluate.py
```

## ⚙️ Configuration

All settings are managed through YAML files in `configs/`:

### Enable/Disable Whitespace Removal

Edit `configs/preprocessing.yaml`:
```yaml
preprocessing:
  whitespace_removal:
    enabled: true  # Set to false to disable
```

### Configure Augmentations

Edit `configs/augmentation.yaml`:
```yaml
augmentations:
  color_jitter:
    enabled: true
    num_samples: 1000
    params:
      brightness: [0.8, 2.0]
      contrast: [0.8, 2.0]
  
  flip:
    enabled: false  # Disable this augmentation
```

### Run Specific Augmentations

```python
from src.augmentation.pipeline import AugmentationPipeline

# Load config and disable unwanted augmentations
pipeline = AugmentationPipeline.from_config_file('configs/augmentation.yaml')
pipeline.disable_augmentation('color_jitter')
pipeline.enable_augmentation('flip')

# Run pipeline
df_images, df_annotations = pipeline.run(df_images, df_annotations, output_dir)
```

## 🔌 Adding New Augmentations

1. Create a new file in `src/augmentation/`:

```python
# src/augmentation/my_augmentation.py
from src.augmentation.base_augmentor import BaseAugmentor
from src.augmentation.registry import AugmentationRegistry

@AugmentationRegistry.register('my_augmentation')
class MyAugmentor(BaseAugmentor):
    def _get_suffix(self):
        return '_my.jpg'
    
    def apply_transform(self, image_tensor, **kwargs):
        # Your transformation logic
        return transformed_tensor
    
    def adjust_annotations(self, annotations, img_width, img_height, **kwargs):
        # Adjust bounding boxes if needed
        return adjusted_annotations
    
    def _perform_augmentation(self, df_images, df_annotations, output_dir):
        # Implementation
        pass
```

2. Add to `configs/augmentation.yaml`:

```yaml
my_augmentation:
  enabled: true
  num_samples: 500
  params:
    your_param: value
```

3. Import in your script:

```python
from src.augmentation import my_augmentation
```

## 🎓 Model Training

### Using Faster R-CNN

```python
from src.models.faster_rcnn import FasterRCNNModel

model = FasterRCNNModel(model_config)
model.to_device()

# Train
for epoch in range(num_epochs):
    train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
    model.save_checkpoint(f'checkpoints/epoch_{epoch}.pth', epoch, optimizer, train_metrics)
```

### Switching Models

Just change the `architecture` in `configs/model_config.yaml`:

```yaml
model:
  architecture: "yolo"  # or "faster_rcnn", "ssd", "custom"
```

## 📊 Data Flow

```
Raw Data → Preprocessing → Augmentation → Training → Evaluation
    ↓           ↓              ↓            ↓          ↓
Original    Temp/         Temp/        Checkpoints  Metrics
Data       Preprocessed  Augmented
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_augmentation.py
```

## 📝 Logging

Logs are saved in `outputs/logs/`:
- `augmentation_<timestamp>.log` - Augmentation pipeline logs
- `training_<timestamp>.log` - Training logs

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Add tests
4. Submit a pull request

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- Based on research in X-ray baggage security
- Uses PyTorch and torchvision for deep learning
- Inspired by modular ML project best practices

## 📧 Contact

[Your Contact Information]

---

**Note**: This is a template project. You'll need to:
1. Implement the Dataset class in `src/data/dataset.py`
2. Complete the training loop in `scripts/04_train.py`
3. Add evaluation metrics in `scripts/05_evaluate.py`
4. Customize for your specific use case

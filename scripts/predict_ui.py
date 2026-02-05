"""
Object Detection Prediction UI - Model Architecture Fix

Fixed version that properly handles model architecture and class count.
Automatically detects the correct number of classes from the checkpoint.

Features:
- Automatic class count detection
- Handles model architecture mismatches
- Load single images or folders
- Adjust confidence threshold
- View predictions with bounding boxes
- Save annotated images
- Export results to JSON
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import json
import warnings
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision.io import read_image
from pathlib import Path
import traceback

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QScrollArea,
    QListWidget, QSplitter, QGroupBox, QTextEdit, QMessageBox,
    QProgressBar, QComboBox, QLineEdit, QCheckBox, QStatusBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QPalette

try:
    from src.models.faster_rcnn import FasterRCNNModel
    from src.utils.config_loader import load_config
except ImportError as e:
    print(f"Warning: Could not import project modules: {e}")
    print("Make sure you're running from the project root directory")


def detect_num_classes_from_checkpoint(checkpoint_path):
    """
    Detect the number of classes from a checkpoint file.
    
    This examines the checkpoint's model state dict to determine
    how many classes the model was trained with.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Look for the box predictor's classification layer
        # The key is usually 'roi_heads.box_predictor.cls_score.weight'
        for key in state_dict.keys():
            if 'cls_score.weight' in key:
                # Shape is [num_classes, feature_dim]
                num_classes = state_dict[key].shape[0]
                print(f"Detected {num_classes} classes from checkpoint (key: {key})")
                return num_classes
        
        print("Could not auto-detect number of classes from checkpoint")
        return None
        
    except Exception as e:
        print(f"Error detecting classes from checkpoint: {e}")
        return None


class PredictionWorker(QThread):
    """Background thread for running predictions."""
    
    progress = pyqtSignal(int, str)  # progress, message
    finished = pyqtSignal(dict)  # results
    error = pyqtSignal(str)  # error message
    
    def __init__(self, model, class_mapping, device, image_paths, conf_threshold):
        super().__init__()
        self.model = model
        self.class_mapping = class_mapping
        self.device = device
        self.image_paths = image_paths
        self.conf_threshold = conf_threshold
        self.is_running = True
    
    def run(self):
        """Run predictions on all images."""
        results = {}
        total = len(self.image_paths)
        
        try:
            for idx, image_path in enumerate(self.image_paths):
                if not self.is_running:
                    break
                
                # Update progress
                progress_pct = int((idx + 1) / total * 100)
                self.progress.emit(
                    progress_pct,
                    f"Processing {Path(image_path).name} ({idx + 1}/{total})..."
                )
                
                # Run prediction
                try:
                    detections = self.predict_image(image_path)
                    results[image_path] = detections
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    results[image_path] = []
            
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def predict_image(self, image_path):
        """Run prediction on a single image."""
        # Load and normalize image
        image = read_image(image_path)
        image = image.float() / 255.0
        
        # Run inference
        with torch.no_grad():
            predictions = self.model.predict([image.to(self.device)])
        
        # Parse predictions
        pred = predictions[0]
        detections = []
        
        for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
            if score > self.conf_threshold:
                class_name = self.class_mapping.get(label.item(), f"Class {label.item()}")
                detections.append({
                    'box': box.cpu().numpy().tolist(),
                    'label': class_name,
                    'score': score.item()
                })
        
        return detections
    
    def stop(self):
        """Stop the worker."""
        self.is_running = False


class ImageViewer(QLabel):
    """Custom widget for displaying images with bounding boxes."""
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QLabel { background-color: #2b2b2b; border: 2px solid #3d3d3d; border-radius: 4px; }")
        self.setMinimumSize(400, 400)
        self.original_pixmap = None
        self.detections = []
        self.show_boxes = True
        self.color_map = {
            'gun': QColor(255, 0, 0),
            'knife': QColor(255, 165, 0),
            'weapon': QColor(255, 255, 0),
            'pistol': QColor(255, 50, 50),
            'rifle': QColor(200, 0, 0),
        }
    
    def load_image(self, image_path):
        """Load and display an image."""
        try:
            self.original_pixmap = QPixmap(image_path)
            if self.original_pixmap.isNull():
                print(f"Failed to load image: {image_path}")
                return
            self.update_display()
        except Exception as e:
            print(f"Error loading image: {e}")
    
    def set_detections(self, detections):
        """Set detection results to display."""
        self.detections = detections
        self.update_display()
    
    def update_display(self):
        """Update the displayed image with bounding boxes."""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return
        
        # Create a copy to draw on
        pixmap = self.original_pixmap.copy()
        
        if self.show_boxes and self.detections:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Set font
            font = QFont("Arial", 14, QFont.Bold)
            painter.setFont(font)
            
            for det in self.detections:
                x1, y1, x2, y2 = det['box']
                label = det['label']
                score = det['score']
                
                # Get color
                color = self.color_map.get(label.lower(), QColor(0, 255, 0))
                
                # Draw rectangle
                pen = QPen(color, 4)
                painter.setPen(pen)
                painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                
                # Draw label background
                text = f"{label} {score:.2f}"
                text_rect = painter.boundingRect(int(x1), int(y1 - 30), 250, 30, Qt.AlignLeft, text)
                text_rect.adjust(-5, -2, 5, 2)
                painter.fillRect(text_rect, color)
                
                # Draw label text
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(text_rect, Qt.AlignCenter, text)
            
            painter.end()
        
        # Scale to fit widget while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)
    
    def toggle_boxes(self, show):
        """Toggle bounding box visibility."""
        self.show_boxes = show
        self.update_display()
    
    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        self.update_display()


class PredictionUI(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.class_mapping = None
        self.device = None
        self.current_image_path = None
        self.image_paths = []
        self.results = {}
        self.worker = None
        
        self.init_ui()
        
        # Auto-load model dialog after UI is initialized
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(500, self.load_model_dialog)
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("🔍 Object Detection Prediction Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel - Controls and file list
        left_panel = self.create_left_panel()
        
        # Right panel - Image viewer and results
        right_panel = self.create_right_panel()
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([380, 1020])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready to load model...")
        
        # Apply dark theme
        self.apply_dark_theme()
    
    def create_left_panel(self):
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        panel.setMaximumWidth(400)
        
        # Model info group
        model_group = QGroupBox("📦 Model Information")
        model_layout = QVBoxLayout()
        self.model_info_label = QLabel("No model loaded\nClick 'Load Model' to begin")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet("QLabel { padding: 10px; }")
        model_layout.addWidget(self.model_info_label)
        
        # Add load model button
        self.load_model_btn = QPushButton("🔄 Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_btn)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Load controls group
        load_group = QGroupBox("📁 Load Images")
        load_layout = QVBoxLayout()
        
        self.load_image_btn = QPushButton("📸 Load Single Image")
        self.load_image_btn.clicked.connect(self.load_single_image)
        self.load_image_btn.setEnabled(False)
        load_layout.addWidget(self.load_image_btn)
        
        self.load_folder_btn = QPushButton("📂 Load Folder")
        self.load_folder_btn.clicked.connect(self.load_folder)
        self.load_folder_btn.setEnabled(False)
        load_layout.addWidget(self.load_folder_btn)
        
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # Image list
        list_group = QGroupBox("🖼️ Image List")
        list_layout = QVBoxLayout()
        
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.on_image_selected)
        list_layout.addWidget(self.image_list)
        
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        # Prediction controls
        pred_group = QGroupBox("⚙️ Prediction Settings")
        pred_layout = QVBoxLayout()
        
        # Confidence threshold
        conf_label = QLabel("Confidence Threshold:")
        pred_layout.addWidget(conf_label)
        
        conf_layout = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.on_conf_changed)
        conf_layout.addWidget(self.conf_slider)
        
        self.conf_value_label = QLabel("0.50")
        self.conf_value_label.setStyleSheet("QLabel { font-weight: bold; min-width: 40px; }")
        conf_layout.addWidget(self.conf_value_label)
        pred_layout.addLayout(conf_layout)
        
        # Show boxes checkbox
        self.show_boxes_cb = QCheckBox("Show Bounding Boxes")
        self.show_boxes_cb.setChecked(True)
        self.show_boxes_cb.stateChanged.connect(self.on_show_boxes_changed)
        pred_layout.addWidget(self.show_boxes_cb)
        
        # Predict button
        self.predict_btn = QPushButton("🔍 Run Prediction")
        self.predict_btn.clicked.connect(self.run_prediction)
        self.predict_btn.setEnabled(False)
        self.predict_btn.setStyleSheet("QPushButton { padding: 10px; font-size: 14px; }")
        pred_layout.addWidget(self.predict_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        pred_layout.addWidget(self.progress_bar)
        
        pred_group.setLayout(pred_layout)
        layout.addWidget(pred_group)
        
        # Export controls
        export_group = QGroupBox("💾 Export Results")
        export_layout = QVBoxLayout()
        
        self.save_image_btn = QPushButton("💾 Save Current Image")
        self.save_image_btn.clicked.connect(self.save_current_image)
        self.save_image_btn.setEnabled(False)
        export_layout.addWidget(self.save_image_btn)
        
        self.save_all_btn = QPushButton("💾 Save All Images")
        self.save_all_btn.clicked.connect(self.save_all_images)
        self.save_all_btn.setEnabled(False)
        export_layout.addWidget(self.save_all_btn)
        
        self.export_json_btn = QPushButton("📄 Export to JSON")
        self.export_json_btn.clicked.connect(self.export_results)
        self.export_json_btn.setEnabled(False)
        export_layout.addWidget(self.export_json_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self):
        """Create the right display panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Image viewer
        viewer_group = QGroupBox("🖼️ Image Viewer")
        viewer_layout = QVBoxLayout()
        
        self.image_viewer = ImageViewer()
        viewer_layout.addWidget(self.image_viewer)
        
        viewer_group.setLayout(viewer_layout)
        layout.addWidget(viewer_group, stretch=3)
        
        # Results panel
        results_group = QGroupBox("📊 Detection Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        self.results_text.setPlaceholderText("Detection results will appear here...")
        results_layout.addWidget(self.results_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group, stretch=1)
        
        return panel
    
    def apply_dark_theme(self):
        """Apply dark theme to the application."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
            }
            QGroupBox {
                border: 2px solid #3d3d3d;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 15px;
                font-weight: bold;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0d5689;
            }
            QPushButton:disabled {
                background-color: #3d3d3d;
                color: #888888;
            }
            QListWidget {
                background-color: #2b2b2b;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-radius: 3px;
            }
            QListWidget::item:selected {
                background-color: #0e639c;
            }
            QListWidget::item:hover {
                background-color: #3d3d3d;
            }
            QTextEdit {
                background-color: #2b2b2b;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 5px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #3d3d3d;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0e639c;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #1177bb;
            }
            QProgressBar {
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                text-align: center;
                background-color: #2b2b2b;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
                border-radius: 3px;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #3d3d3d;
                border-radius: 3px;
                background-color: #2b2b2b;
            }
            QCheckBox::indicator:checked {
                background-color: #0e639c;
                image: url(none);
            }
            QCheckBox::indicator:hover {
                border: 2px solid #0e639c;
            }
            QStatusBar {
                background-color: #252525;
                color: #ffffff;
                border-top: 1px solid #3d3d3d;
            }
            QLabel {
                color: #ffffff;
            }
        """)
    
    def load_model_dialog(self):
        """Show dialog to load model on startup."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText("Load a trained model to begin")
        msg.setInformativeText("Please select your model checkpoint and class mapping files.")
        msg.setWindowTitle("Load Model")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        
        if msg.exec_() == QMessageBox.Ok:
            self.load_model()
    
    def load_model(self):
        """Load the trained model with automatic class detection."""
        # Get checkpoint path
        checkpoint_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model Checkpoint",
            "checkpoints",
            "PyTorch Model (*.pth *.pt);;All Files (*)"
        )
        
        if not checkpoint_path:
            return
        
        # Get class mapping path
        class_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Class Mapping",
            "outputs/models",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not class_path:
            return
        
        try:
            self.status_bar.showMessage("Loading model...")
            QApplication.processEvents()  # Update UI
            
            # Setup device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"\n{'='*60}")
            print(f"MODEL LOADING PROCESS")
            print(f"{'='*60}")
            print(f"Using device: {self.device}")
            
            # Load class mapping
            print(f"\nLoading class mapping from: {class_path}")
            with open(class_path, 'r') as f:
                class_data = json.load(f)
                self.class_mapping = {int(k): v for k, v in class_data['idx_to_class'].items()}
                num_classes_from_json = class_data['num_classes']
            print(f"Classes from JSON: {num_classes_from_json}")
            print(f"Class names: {list(self.class_mapping.values())}")
            
            # Auto-detect number of classes from checkpoint
            print(f"\nDetecting number of classes from checkpoint...")
            num_classes_from_checkpoint = detect_num_classes_from_checkpoint(checkpoint_path)
            
            # Use the checkpoint's class count (this is the correct one!)
            if num_classes_from_checkpoint:
                num_classes = num_classes_from_checkpoint
                print(f"✅ Using {num_classes} classes from checkpoint")
                
                if num_classes != num_classes_from_json:
                    print(f"⚠️  Warning: JSON says {num_classes_from_json} classes, but checkpoint has {num_classes}")
                    print(f"   Using checkpoint value ({num_classes}) to avoid size mismatch")
            else:
                num_classes = num_classes_from_json
                print(f"⚠️  Could not auto-detect from checkpoint, using JSON value: {num_classes}")
            
            # Load model config
            config_path = 'configs/model_config.yaml'
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            print(f"\nLoading config from: {config_path}")
            model_config = load_config(config_path)
            
            # Update config with CORRECT number of classes
            if 'model' not in model_config:
                model_config['model'] = {}
            model_config['model']['num_classes'] = num_classes
            print(f"Model config num_classes set to: {num_classes}")
            
            # Create model
            print(f"\nCreating model with {num_classes} classes...")
            self.model = FasterRCNNModel(model_config['model'])
            print(f"✅ Model created successfully")
            
            # Load checkpoint
            print(f"\nLoading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                print(f"\nLoading model weights...")
                try:
                    # Try strict loading
                    self.model.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                    print("✅ Model weights loaded successfully (strict=True)")
                except RuntimeError as e:
                    print(f"⚠️  Strict loading failed: {e}")
                    print("   Trying non-strict loading...")
                    # Try non-strict loading
                    missing, unexpected = self.model.model.load_state_dict(
                        checkpoint['model_state_dict'], 
                        strict=False
                    )
                    if missing:
                        print(f"   Missing keys: {missing[:5]}...")  # Show first 5
                    if unexpected:
                        print(f"   Unexpected keys: {unexpected[:5]}...")  # Show first 5
                    print("✅ Model weights loaded (strict=False)")
            else:
                print(f"\nLoading model weights...")
                try:
                    # Try strict loading
                    self.model.model.load_state_dict(checkpoint, strict=True)
                    print("✅ Model weights loaded  without model state successfully (strict=True)")
                except RuntimeError as e:
                    print(f"⚠️  Strict loading failed: {e}")
                    print("   Trying non-strict loading...")
                    # Try non-strict loading
                    missing, unexpected = self.model.model.load_state_dict(
                        checkpoint, 
                        strict=False
                    )
                    if missing:
                        print(f"   Missing keys: {missing[:5]}...")  # Show first 5
                    if unexpected:
                        print(f"   Unexpected keys: {unexpected[:5]}...")  # Show first 5
                    print("✅ Model weights loaded (strict=False)")
                
            
            # Move model to device and set to eval mode
            print(f"\nMoving model to {self.device}...")
            self.model.to_device(self.device)
            self.model.model.eval()
            print("✅ Model set to evaluation mode")
            
            # Update UI
            epoch = checkpoint.get('epoch', 'unknown')
            classes = ', '.join(self.class_mapping.values())
            device_name = "GPU (CUDA)" if self.device.type == 'cuda' else "CPU"
            
            self.model_info_label.setText(
                f"✅ Model Loaded Successfully!\n\n"
                f"📍 Epoch: {epoch}\n"
                f"🖥️ Device: {device_name}\n"
                f"🔢 Classes: {num_classes}\n"
                f"🏷️ Labels: {classes}\n"
                f"📁 Checkpoint: {Path(checkpoint_path).name}"
            )
            
            self.status_bar.showMessage(f"✅ Model loaded successfully from epoch {epoch}")
            self.load_image_btn.setEnabled(True)
            self.load_folder_btn.setEnabled(True)
            
            print(f"\n{'='*60}")
            print(f"MODEL LOADING COMPLETE!")
            print(f"{'='*60}\n")
            
            QMessageBox.information(
                self,
                "Model Loaded",
                f"✅ Model loaded successfully!\n\n"
                f"Epoch: {epoch}\n"
                f"Device: {device_name}\n"
                f"Classes: {num_classes}\n"
                f"Labels: {classes}"
            )
            
        except FileNotFoundError as e:
            error_msg = f"File not found:\n{str(e)}\n\nPlease check that the file exists."
            print(f"\n❌ FileNotFoundError: {e}")
            QMessageBox.critical(self, "File Not Found", error_msg)
            self.status_bar.showMessage("❌ Failed to load model - file not found")
            
        except KeyError as e:
            error_msg = f"Invalid checkpoint format:\n{str(e)}\n\nThe checkpoint file may be corrupted or from an incompatible version."
            print(f"\n❌ KeyError: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(self, "Invalid Checkpoint", error_msg)
            self.status_bar.showMessage("❌ Failed to load model - invalid format")
            
        except RuntimeError as e:
            error_msg = (
                f"Model architecture mismatch:\n\n{str(e)}\n\n"
                f"This usually means:\n"
                f"- The checkpoint was trained with different number of classes\n"
                f"- The model architecture has changed\n"
                f"- The checkpoint file is corrupted\n\n"
                f"Try:\n"
                f"- Using the correct checkpoint file\n"
                f"- Checking the class mapping file\n"
                f"- Re-training the model"
            )
            print(f"\n❌ RuntimeError: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(self, "Model Architecture Error", error_msg)
            self.status_bar.showMessage("❌ Failed to load model - architecture mismatch")
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid class mapping file:\n{str(e)}\n\nThe JSON file is malformed."
            print(f"\n❌ JSONDecodeError: {e}")
            QMessageBox.critical(self, "Invalid JSON", error_msg)
            self.status_bar.showMessage("❌ Failed to load model - invalid JSON")
            
        except Exception as e:
            error_msg = (
                f"Unexpected error loading model:\n\n{str(e)}\n\n"
                f"Error type: {type(e).__name__}\n\n"
                f"Please check the console output for details."
            )
            print(f"\n❌ Exception: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(self, "Error Loading Model", error_msg)
            self.status_bar.showMessage("❌ Failed to load model")
    
    def load_single_image(self):
        """Load a single image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        
        if file_path:
            self.image_paths = [file_path]
            self.update_image_list()
            self.load_image_to_viewer(file_path)
            self.predict_btn.setEnabled(True)
            self.status_bar.showMessage(f"Loaded: {Path(file_path).name}")
    
    def load_folder(self):
        """Load all images from a folder."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder Containing Images"
        )
        
        if folder_path:
            # Get all image files
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            self.image_paths = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(image_extensions)
            ]
            
            if self.image_paths:
                self.update_image_list()
                self.load_image_to_viewer(self.image_paths[0])
                self.predict_btn.setEnabled(True)
                self.status_bar.showMessage(f"✅ Loaded {len(self.image_paths)} images from folder")
            else:
                QMessageBox.warning(
                    self,
                    "No Images Found",
                    f"No image files found in:\n{folder_path}\n\nSupported formats: JPG, PNG, BMP"
                )
    
    def update_image_list(self):
        """Update the image list widget."""
        self.image_list.clear()
        for path in self.image_paths:
            self.image_list.addItem(Path(path).name)
        
        if self.image_paths:
            self.image_list.setCurrentRow(0)
    
    def on_image_selected(self, item):
        """Handle image selection from list."""
        index = self.image_list.currentRow()
        if 0 <= index < len(self.image_paths):
            self.load_image_to_viewer(self.image_paths[index])
    
    def load_image_to_viewer(self, image_path):
        """Load an image to the viewer."""
        self.current_image_path = image_path
        self.image_viewer.load_image(image_path)
        
        # Load existing results if available
        if image_path in self.results:
            detections = self.results[image_path]
            self.image_viewer.set_detections(detections)
            self.display_results(detections)
        else:
            self.image_viewer.set_detections([])
            self.results_text.clear()
        
        self.status_bar.showMessage(f"Viewing: {Path(image_path).name}")
    
    def on_conf_changed(self, value):
        """Handle confidence threshold change."""
        conf = value / 100.0
        self.conf_value_label.setText(f"{conf:.2f}")
    
    def on_show_boxes_changed(self, state):
        """Handle show boxes checkbox change."""
        self.image_viewer.toggle_boxes(state == Qt.Checked)
    
    def run_prediction(self):
        """Run prediction on loaded images."""
        if not self.image_paths or self.model is None:
            return
        
        # Disable controls
        self.predict_btn.setEnabled(False)
        self.load_image_btn.setEnabled(False)
        self.load_folder_btn.setEnabled(False)
        self.load_model_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Get confidence threshold
        conf_threshold = self.conf_slider.value() / 100.0
        
        # Create and start worker
        self.worker = PredictionWorker(
            self.model,
            self.class_mapping,
            self.device,
            self.image_paths,
            conf_threshold
        )
        self.worker.progress.connect(self.on_prediction_progress)
        self.worker.finished.connect(self.on_prediction_finished)
        self.worker.error.connect(self.on_prediction_error)
        self.worker.start()
    
    def on_prediction_progress(self, value, message):
        """Handle prediction progress updates."""
        self.progress_bar.setValue(value)
        self.status_bar.showMessage(message)
    
    def on_prediction_finished(self, results):
        """Handle prediction completion."""
        self.results = results
        
        # Enable controls
        self.predict_btn.setEnabled(True)
        self.load_image_btn.setEnabled(True)
        self.load_folder_btn.setEnabled(True)
        self.load_model_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.save_image_btn.setEnabled(True)
        self.save_all_btn.setEnabled(True)
        self.export_json_btn.setEnabled(True)
        
        # Display results for current image
        if self.current_image_path in results:
            detections = results[self.current_image_path]
            self.image_viewer.set_detections(detections)
            self.display_results(detections)
        
        # Show summary
        total_detections = sum(len(dets) for dets in results.values())
        self.status_bar.showMessage(
            f"✅ Prediction complete! Found {total_detections} objects in {len(results)} images"
        )
        
        QMessageBox.information(
            self,
            "Prediction Complete",
            f"✅ Successfully processed {len(results)} images\n\n"
            f"📊 Total detections: {total_detections}\n"
            f"🎯 Confidence threshold: {self.conf_slider.value() / 100.0:.2f}"
        )
    
    def on_prediction_error(self, error_msg):
        """Handle prediction errors."""
        self.predict_btn.setEnabled(True)
        self.load_image_btn.setEnabled(True)
        self.load_folder_btn.setEnabled(True)
        self.load_model_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        QMessageBox.critical(
            self,
            "Prediction Error",
            f"An error occurred during prediction:\n\n{error_msg}\n\nPlease check the console for details."
        )
        self.status_bar.showMessage("❌ Prediction failed")
    
    def display_results(self, detections):
        """Display detection results in text area."""
        if not detections:
            self.results_text.setPlainText(
                "No objects detected with current confidence threshold.\n\n"
                f"Threshold: {self.conf_slider.value() / 100.0:.2f}\n"
                "Try lowering the threshold to see more detections."
            )
            return
        
        text = f"Found {len(detections)} object(s):\n"
        text += "=" * 50 + "\n\n"
        
        for i, det in enumerate(detections, 1):
            text += f"{i}. {det['label'].upper()}\n"
            text += f"   Confidence: {det['score']:.3f} ({det['score']*100:.1f}%)\n"
            x1, y1, x2, y2 = det['box']
            text += f"   Bounding Box: ({x1:.1f}, {y1:.1f}) → ({x2:.1f}, {y2:.1f})\n"
            text += f"   Size: {x2-x1:.1f} x {y2-y1:.1f} pixels\n\n"
        
        self.results_text.setPlainText(text)
    
    def save_current_image(self):
        """Save the current annotated image."""
        if not self.current_image_path or self.current_image_path not in self.results:
            QMessageBox.warning(
                self,
                "No Results",
                "No prediction results available for current image.\nRun prediction first."
            )
            return
        
        # Get save path
        default_name = Path(self.current_image_path).stem + "_predicted.jpg"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Annotated Image",
            default_name,
            "JPEG Image (*.jpg);;PNG Image (*.png);;All Files (*)"
        )
        
        if save_path:
            self.save_annotated_image(
                self.current_image_path,
                self.results[self.current_image_path],
                save_path
            )
            QMessageBox.information(
                self,
                "Image Saved",
                f"Annotated image saved successfully!\n\n{save_path}"
            )
            self.status_bar.showMessage(f"✅ Saved: {Path(save_path).name}")
    
    def save_all_images(self):
        """Save all annotated images."""
        if not self.results:
            QMessageBox.warning(
                self,
                "No Results",
                "No prediction results available.\nRun prediction first."
            )
            return
        
        # Get output folder
        output_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder for Annotated Images"
        )
        
        if output_folder:
            saved_count = 0
            errors = []
            
            for image_path, detections in self.results.items():
                try:
                    base_name = Path(image_path).stem
                    output_path = os.path.join(output_folder, f"{base_name}_predicted.jpg")
                    self.save_annotated_image(image_path, detections, output_path)
                    saved_count += 1
                except Exception as e:
                    errors.append(f"{Path(image_path).name}: {str(e)}")
            
            if errors:
                error_msg = "\n".join(errors[:5])  # Show first 5 errors
                QMessageBox.warning(
                    self,
                    "Some Images Failed",
                    f"Saved {saved_count} images successfully.\n\n"
                    f"Errors:\n{error_msg}"
                )
            else:
                QMessageBox.information(
                    self,
                    "All Images Saved",
                    f"✅ Successfully saved {saved_count} annotated images!\n\nLocation:\n{output_folder}"
                )
            
            self.status_bar.showMessage(f"✅ Saved {saved_count} images to {output_folder}")
    
    def save_annotated_image(self, image_path, detections, output_path):
        """Save an annotated image."""
        # Open image
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
        
        # Color mapping
        colors = {
            'gun': (255, 0, 0),
            'knife': (255, 165, 0),
            'weapon': (255, 255, 0),
            'pistol': (255, 50, 50),
            'rifle': (200, 0, 0),
        }
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            score = det['score']
            
            color = colors.get(label.lower(), (0, 255, 0))
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            
            # Draw label
            text = f"{label} {score:.2f}"
            text_bbox = draw.textbbox((x1, y1 - 30), text, font=font)
            text_bbox = (text_bbox[0] - 5, text_bbox[1] - 2, text_bbox[2] + 5, text_bbox[3] + 2)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - 30), text, fill=(255, 255, 255), font=font)
        
        img.save(output_path, quality=95)
    
    def export_results(self):
        """Export results to JSON."""
        if not self.results:
            QMessageBox.warning(
                self,
                "No Results",
                "No prediction results available.\nRun prediction first."
            )
            return
        
        # Get save path
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results to JSON",
            "predictions_results.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if save_path:
            # Create export data with metadata
            export_data = {
                "metadata": {
                    "total_images": len(self.results),
                    "total_detections": sum(len(dets) for dets in self.results.values()),
                    "confidence_threshold": self.conf_slider.value() / 100.0,
                    "classes": list(self.class_mapping.values())
                },
                "predictions": self.results
            }
            
            with open(save_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            QMessageBox.information(
                self,
                "Results Exported",
                f"✅ Results exported successfully!\n\n{save_path}\n\n"
                f"Total images: {len(self.results)}\n"
                f"Total detections: {sum(len(dets) for dets in self.results.values())}"
            )
            self.status_bar.showMessage(f"✅ Exported results to {Path(save_path).name}")
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                'Prediction in Progress',
                'A prediction is currently running. Are you sure you want to quit?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Object Detection Predictor")
    app.setStyle('Fusion')  # Use Fusion style for better dark theme
    
    window = PredictionUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Quick launcher for the Prediction UI

Usage:
    python run_ui.py
    python run_ui.py --checkpoint path/to/model.pth --classes path/to/classes.json
"""

import sys
import os
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Ensure PyQt5 is installed
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    print("❌ PyQt5 is not installed!")
    print("   Install it with: pip install PyQt5")
    sys.exit(1)

# Import the UI
from predict_ui import PredictionUI


def main():
    """Launch the application."""
    parser = argparse.ArgumentParser(description='Object Detection Prediction UI')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint (optional)')
    parser.add_argument('--classes', type=str, help='Path to class mapping JSON (optional)')
    
    args = parser.parse_args()
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Object Detection Predictor")
    
    # Create main window
    window = PredictionUI()
    
    # Auto-load model if paths provided
    if args.checkpoint and args.classes:
        if os.path.exists(args.checkpoint) and os.path.exists(args.classes):
            print(f"Auto-loading model from {args.checkpoint} and classes {args.classes}")
            window.load_model_from_paths(args.checkpoint, args.classes)
        else:
            print("⚠️  Warning: Provided checkpoint or class file not found")
    
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

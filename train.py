#!/usr/bin/env python3
"""
Simple YOLOv8 Object Detection Training Script
"""

from ultralytics import YOLO

# =============================================================================
# CONFIGURATION VARIABLES - MODIFY THESE AS NEEDED
# =============================================================================

# Dataset
DATA_YAML = "mango_tree_classification/data.yaml"

# Model
MODEL_SIZE = "n"                    # 'n', 's', 'm', 'l', 'x'

# Training Parameters
EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 16
LEARNING_RATE = 0.01
PATIENCE = 50


# Output
PROJECT_DIR = "runs/classify"
EXPERIMENT_NAME = "mango_tree_training"

# =============================================================================


def main():
    print("Starting YOLOv8 Object Detection Training...")
    
    model = YOLO(f"yolov8{MODEL_SIZE}.pt")
    
    # Train
    # The model.train() function is smart and will automatically run in 'detect' mode
    # because you loaded a detection model.
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        # ... (all your other training parameters are fine)
    )
    
    print(f"\nTraining completed!")
    print(f"Best model saved: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()

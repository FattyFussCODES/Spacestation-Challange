# Spacestation-Challange
Dataset
Test Set: 1,408 images containing 5,729 annotated instances
Conditions: Includes diverse lighting variations (dark, light, very dark, very light) and varying levels of scene clutter
Annotation Format: YOLO format with normalized bounding box coordinates
Model Architecture
Base Model: YOLOv8s (11.1M parameters, 72 layers)
Input Resolution: 640 × 640 pixels
Training Details: Trained for 10 epochs using the AdamW optimizer with early stopping enabled
Optimizer: AdamW
Learning Rate: 8e-05 → 5e-05
Momentum: 0.9
Mosaic Augmentation: 0.4
Patience (Early Stopping): 6 epochs
Batch Size: 16
Overall Performance (Test Set)
Metric Score
mAP@0.5 ->0.691
mAP@0.5–0.95->0.594
mAP@0.75->0.632
Inference Speed ~80 ms per image

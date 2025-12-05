# Model Artifacts

This directory contains **trained model weights** (binary files), NOT Python code.

## Contents

- `yolo_fashion.pt` - Fine-tuned YOLOv8 detection model
- `classifier.pth` - EfficientNet garment classifier
- `tagger.pth` - Attribute tagging model
- `compatibility.pth` - Learned outfit compatibility model (V1+)
- `clip/` - CLIP model weights

## Note

- These are binary files, not Python modules
- Python ORM models live in `backend/app/models/`
- Use Git LFS if versioning these files


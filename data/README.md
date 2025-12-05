# Data Directory

This directory is split into two concerns:

## `runtime/` - User-facing runtime data

- `uploads/` - Original user-uploaded images
- `processed/` - Cropped/processed garment images
- `thumbnails/` - Thumbnail cache

**Note:** This data is user-generated and should be backed up separately. Can be moved to S3 in production.

## `datasets/` - Training datasets

- `deepfashion2/` - DeepFashion2 dataset for detection/classification
- `imaterialist/` - iMaterialist Fashion dataset for attributes
- `polyvore/` - Polyvore Outfits dataset for compatibility

**Note:** These are large datasets (gitignored). Can be symlinked to external storage.


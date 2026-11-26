# Deep Residual Attention U-Net for Retinal Vessel Segmentation

## Project Overview

This project implements a Deep Residual Attention U-Net for retinal blood vessel segmentation from colour fundus images. It combines residual convolution blocks, attention gates, and an ASPP-inspired bridge to help identify thin and low-contrast retinal vessels.

## Key Features

- Green-channel extraction for retinal vessel contrast
- CLAHE and gamma correction for image enhancement
- Z-score normalization followed by min-max scaling
- Residual convolution blocks using depth-wise separable convolutions
- Attention gates on encoder-decoder skip connections
- ASPP-inspired multi-scale feature extraction
- Binary Cross-Entropy + Dice loss
- Dice coefficient, IoU, and binary accuracy metrics
- Morphological closing and skeletonization for prediction refinement

## Model Architecture

The network contains:
1. Encoder with four residual convolution blocks and max-pooling
2. ASPP-inspired bridge with multiple dilation rates
3. Decoder with upsampling and attention-gated skip connections
4. 1 x 1 convolution with sigmoid activation for the vessel probability map

## Dataset

The research paper describes evaluation using the FIVES (Fundus Image Vessel Segmentation) dataset, containing 800 colour fundus images. The paper reports a split of 600 training images and 200 testing images.

The dataset is not included in this repository. Obtain it from its official source and arrange the files as follows:

```text
dataset/
├── train/
│   ├── images/
│   └── masks/
└── test/
    └── images/
```

Update `BASE_DIR` in `train.py` to point to your local dataset folder before running the script.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## How to Run

1. Download and arrange the dataset using the folder structure above.
2. Update `BASE_DIR` in `train.py`.
3. Run:

```bash
python train.py
```

The script trains the model, saves the best model checkpoint, and writes raw, binary, closed, and skeletonized predictions to the results directory.

## Training Configuration

The uploaded implementation uses:
- Image size: 224 x 224
- Batch size: 8
- Maximum epochs: 150
- Optimizer: Adam
- Initial learning rate: 0.001
- Loss: Binary Cross-Entropy + Dice Loss
- Early stopping patience: 15
- Reduce learning rate on validation-loss plateau

## Reported Research Results

The accompanying research paper reports the following proposed-model results:

| Metric | Reported value |
|---|---:|
| Accuracy | 97.77% |
| F1-score | 0.816 |
| Recall | 0.80 |
| Mean IoU | 0.692 |

These are the values reported in the paper; results from a new run may differ depending on the dataset split, preprocessing, and training environment.



## Notes

- The FIVES dataset and trained model weights are not included in this repository.
- Check dataset terms before downloading or redistributing data.
- This project is for academic and research purposes and is not a medical diagnostic device.

  
## Research Publication

**Title:** A Deep Residual Attention U-Net for Robust Retinal Vessel Segmentation

**Conference:** 2025 International Conference on Signal Processing, Computation, Electronics, Power and Telecommunication (IConSCEPT).

**DOI:** https://doi.org/10.1109/IConSCEPT66142.2025.11436904


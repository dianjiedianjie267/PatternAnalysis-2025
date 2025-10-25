# Brain MRI Segmentation (OASIS) - 2D U-Net Pipeline

## 1. Task Overview
This project performs 2D semantic segmentation on brain MRI slices.
The goal is to separate brain tissues in each slice using a U-Net style model.

## 2. Why this is useful
Accurate brain tissue segmentation is important for medical analysis and planning.
This task is a standard benchmark in medical imaging and gives us a controlled way
to measure model quality using Dice score.

## 3. Method (short summary)
We use a U-Net style encoder–decoder with skip connections.
The model takes a 2D MRI slice and predicts a pixel-wise mask.
Training is supervised with ground truth labels.

## 4. Repo structure
- dataset.py : load and preprocess OASIS MRI slices
- modules.py : neural network model and loss/metrics
- train.py   : training and validation loop
- predict.py : inference demo using a trained checkpoint

## 5. Planned results
We will report Dice score on a held-out test set and include a visual example
(original slice + predicted mask overlay).


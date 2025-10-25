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

## 6. How to train
1. The brain MRI dataset for this assignment is already prepared on the COMP3710 cluster at:

   /home/groups/comp3710/OASIS

   It contains 2D PNG brain MRI slices and segmentation masks. The slices are already split by subject into:
   - keras_png_slices_train / keras_png_slices_seg_train
   - keras_png_slices_validate / keras_png_slices_seg_validate
   - keras_png_slices_test / keras_png_slices_seg_test

   So there is already a train / validate / test split with no patient leakage.

2. To train on the cluster (from within a Python environment with PyTorch):

        python train.py \
            --data_root /home/groups/comp3710/OASIS \
            --epochs 20 \
            --batch_size 4 \
            --lr 1e-3

3. The training script will:
   - train a 2D U-Net for brain MRI tissue segmentation
   - save the best checkpoint to `best_model.pth`
   - save a curve plot as `training_curve.png`
   - print validation Dice each epoch


## 7. How to run inference
After training, run:

    python predict.py

This will:
- load `best_model.pth`
- run the trained network on a test MRI slice
- generate `prediction_example.png`, which shows:
  - the input MRI slice
  - the ground truth mask
  - the predicted mask

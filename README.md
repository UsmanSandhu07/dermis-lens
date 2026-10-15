# Dermis Lens

Binary lesion screening demo: EfficientNet-B0 fine-tuned for benign vs malignant, plus a Streamlit viewer that scores a single image.

This is a research prototype, not a medical device.

## Results (held-out combined test, 2,660 images)

| Metric | Value |
| --- | --- |
| AUROC | 0.9659 |
| Accuracy | 0.9019 |
| Recall (malignant) | 0.8854 |
| Specificity | 0.9176 |

Training used mixed precision on a T4, ImageNet-normalized 224×224 inputs, weighted BCE (`pos_weight` 1.14), and fold 0 of StratifiedGroupKFold (seed 42). Best checkpoint: epoch 24.

## Local demo

```bash
pip install -r requirements.txt
streamlit run mainapp.py
```

Weights file in this folder: `Fold0_efficientnet_AUROC0.9614_epoch20.pth`.

## Data mix

Two public sources were merged (melanoma set + skin-cancer set). Training ~14.5k images, test 2,660, roughly balanced at evaluation time. Augmentation is albumentations (flips, rotate, shift-scale, brightness/contrast).

## Head

Classifier on 1280-d features:

`Dropout 0.3 → Linear 256 → ReLU → BatchNorm → Dropout 0.2 → Linear 1`

Early EfficientNet blocks stay frozen; last three blocks plus the head are trained.

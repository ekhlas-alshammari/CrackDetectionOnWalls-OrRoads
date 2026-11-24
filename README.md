# CrackDetectionOnWalls-OrRoads

## Project
This project classifies surface cracks (crack vs. no crack) on walls and roads from images to help building and road maintenance inspectors spot potential surface damage more quickly.

- **Developers:** Ekhlas Alshammari, Roa Alnazawi
- **Project Type:** Team capstone project (SDAIA Academy — Computer Vision, November 2026)
- **Domain:** Surface crack classification on walls and roads (`Negative` / No Crack vs. `Positive` / Crack)
- 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ekhlas-alshammari/CrackDetectionOnWalls-OrRoads/blob/main/project.ipynb)

## Dataset
- **Source:** `arunrk7/surface-crack-detection` on Kaggle — https://www.kaggle.com/datasets/arunrk7/surface-crack-detection (40,000 total images across two classes: `Negative` and `Positive`).
- **Sampling & Split:** Using `SEED = 42` with SHA-256 deduplication (skipping 10 exact duplicates), a balanced subset of 3,000 images (1,500 per class) is split into:
  - **Train:** 2,100 images (70%; 1,050 per class)
  - **Validation:** 450 images (15%; 225 per class)
  - **Test:** 450 images (15%; 225 per class)
- **Sample Grid:** A 2x4 sample image grid is generated in `project.ipynb`.

## Model
- **Architecture:** Pretrained `ResNet-18` (PyTorch) with a binary classification head (`224x224` input with ImageNet normalization).
- **Training Settings:** Fine-tuned for 4 epochs with `batch_size = 32`, `CrossEntropyLoss`, `Adam` (`lr = 1e-4`), and deterministic CUDA settings (`SEED = 42`) on a Google Colab T4 GPU.

## Results
### Training & Validation Curves
![Training and Evaluation Dashboard](dashboard.png)

The training and validation loss and accuracy curves across the 4 epochs show steady learning and convergence without notable overfitting.

### Before vs. After Training (Test Set — 450 Images)
| Metric | Before Training | After Training |
| :--- | :--- | :--- |
| **Loss** | `0.7386` | `0.0098` |
| **Accuracy** | `50.00%` (`0.5000`) | `99.78%` (`0.9978`) |
| **Precision** | `50.00%` (`0.5000`) | `100.00%` (`1.0000`) |
| **Recall** | `100.00%` (`1.0000`) | `99.56%` (`0.9956`) |
| **F1-Score** | `0.6667` | `0.9978` |

![Final Analysis](final_analysis.png)

### Confusion Matrix & Summary of Change
- **Test Set Confusion Matrix (After Training):** `TN = 225`, `FP = 0`, `FN = 1`, `TP = 224` across 450 test images. *(You can also embed your confusion matrix image here.)*
- **What Changed and Why:** Before fine-tuning, the model performed near random/majority-class guessing (`50.00%` accuracy and `0.7386` loss) because the new classification head was untrained on crack features. After 4 epochs of fine-tuning on the training set, loss dropped to `0.0098` and accuracy reached `99.78%` (`449/450` correct) as the network adapted its representations to wall and road surface cracks.
- **Limitations:** Image-level classification only (no bounding boxes or pixel segmentation), SHA-256 removes identical files rather than visually similar crops, and real-world field images with harsh lighting, shadows, or motion blur may behave differently.

## How to Run
1. Open `project.ipynb` in Google Colab.
2. Set the runtime to GPU (`Runtime -> Change runtime type -> T4 GPU`).
3. If prompted for Kaggle credentials, add `KAGGLE_USERNAME` and `KAGGLE_KEY` in Colab Secrets.
4. Run all cells in order from top to bottom to reproduce the dataset sampling, splits, training, evaluation metrics, and plots.

## Training-program attribution
This project was completed for the Computer Vision capstone using `ResNet-18` for wall and road surface crack classification, delivered by SDAIA Academy via Learning Space as a 3-day program. Session: November 2026.

Training-program reference: [SDAIA Academy on GitHub](https://github.com/SDAIAAcademy).

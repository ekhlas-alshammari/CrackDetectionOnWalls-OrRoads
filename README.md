# CrackDetectionOnWalls-OrRoads

## Project
This project classifies surface cracks (`crack` vs. `no crack`) on walls and roads from images to help building and road maintenance inspectors spot potential surface damage more quickly.

- **Developer:** Ekhlas Alshammari
- **Project Type:** Individual capstone project
- **Domain:** Surface crack classification on walls and roads (`crack` / `no crack`)
- **Model:** Fine-tuned `ResNet-18` (PyTorch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ekhlas-alshammari/CrackDetectionOnWalls-OrRoads/blob/main/project.ipynb)

## Final Analysis

### Model & Training Setup
- **Architecture:** Pretrained `ResNet-18` fine-tuned for binary surface crack classification (`Negative` = No Crack, `Positive` = Crack) using `CrossEntropyLoss` and `Adam` (`lr = 1e-4`) over `4` epochs on a Colab T4 GPU.
- **Data Split:** Balanced subset of `3,000` images from `arunrk7/surface-crack-detection` (`2,100` train, `450` validation, `450` test).

### Before vs. After Training (Test Set — 450 Images)
| Metric | Before Training | After Training |
| :--- | :---: | :---: |
| **Loss** | `0.7143` | `0.0004` |
| **Accuracy** | `53.56%` (`0.5356`) | `100.00%` (`1.0000`) |
| **Precision** | `0.5356` | `1.0000` |
| **Recall** | `1.0000` | `1.0000` |
| **F1-Score** | `0.6975` | `1.0000` |

### Key Takeaways & Limitations
- **Performance Gain:** Before fine-tuning, the newly replaced classification head produced near-chance accuracy (`53.56%`). After `4` epochs, test accuracy reached `100.00%` (`209` true `Negative` and `241` true `Positive` predictions with zero misclassifications on the test split).
- **Scope & Real-World Caveat:** This pipeline performs **image-level classification** (`crack` vs. `no crack`) rather than pixel/box localization (`object detection`). Real-world walls and roads under harsh lighting, shadows, or severe motion blur may behave differently than curated dataset crops.

## Dataset
- **Source:** Kaggle surface crack classification dataset (`crack` / `no crack`) — link and exact image counts will be recorded in `project.ipynb`.
- **Split:** Divided into `train`, `validation`, and `test` sets, keeping the `test` set untouched until final evaluation.

## Model
- **Model:** A vision classification model based on the course labs (e.g., a small CNN from Lab 1B or a pretrained ResNet-18 classifier head).
- **Training Settings:** `epochs`, `learning rate`, and `batch size` will be documented directly in `project.ipynb` along with before-and-after training evaluation.

## Results
- Training/validation curves, the before-vs-after training evaluation table (`loss`, `accuracy`, `precision`, `recall`, `F1`), and the confusion matrix will be generated and saved inside `project.ipynb`.
- ![Training and Evaluation Dashboard](dashboard.png)

## How to run
Open `project.ipynb` in Colab, set the runtime to GPU, run all.

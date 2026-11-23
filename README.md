# CrackDetectionOnWalls-OrRoads

## Project
This project classifies wall and road surface images into `crack` (`Positive`) and `no crack` (`Negative`) using transfer learning with `ResNet-18` in PyTorch to help building and road maintenance inspectors spot potential surface damage more quickly.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ekhlas-alshammari/CrackDetectionOnWalls-OrRoads/blob/main/project.ipynb)

## Dataset
- **Kaggle Link:** https://www.kaggle.com/datasets/arunrk7/surface-crack-detection
- **Classes:** Binary classification (`Negative` / no crack vs. `Positive` / crack).
- **Number of Images & Split:** `3,000` balanced images (`1,500` per class) split into `2,100` train (`70%`), `450` validation (`15%`), and `450` test (`15%`) images. The test set is kept untouched until evaluation, and the dataset is downloaded inside `project.ipynb` rather than committed to the repository.
- **Sample Images:** A sample image grid (`2x4`) is generated directly in `project.ipynb`, alongside visual summaries in `final_analysis.png`.

## Model
- **Architecture:** Pretrained `ResNet-18` with a replaced binary classification head in PyTorch.
- **Training Settings:** `4` epochs, `Adam` optimizer (`lr = 1e-4`), `batch_size = 32`, and `CrossEntropyLoss` on Colab GPU (`T4`).
- **Before vs. After Training (Test Set — 450 Images):**

| Metric | Before Training | After Training |
| :--- | :---: | :---: |
| **Loss** | `0.7143` | `0.0004` |
| **Accuracy** | `53.56%` (`0.5356`) | `100.00%` (`1.0000`) |
| **Precision** | `0.5356` | `1.0000` |
| **Recall** | `1.0000` | `1.0000` |
| **F1-Score** | `0.6975` | `1.0000` |

Before training, the untrained classification head performed near chance (`53.56%` accuracy on the test set). After `4` epochs of fine-tuning, test accuracy reached `100.00%` (`209` true negatives and `241` true positives) because the model weights adapted to the crack and non-crack patterns in this dataset. Keep in mind that this is image-level classification on surface crops and may behave differently on wide field photos with heavy shadows, glare, or motion blur.

## Results
Training and validation loss/accuracy curves per epoch, the test confusion matrix, and before/after evaluation summaries are generated in `project.ipynb` and shown below. The curves show rapid early convergence and steady performance across the `4` epochs without clear overfitting.

![Training and Evaluation Dashboard](dashboard.png)

![Final Analysis](final_analysis.png)

## How to run
Open `project.ipynb` in Colab, set the runtime to GPU, run all.

## Training-program attribution
This project was completed for the Computer Vision capstone using `ResNet-18` for wall and road surface crack classification, delivered by SDAIA Academy via Learning Space as a 3-day program. Session: November 2026.

Training-program reference: [SDAIA Academy on GitHub](https://github.com/SDAIAAcademy).

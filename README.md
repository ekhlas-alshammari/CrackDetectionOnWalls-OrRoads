# CrackDetectionOnWalls-OrRoads

## Project
This project classifies surface cracks (`crack` vs. `no crack`) on walls and roads from images to help building and road maintenance inspectors spot potential surface damage more quickly.

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

# Loan Default Prediction with Federated Learning

A PyTorch project that predicts loan default risk and compares three training paradigms on the same neural network: **centralized training**, **federated averaging (FedAvg)**, and a **multi-armed-bandit-driven federated learning (FL-MAB)** approach that intelligently selects which clients participate in each training round.

## Overview

The notebook trains a deep neural network to classify loan applicants as default risk (`Risk_Flag`) or not, then simulates a federated learning environment where the dataset is split across 5 synthetic "clients" (e.g. banks or branches) using unsupervised clustering. It evaluates whether federated training — with and without smart client selection — can match the performance of training on all the data centrally.

**Key question explored:** Can federated learning approaches (FedAvg, FL-MAB) match centralized model performance while keeping data decentralized?

## Dataset

- **Source:** `train.csv` (Kaggle dataset, loan default data — originally loaded from `/kaggle/input/data-fed-learn/train.csv`)
- **Target variable:** `Risk_Flag` (binary: 1 = defaulted/high risk, 0 = no default)
- **Features include:** Income, Age, Experience, Married/Single status, House/Car ownership, Profession, City, State, Current Job Years, Current House Years
- **Engineered features:**
  - `Income_Age_Ratio` — income normalized by age
  - `Exp_Job_Ratio` — total experience normalized by current job tenure
  - `Stability_Score` — combined job + house tenure
  - `Married_Income_Interaction` — income interaction with marital status

## Methodology

### 1. Preprocessing
- Drop ID column, engineer 4 new features
- Label-encode categorical/low-cardinality columns
- Stratified 3-way split: train (80%) / validation (~11%) / test (10%)
- Standardize features with `StandardScaler`

### 2. Model Architecture — `UpgradedLoanDefaultNN`
A feed-forward neural network:
- Hidden layers: `[256, 128, 64, 32]`
- Each block: `Linear → LayerNorm → ReLU → Dropout(0.3)`
- Output: `Linear → Sigmoid` (binary classification)
- Kaiming weight initialization

### 3. Loss Function — `EnhancedFocalLoss`
A focal loss variant (`alpha=0.25`, `gamma=3.0`) designed to handle class imbalance between defaulters and non-defaulters by down-weighting easy/majority examples.

### 4. Training Paradigms

| Approach | Description |
|---|---|
| **Central** | Standard training on the full training set (300 epochs, `AdamW`, `OneCycleLR` scheduler) |
| **FedAvg** | Data split into 5 clients via GMM clustering on PCA-reduced features (20 non-overlapping chunks per client). Each round, all 5 clients train locally (15 epochs) and their weights are averaged |
| **FL-MAB** | Same client setup as FedAvg, but client participation each round is chosen by a **UCB (Upper Confidence Bound) multi-armed bandit**, selecting the top-3 most promising clients per round instead of all 5 |

Both federated approaches run for **20 rounds** with **15 local epochs per client per round**.

### 5. Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Evaluated at both the default threshold (0.5) and the F1-optimal threshold per model
- Visualizations: training curves, client F1 trends, FL-MAB client selection heatmap, prediction score distributions, ROC/PR curves

## Project Structure

```
.
├── Loan_Default_prediction.ipynb   # Main notebook (this project)
├── train.csv                       # Input dataset (not included — see Data Setup)
└── outputs/
    ├── central_model_final.pth         # Trained centralized model weights
    ├── fedavg_model_20r_15e.pth        # Trained FedAvg model weights
    ├── flmab_model_20r_15e.pth         # Trained FL-MAB model weights
    ├── central_history.csv             # Centralized training history
    ├── fedavg_history.csv              # FedAvg training history
    ├── flmab_history.csv               # FL-MAB training history
    ├── final_comparison.csv            # Final metrics comparison table
    ├── training_curves_research.png    # F1/loss curves, client trends, comparison bars
    ├── threshold_histograms_research.png  # Prediction probability distributions
    └── roc_pr_curves.png               # ROC and Precision-Recall curves
```

## Requirements

```
python >= 3.8
pandas
numpy
matplotlib
seaborn
scikit-learn
torch
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch
```

A GPU is optional but recommended — the notebook auto-detects CUDA and falls back to CPU otherwise.

## Data Setup

The notebook expects the dataset at:
```
/kaggle/input/data-fed-learn/train.csv
```

To run outside Kaggle, download the loan default dataset and update the path in the second cell to point to your local copy, e.g.:
```python
df = pd.read_csv("./data/train.csv")
```

## Usage

1. Place `train.csv` in the expected path (or update the path as above).
2. Run all cells in order — the notebook is sequential and each stage depends on outputs from the previous one:
   - Environment setup → preprocessing → train/val/test split
   - Centralized model training
   - Client clustering (GMM) for federated simulation
   - FedAvg training
   - FL-MAB training
   - Evaluation and comparison
   - Plot generation and model/result export

```bash
jupyter notebook Loan_Default_prediction.ipynb
```

## Results

The notebook produces a final comparison table (`final_comparison.csv`) reporting Accuracy, Precision, Recall, F1-score, and ROC-AUC for each of the three approaches (Central, FedAvg, FL-MAB) at their optimal classification thresholds, letting you directly compare whether decentralized training closes the gap with centralized training — and whether smart client selection (FL-MAB) outperforms naive FedAvg.

## Reproducibility

Random seeds are fixed throughout (`numpy`, `torch`, CUDA) with `seed=42`, and deterministic cuDNN settings are enabled for reproducible results across runs.

## Notes

- Client "banks" are synthetic — created via Gaussian Mixture Model clustering on PCA-reduced features, simulating realistic non-IID data distribution across clients (a key challenge in real federated learning).
- The UCB bandit in FL-MAB balances exploration (trying under-sampled clients) and exploitation (favoring clients with historically high F1 contributions).

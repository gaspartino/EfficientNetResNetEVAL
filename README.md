# EfficientNetResNetEval

A framework for training and evaluating deep learning models for traffic light recognition using **ResNet-50** and **EfficientNet-B0**. The project supports robustness evaluation under adversarial attacks and allows multiple independent training runs for statistical analysis.

## Features

- Train and evaluate:
  - ResNet-50
  - EfficientNet-B0
- Support for traffic light datasets:
  - BSTL (Brazilian Traffic Light Dataset)
  - LISA Traffic Light Dataset
- Adversarial robustness evaluation with:
  - FGSM
  - PGD
  - MIM
  - AutoAttack
- Multiple independent training runs.
- Optional ImageNet normalization.
- Automatic model saving and loading.
- Configurable training and evaluation through command-line arguments.

---

## Project Structure

```text
EfficientNetResNetEval/
├── main.py
├── train.py
├── attack.py
├── utils.py
├── saved_models/
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/EfficientNetResNetEval.git
cd EfficientNetResNetEval
```

Install dependencies:

```bash
pip install torchattacks
```

---

## Datasets

Datasets are downloaded automatically through KaggleHub.

### BSTL
Brazilian Traffic Light Dataset containing traffic light images collected under various real-world conditions.

### LISA
Traffic light recognition dataset derived from the LISA Traffic Light Dataset.

---

## Supported Models

| Argument | Model |
|-----------|---------|
| `res` | ResNet-50 |
| `eff` | EfficientNet-B0 |

---

## Command-Line Arguments

| Argument | Description | Default |
|-----------|-------------|---------|
| `-d`, `--dataset` | Dataset to use (`bstl` or `lisa`) | `bstl` |
| `-m`, `--model` | Model architecture (`res` or `eff`) | `res` |
| `-e`, `--epochs` | Number of training epochs | `40` |
| `-b`, `--batch_size` | Training batch size | `512` |
| `-lr`, `--learning_rate` | Learning rate | `0.001` |
| `-ts`, `--test_batch_size` | Test batch size | `256` |
| `-norm`, `--normalize` | Apply ImageNet normalization | Disabled |
| `-ignore_aa`, `--ignore_autoattack` | Skip AutoAttack evaluation | Disabled |
| `-tl`, `--total_loops` | Number of independent training runs | `1` |

---

## Usage Examples

### Train with default settings

```bash
python main.py
```

### Train ResNet-50 on BSTL

```bash
python main.py --dataset bstl --model res
```

### Train EfficientNet-B0 on LISA

```bash
python main.py --dataset lisa --model eff
```

### Train for 100 epochs

```bash
python main.py --epochs 100
```

### Run 10 independent experiments

```bash
python main.py --total_loops 10
```

### Enable ImageNet normalization

```bash
python main.py --normalize
```

### Skip AutoAttack evaluation

```bash
python main.py --ignore_autoattack
```

### Full example

```bash
python main.py \
    --dataset bstl \
    --model eff \
    --epochs 50 \
    --batch_size 256 \
    --learning_rate 0.0005 \
    --normalize \
    --total_loops 10
```

---

## Adversarial Evaluation

The framework evaluates model robustness against several adversarial attacks:

- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent (PGD)
- Momentum Iterative Method (MIM)
- AutoAttack

For each attack and perturbation level, the following metrics are reported:

- Accuracy
- Precision
- Recall
- F1-Score

---

## Model Storage

Trained models are automatically saved in:

```text
saved_models/<dataset>/
```

Examples:

```text
saved_models/bstl/res_0.pth
saved_models/bstl/res_1.pth
saved_models/lisa/eff_0.pth
```

---

## Research Purpose

This repository was developed to facilitate research on adversarial robustness in traffic light recognition systems. It enables systematic comparison of different neural network architectures under multiple attack scenarios and repeated experimental runs.

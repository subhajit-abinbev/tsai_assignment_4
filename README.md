# MNIST Small CNNs

A compact PyTorch project for training small convolutional neural networks on the MNIST dataset. This repository contains a Jupyter notebook (`mnist_cnn_model.ipynb`) that implements multiple small CNN architectures, trains them, evaluates them on the test set, and prints useful model summaries and logs.

## Contents

- `mnist_cnn_model.ipynb` — main notebook with data loading, EDA, three model variants, training and evaluation code.
- `requirements.txt` — Python packages required to run the notebook (use a virtual environment).
- `data/` — (ignored in git) raw MNIST files. A `.gitignore` is included to prevent pushing this folder.

## Quick start

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Open the notebook with Jupyter or VS Code and run cells:

```powershell
jupyter notebook mnist_cnn_model.ipynb
```

Notes:
- The notebook will automatically download the MNIST data into `./data` if it is not already present.
- A `.gitignore` file is present to keep `data/` out of the repo.

## Architectures (simple descriptions)

The notebook defines and experiments with multiple compact CNNs. All are designed for 1×28×28 MNIST grayscale input.

### SmallCNN_1 (minimal parameters)
- conv1: Conv2d(1 -> 4, kernel=3, padding=1)
- bn1: BatchNorm2d(4)
- conv2: Conv2d(4 -> 4, kernel=3, padding=1)
- bn2: BatchNorm2d(4)
- Pooling: two sequential 2×2 max-pool layers (after each conv) → spatial dims reduce 28→14→7
- fc: Linear(4*7*7 -> 10)

Behavior: leaky ReLU activations are used after each conv+bn. This model is extremely small (few parameters) and aims to reach high accuracy while minimizing parameter count.

### SmallCNN_2 (higher accuracy, still compact)
- conv1: Conv2d(1 -> 8, kernel=3, padding=1) + BatchNorm
- conv2: Conv2d(8 -> 16, kernel=3, padding=1) + BatchNorm
- conv3: Conv2d(16 -> 32, kernel=3, padding=1) + BatchNorm
- Pooling: 2×2 max-pool after each conv → dims 28→14→7→3
- dropout: Dropout(0.25) before FC
- fc1: Linear(32*3*3 -> 62)
- fc2: Linear(62 -> 10)

Behavior: leaky ReLU activations and batch normalization are used to improve training stability and generalization.

## Line-by-line contract (what each model expects and returns)
- Inputs: batch tensors of shape `(batch_size, 1, 28, 28)` (float, normalized)
- Outputs: raw logits of shape `(batch_size, 10)`; feed these to `CrossEntropyLoss` or `softmax`+`argmax` for predictions.
- Error modes: mismatched input shapes or incorrect dtypes will raise standard PyTorch errors.

## What logs and outputs the notebook produces
When you run the notebook, expect the following printed outputs and artifacts (these are the "logs"):

- "CUDA Available? True/False" — printed early, indicates whether GPU is available.
- Mean and standard deviation of raw training images — used to create the normalization transform; printed as two floats.
- EDA visualization — matplotlib figures showing sample MNIST images.
- During training (the provided loops):
  - No per-batch printed progress by default; the loop computes and updates model weights silently.
  - After training, the notebook runs evaluation loops that print accuracy values:
    - `Train Accuracy: xx.xx%` (when computed)
    - `Test Accuracy: xx.xx%`
- `torchsummary` model summary tables — for each model you call `summary(model, input_size=(1,28,28))` and the notebook prints per-layer output shapes and parameter counts.
- Final evaluation function output — an explicit printed `Train Accuracy` and `Test Accuracy` for `SmallCNN_2`.

If you want more detailed logs (per-epoch loss, learning curves, or TensorBoard integration), consider adding:
- print statements inside the training loop (loss per batch/epoch)
- store losses/accuracies in lists and plot them with matplotlib
- integrate TensorBoard (`torch.utils.tensorboard.SummaryWriter`) for scalars and model graphs

## Notes on reproducibility and data
- The notebook uses the MNIST dataset from `torchvision.datasets`. If the raw files already exist under `data/MNIST/raw` they won't be re-downloaded.
- The repository includes a `.gitignore` that prevents `data/` from being committed. This keeps the repository lightweight and avoids leaking large data files.

## Where to look in the notebook
- Preprocessing and mean/std computation: the cells that load the full training set and print `.mean()` and `.std()`.
- Model definitions: search the notebook for the classes `SmallCNN`, `SmallCNN_1`, and `SmallCNN_2`.
- Training loops: cells that create `model`, `criterion`, `optimizer` and then iterate `for images, labels in trainloader:`.
- Evaluation: the `evaluate` function and the `model.eval()` loops.

## Next steps / suggestions
- Add per-epoch logging and a plot of training/test accuracy vs epoch.
- Save best model checkpoints (`torch.save`) to `models/` and add `models/` to `.gitignore` if you don't want them committed.
- Add unit tests for the `evaluate` function (small synthetic batches) to guarantee correctness.

---

# DL_scratch_models

Deep learning models implemented from scratch, primarily with NumPy, along with supporting notebooks and reference PDFs.

## Contents

### Notebooks
- **[CBOW 1.ipynb](CBOW%201.ipynb)** — Continuous Bag of Words (CBOW) word-embedding model built from scratch in NumPy. Covers:
  - Text cleaning and vocabulary construction
  - Context/target pair generation (window size 2)
  - Random embedding matrix of dimension 10
  - A manual forward pass (embedding lookup → linear layer → log-softmax)
  - A manual backward pass with cross-entropy gradients
  - SGD optimization loop (80 epochs, lr = 0.03)
  - Loss curve, prediction, and accuracy evaluation

### PDFs

- **[CBOW 1.pdf](CBOW%201.pdf)** — Rendered export of the CBOW notebook above.
  Flow: toy corpus → clean text → build vocab → (context, target) pairs → random embedding matrix → forward (embedding → linear → log-softmax) → backward (cross-entropy gradient) → SGD update → plot loss and predict center words.

- **[Autoencoder Credit Card Fraud Detection 1 - Jupyter Notebook.pdf](Autoencoder%20Credit%20Card%20Fraud%20Detection%201%20-%20Jupyter%20Notebook.pdf)** — Unsupervised fraud detection on the Kaggle credit-card dataset using a Keras autoencoder.
  Flow: load CSV → inspect class imbalance (284,315 normal vs 492 fraud) → scale `Time`/`Amount` → train/test split → train autoencoder on **normal transactions only** (encoder: 30 → 14 → 7 → 4 with dropout + L2; decoder mirrors it) → compile with MSE + Adam → fit with `ModelCheckpoint` + `EarlyStopping` → use reconstruction error as the anomaly score for fraud.

- **[train_test.pdf](train_test.pdf)** — MNIST digit classifier with a small Keras CNN.
  Flow: load MNIST from `.npz` → scale to [0,1] and add channel dim → one-hot labels → `Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dropout(0.5) → Dense(10, softmax)` → compile with categorical cross-entropy + SGD → train 10 epochs → evaluate (~97.7% test accuracy).

- **[DOC-20221123-WA0001..pdf](DOC-20221123-WA0001..pdf)** — MNIST classifier with a minimal fully-connected Keras model.
  Flow: load MNIST via `tf.keras.datasets` → inspect shape and sample image → normalize by `/255` → `Flatten → Dense(128, relu) → Dense(10, softmax)` → train and evaluate on the test set.
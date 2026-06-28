# MailGuard AI

An email spam detector powered by a Transformer neural network built entirely from scratch in PyTorch. The model reaches **97.0% accuracy** on a held-out test set of real emails.

No pre-trained models or external ML libraries (HuggingFace, etc.) were used. Every component — multi-head self-attention, sinusoidal positional encoding, layer normalization, feed-forward networks, and a padding-aware pooling layer — is implemented manually.

### [Live Demo](https://mailguard-ai.streamlit.app)

## How It Works

An email goes through the following pipeline:

1. **Text cleaning** — URLs and email addresses are replaced with placeholder tokens, non-alphabetic characters are dropped, and the text is lowercased.
2. **Tokenization** — Each word is mapped to an integer using a custom vocabulary of 30,000 tokens built **from the training split only**. The sequence is padded or truncated to 256 tokens.
3. **Transformer encoding** — The token sequence passes through an embedding layer, positional encoding, and 4 Transformer encoder blocks. Each block applies multi-head self-attention (8 heads) — masking out padding positions — followed by a GELU feed-forward network.
4. **Classification** — The encoder outputs are pooled with a **masked mean** (padding positions are ignored) and passed through a classification head that outputs a probability for each class: ham or spam.

## Model Architecture

```
Email text
  → Tokenization (custom 30K vocab)
  → Token Embedding (d=256) + Positional Encoding
  → Transformer Block × 4
      ├── Multi-Head Self-Attention (8 heads, d_k=32, padding-masked)
      ├── Residual Connection + Layer Norm (pre-LN)
      ├── Feed-Forward (256 → 1024 → 256, GELU)
      └── Residual Connection + Layer Norm (pre-LN)
  → Masked Mean Pooling (ignores padding)
  → Linear (256 → 256, GELU)
  → Linear (256 → 2)
  → Softmax → ham / spam
```

Total parameters: **10,905,858**

## Dataset

The model is trained on the [SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/) — real emails, with **both ham and spam drawn from the same source**. This matters: if ham and spam came from different datasets, the model could learn to recognize the corpus rather than the spam itself.

| Split | Emails |
|-------|--------|
| Ham | 4,106 |
| Spam | 1,654 |
| **Total** | **5,760** |

Exact duplicate emails are removed before splitting (286 dropped) to prevent train/test leakage. The data is split 70/15/15 into stratified train / validation / test sets (4,032 / 864 / 864). Ham-to-spam ratio is ~2.5:1; class weights in the loss offset the imbalance. Only real emails are used — no SMS or synthetic data.

## Training

Run on a GPU via the Colab notebook (a few minutes) or locally on CPU (~19 minutes for the committed model).

- Optimizer: AdamW (lr=3e-4, weight_decay=0.01)
- Learning-rate schedule: cosine annealing
- Loss: cross-entropy with inverse-frequency class weights
- Gradient clipping: max norm 1.0
- Batch size: 64
- **Early stopping**: best checkpoint selected by validation loss (patience 4)
- Seeded (`random` / `numpy` / `torch`) for reproducible runs

## Results

Evaluated on 864 unseen test emails:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Ham | 0.98 | 0.97 | 0.98 | 616 |
| Spam | 0.94 | 0.96 | 0.95 | 248 |

**Overall accuracy: 97.0%**

Confusion matrix:
```
              Predicted
              Ham    Spam
Actual Ham  [ 600     16 ]
Actual Spam [  10    238 ]
```

## Limitations

The SpamAssassin corpus dates from the early 2000s, and its ham is dominated by technical mailing-list and discussion email. The model therefore reflects that distribution: it is accurate on similar email but can misclassify modern transactional or casual messages (e.g. e-commerce shipping notifications) whose style resembles period promotional spam. Broadening to a more recent, diverse corpus is the natural next step.

## Project Structure

```
├── app.py                     # Streamlit UI (rendering only)
├── assets/
│   └── styles.css             # UI styling (kept out of logic)
├── src/
│   ├── config.py              # Single source of truth: hyperparameters, paths, seed
│   ├── transformer_model.py   # Transformer implementation from scratch
│   ├── preprocessing.py       # Text cleaning, vocabulary, encoding
│   ├── prepare_data.py        # SpamAssassin download + dedup
│   ├── train.py               # Training loop, early stopping, evaluation
│   └── inference.py           # Model loading + prediction (imported by the UI)
├── models/
│   ├── model.safetensors      # Trained weights (Git LFS)
│   ├── vocab.json             # Fitted vocabulary
│   ├── config.json            # Model hyperparameters
│   └── metrics.json           # Training history and test metrics
├── tests/                     # Forward shape, preprocessing round-trip, training step
├── notebooks/
│   └── train_colab.ipynb      # Colab notebook for GPU training
├── pyproject.toml             # ruff / mypy / pytest config
├── requirements.txt           # Runtime dependencies (pinned)
└── requirements-dev.txt       # Dev/tooling dependencies (pinned)
```

## Deployment

The app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud), connected directly to this GitHub repository. Model weights are stored via Git LFS. Any push to `main` triggers an automatic redeployment.

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Train the model

On Google Colab (GPU):

1. Open `notebooks/train_colab.ipynb` in [Google Colab](https://colab.research.google.com).
2. Set the runtime to GPU (Runtime → Change runtime type → GPU).
3. Run all cells — the notebook downloads the corpus, trains the model, and exports the artifacts.
4. Download `trained_model.zip` and extract it into `models/`.

Or locally, from the repository root:

```bash
python -m src.prepare_data   # download SpamAssassin into data/processed/
python -m src.train          # train and write artifacts into models/
```

## Development

```bash
pip install -r requirements-dev.txt
ruff format . && ruff check . && mypy src && pytest
```

## Tech Stack

- **PyTorch** — tensors, autograd, model building
- **safetensors** — safe model-weight serialization
- **Streamlit** — web interface and deployment
- **scikit-learn** — evaluation metrics (precision, recall, F1, confusion matrix)
- **ruff / mypy / pytest** — linting, type checking, tests

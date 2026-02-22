# MailGuard AI

An AI-powered email spam detector using a Transformer neural network built entirely from scratch with PyTorch. The model achieves **97.7% accuracy** on a test set of real emails.

No pre-trained models or external ML libraries (HuggingFace, etc.) were used. Every component — multi-head self-attention, positional encoding, layer normalization, feed-forward networks — is implemented manually.

### 🔗 [Live Demo](https://mailguard-ai.streamlit.app)

## How It Works

An email goes through the following pipeline:

1. **Text cleaning** — URLs, email addresses, and special characters are removed. The text is lowercased and tokenized.
2. **Tokenization** — Each word is mapped to an integer using a custom vocabulary of 30,000 tokens built from the training data. The sequence is padded or truncated to 256 tokens.
3. **Transformer encoding** — The token sequence passes through an embedding layer, positional encoding, and 4 Transformer encoder blocks. Each block applies multi-head self-attention (8 heads) followed by a feed-forward network with GELU activation.
4. **Classification** — The encoder outputs are averaged (global average pooling) and passed through a classification head that outputs a probability for each class: ham or spam.

## Model Architecture

```
Email text
  → Tokenization (custom 30K vocab)
  → Token Embedding (d=256) + Positional Encoding
  → Transformer Block × 4
      ├── Multi-Head Self-Attention (8 heads, d_k=32)
      ├── Residual Connection + Layer Norm
      ├── Feed-Forward (256 → 1024 → 256, GELU)
      └── Residual Connection + Layer Norm
  → Global Average Pooling
  → Linear (256 → 256, GELU)
  → Linear (256 → 2)
  → Softmax → ham / spam
```

Total parameters: **10,905,858**

## Training

The model was trained on Google Colab (T4 GPU) for 15 epochs (~5 minutes).

Training configuration:
- Optimizer: AdamW (lr=3e-4, weight_decay=0.01)
- Learning rate schedule: cosine annealing
- Loss: cross-entropy with class weights (to handle 3:1 ham/spam imbalance)
- Gradient clipping: max norm 1.0
- Batch size: 64

## Dataset

Two public email corpora were combined:

| Source | Description | Samples |
|--------|-------------|---------|
| [SpamAssassin](https://spamassassin.apache.org/old/publiccorpus/) | Public spam/ham email corpus | 6,046 |
| [Enron](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) | Corporate email dataset (ham only, subsampled) | 1,538 |
| **Total** | | **7,584** |

The dataset is split 70/15/15 into train, validation, and test sets. Ham-to-spam ratio is 3:1.

Only real emails were used — no SMS messages or synthetic data.

## Results

Evaluated on 1,138 unseen test emails:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Ham | 0.99 | 0.98 | 0.98 | 847 |
| Spam | 0.94 | 0.97 | 0.96 | 291 |

**Overall accuracy: 97.7%**

Confusion matrix:
```
              Predicted
              Ham    Spam
Actual Ham  [ 830     17 ]
Actual Spam [   9    282 ]
```

## Project Structure

```
├── app.py                     # Streamlit web application
├── src/
│   ├── transformer_model.py   # Transformer implementation from scratch
│   ├── preprocessing.py       # Text cleaning, tokenization, vocabulary
│   ├── prepare_data.py        # Dataset download and preparation
│   └── train.py               # Training loop with evaluation
├── models/
│   ├── transformer_best.pth   # Trained model weights (Git LFS)
│   ├── preprocessor.pkl       # Fitted tokenizer (Git LFS)
│   ├── config.json            # Model hyperparameters
│   └── metrics.json           # Training history and test metrics
├── notebooks/
│   └── train_colab.ipynb      # Google Colab notebook for GPU training
└── requirements.txt
```

## Deployment

The app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud) and connected directly to this GitHub repository. Model weights are stored via Git LFS. Any push to `main` triggers an automatic redeployment.

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Train the model

1. Open `notebooks/train_colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to GPU (Runtime → Change runtime type → T4 GPU)
3. Run all cells — the notebook downloads the datasets, trains the model, and exports the weights
4. Download the `trained_model.zip` file and extract it into the `models/` directory

## Tech Stack

- **PyTorch** — tensor operations, autograd, model building
- **Streamlit** — web interface and deployment
- **scikit-learn** — evaluation metrics (precision, recall, F1, confusion matrix)
- **Google Colab** — GPU training environment

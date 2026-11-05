# Multiclass Sentiment Classifier

A text sentiment classifier built with a TF-IDF vectorizer and logistic
regression. The training script evaluates the model on a stratified holdout set
and saves the trained model and evaluation artifacts under `outputs/`.

## Requirements

- Python
- The packages listed in `requirements.txt`
- A local dataset at `dataset/sentiment.csv` (the dataset is intentionally
  excluded from Git)

The CSV must contain `text` and `sentiment` columns. Other columns are ignored
by the training script. Each sentiment class must have at least two valid text
rows.

## Setup (Windows PowerShell)

From the project root, create a virtual environment and install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place your CSV at `dataset\sentiment.csv`. Then train and evaluate the model:

```powershell
.\.venv\Scripts\python.exe src\train_model.py
```

Start interactive predictions:

```powershell
.\.venv\Scripts\python.exe src\predict.py
```

Enter a sentence when prompted. Type `exit` to quit.

## Generated files

Training creates or overwrites these local files in `outputs/`:

- `sentiment_model.joblib` — trained model used by the prediction script
- `metrics.json` — accuracy, macro-averaged metrics, and split sizes
- `classification_report.txt` — per-class classification report
- `confusion_matrix.png` — confusion matrix visualization

The dataset and generated artifacts are ignored by Git; `outputs/.gitkeep` is
kept so the output directory remains present in a checkout.

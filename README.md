# Learn
Learning Git and Github

# ASSIGNMENT 10
# Amazon Alexa Customer Review Sentiment Analysis

This project trains sentiment models on Amazon Alexa customer reviews and compares traditional NLP pipelines to a fine-tuned BERT classifier.

## Project structure
- `notebooks/Assignment10_Sentiment.ipynb` — main Colab notebook
- `figures/` — exported plots (confusion matrices, ROC curves, PCA)
- `models/` — saved BERT model and tokenizer (optional, git-ignored if large)
- `report.docx` and `report.pdf` — two-page write-up
- `requirements.txt` — pinned versions

## Dataset
Amazon Alexa reviews TSV with columns: `rating`, `date`, `variation`, `verified_reviews`, `feedback`.
Target is `feedback` where 1 is positive and 0 is negative.

## Quick start (Colab)
1. Open `notebooks/Assignment10_Sentiment.ipynb` in Google Colab.
2. Run all cells to install dependencies, load data, train models, and generate figures.
3. At the end, the notebook prints evaluation metrics and shows plots.

## Methods
- Preprocessing: drop `date` and `rating`, fill missing text, one-hot encode `variation`.
- Features: BoW and TF-IDF with 2,000 features.
- Models: Logistic Regression, Random Forest with GridSearchCV.
- LLM: BERT base uncased fine-tuned with Hugging Face Trainer (max length 128).
- Metrics: accuracy, precision, recall, F1, ROC-AUC.

## Results summary
- Logistic Regression (TF-IDF): Accuracy ~0.925, low negative recall.
- Random Forest (TF-IDF + dummies): Accuracy ~0.929, low negative recall.
- **BERT**: Accuracy ~0.959, Precision ~0.962, Recall ~0.995, F1 ~0.978, ROC-AUC ~0.978.

## Deployment sketch
- Package tokenizer and model.
- FastAPI endpoint `/predict` takes raw text and returns `{"label": 0|1, "prob": float}`.
- Add monitoring and threshold tuning for the negative class.

## Reproducibility
- Random seeds set to 42 where applicable.
- See notebook for exact hyperparameters and plots.

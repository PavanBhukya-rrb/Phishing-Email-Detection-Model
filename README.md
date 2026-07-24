# Phishing Email Detection Model

A machine learning model built with **Scikit-learn** that classifies emails as **"Phishing"** or **"Safe"** by analyzing textual content, URL patterns, and structural features.

## Features
- Trains on a dataset of phishing and legitimate emails
- Extracts and analyzes email features (URLs, suspicious keywords, domain patterns, etc.)
- Classifies emails as "Phishing" or "Safe"
- Displays accuracy, confusion matrix, and classification report
- Visualizes model performance with matplotlib/seaborn

## How It Works
1. **Dataset Generation** - A realistic synthetic dataset of 1000+ phishing and safe emails
2. **Feature Extraction** - Extracts 10+ features including URL count, keyword frequency, domain patterns
3. **Model Training** - Uses Random Forest Classifier with train/test split (80/20)
4. **Evaluation** - Reports accuracy, precision, recall, F1-score, and confusion matrix

## Sample Classification
```
Email: "Urgent! Your account has been compromised. Click here to verify: http://bit.ly/2xPhish"
 → PREDICTION: 🔴 PHISHINGS

Email: "Meeting tomorrow at 3pm. Please bring the quarterly report."
 → PREDICTION: 🟢 SAFE
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Output
- Console output showing accuracy and classification metrics
- Confusion matrix visualization (saved as `confusion_matrix.png`)
- Sample email classifications with predictions

## Dependencies
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
S
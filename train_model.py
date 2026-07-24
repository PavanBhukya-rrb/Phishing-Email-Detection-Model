"""
Model training module.
Trains a Random Forest classifier (and alternative models) on extracted email features.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix


def train_tfidf_vectorizer(texts, max_features=500):
    """
    Train a TF-IDF vectorizer on email body text.

    Args:
        texts: List of email text strings
        max_features: Maximum number of features for TF-IDF

    Returns:
        Trained TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=2,
    )
    vectorizer.fit(texts)
    return vectorizer


def split_data(feature_df, test_size=0.2, random_state=42):
    """
    Split the feature DataFrame into training and testing sets.

    Args:
        feature_df: DataFrame with features and 'label' column
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    feature_cols = [col for col in feature_df.columns if col != "label"]
    X = feature_df[feature_cols].values
    y = feature_df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier.

    Args:
        X_train: Training feature matrix
        y_train: Training labels
        n_estimators: Number of trees in the forest
        random_state: Random seed

    Returns:
        Trained RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train, random_state=42):
    """
    Train a Logistic Regression classifier.

    Args:
        X_train: Training feature matrix
        y_train: Training labels
        random_state: Random seed

    Returns:
        Trained LogisticRegression
    """
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model and print metrics.

    Args:
        model: Trained classifier
        X_test: Test feature matrix
        y_test: Test labels
        model_name: Name to display in output

    Returns:
        Dictionary with accuracy, confusion matrix, and classification report
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n{'='*60}")
    print(f"{' ' * 15}{model_name} Evaluation Results")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": y_pred,
    }

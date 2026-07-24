"""
Phishing Email Detection Model
================================
End-to-end pipeline:
1. Generate synthetic email dataset
2. Extract features (URLs, keywords, structure)
3. Train Random Forest & Logistic Regression classifiers
4. Evaluate with accuracy, confusion matrix, and classification report
5. Visualize results
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Local imports
from data.generate_dataset import generate_dataset
from src.feature_extractor import extract_features_dataframe, get_feature_names
from src.train_model import (
    train_random_forest,
    train_logistic_regression,
    evaluate_model,
)
from src.evaluate import (
    plot_confusion_matrix,
    plot_feature_importance,
    print_sample_classifications,
)


def main():
    print("=" * 70)
    print("   PHISHING EMAIL DETECTION MODEL")
    print("   Machine Learning-based Email Classifier using Scikit-learn")
    print("=" * 70)

    # =========================================================
    # STEP 1: Generate Dataset
    # =========================================================
    print("\n[1/5] Generating synthetic email dataset...")
    dataset_path = generate_dataset(num_samples=1200, phishing_ratio=0.5)

    df = pd.read_csv(dataset_path)
    print(f"   Loaded {len(df)} emails from dataset")
    print(f"   Labels distribution:\n{df['label'].value_counts()}")

    # =========================================================
    # STEP 2: Extract Features
    # =========================================================
    print("\n[2/5] Extracting features from emails...")
    feature_df = extract_features_dataframe(df)
    feature_names = get_feature_names()
    print(f"   Extracted {len(feature_names)} features: {feature_names}")
    print(f"   Feature DataFrame shape: {feature_df.shape}")
    print("\n   Sample feature values (first 5 rows):")
    print(feature_df.head())

    # =========================================================
    # STEP 3: Train/Test Split (with index tracking)
    # =========================================================
    print("\n[3/5] Splitting data into training and testing sets...")

    X = feature_df[feature_names].values
    y = feature_df["label"].values
    indices = np.arange(len(feature_df))

    # Use 6-output unpacking: X_train, X_test, y_train, y_test, idx_train, idx_test
    split_results = train_test_split(
        X, y, indices, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test, y_train, y_test, idx_train, idx_test = split_results

    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples:  {len(X_test)}")

    # =========================================================
    # STEP 4: Train Models
    # =========================================================
    print("\n[4/5] Training classifiers...")

    # Train Random Forest
    print("\n   --- Random Forest Classifier ---")
    rf_model = train_random_forest(X_train, y_train, n_estimators=200)
    rf_results = evaluate_model(rf_model, X_test, y_test, model_name="Random Forest")

    # Train Logistic Regression
    print("\n   --- Logistic Regression Classifier ---")
    lr_model = train_logistic_regression(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_test, y_test, model_name="Logistic Regression")

    # =========================================================
    # STEP 5: Visualize Results
    # =========================================================
    print("\n[5/5] Generating visualizations...")

    # Confusion Matrix for Random Forest
    plot_confusion_matrix(
        y_test,
        rf_results["predictions"],
        labels=["Phishing", "Safe"],
        save_path="confusion_matrix.png",
    )

    # Feature Importance for Random Forest
    plot_feature_importance(
        rf_model,
        feature_names,
        top_n=len(feature_names),
        save_path="feature_importance.png",
    )

    # Confusion Matrix for Logistic Regression
    plot_confusion_matrix(
        y_test,
        lr_results["predictions"],
        labels=["Phishing", "Safe"],
        save_path="confusion_matrix_lr.png",
    )

    # =========================================================
    # Sample Classifications (with correct email mapping)
    # =========================================================
    print("\n   Displaying sample classifications...")
    num_samples = min(5, len(X_test))
    test_indices = np.random.choice(len(X_test), size=num_samples, replace=False)
    X_test_samples = X_test[test_indices]
    y_test_samples = y_test[test_indices]
    # Map back to original dataframe using idx_test
    original_indices = idx_test[test_indices]
    sample_texts = df.iloc[original_indices]["email"].values

    print_sample_classifications(rf_model, X_test_samples, sample_texts, y_test_samples)

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 70)
    print("   SUMMARY")
    print("=" * 70)
    print(f"   Best Model: Random Forest Classifier")
    print(f"   Accuracy:   {rf_results['accuracy']:.4f} ({rf_results['accuracy'] * 100:.2f}%)")
    print(f"   Features Used: {len(feature_names)}")
    print(f"   Training Samples: {len(X_train)}")
    print(f"   Testing Samples:  {len(X_test)}")
    print(f"\n   Output Files Generated:")
    print(f"   - confusion_matrix.png     (Random Forest)")
    print(f"   - confusion_matrix_lr.png   (Logistic Regression)")
    print(f"   - feature_importance.png")
    print(f"   - data/dataset.csv         (Generated Dataset)")
    print("=" * 70)

    # Return the best model and results
    return rf_model, rf_results


if __name__ == "__main__":
    main()

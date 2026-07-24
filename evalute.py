"""
Evaluation module.
Visualizes model performance with confusion matrix and feature importance.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path="confusion_matrix.png"):
    """
    Plot and save a confusion matrix visualization.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of label names (default: ["Phishing", "Safe"])
        save_path: Path to save the plot image
    """
    if labels is None:
        labels = ["Phishing", "Safe"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")

    # Plot using seaborn for better aesthetics
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=1,
        linecolor="gray",
        cbar=True,
    )

    plt.title("Confusion Matrix - Phishing Email Detection", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    # Add text annotations for TP, TN, FP, FN
    # TP: top-left, TN: bottom-right, FP: top-right, FN: bottom-left
    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]

    # Calculate percentages
    total = cm.sum()
    annotations = [
        f"TP: {tp}\n({tp/total*100:.1f}%)",
        f"FN: {fn}\n({fn/total*100:.1f}%)",
        f"FP: {fp}\n({fp/total*100:.1f}%)",
        f"TN: {tn}\n({tn/total*100:.1f}%)",
    ]

    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5, i + 0.85, annotations[i * 2 + j],
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix saved to: {os.path.abspath(save_path)}")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=10, save_path="feature_importance.png"):
    """
    Plot and save feature importance from a Random Forest model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
        save_path: Path to save the plot image
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        names = [feature_names[i] for i in indices]
        values = importances[indices]

        ax = sns.barplot(x=values, y=names, palette="viridis")
        plt.title(f"Top {top_n} Feature Importances", fontsize=16, fontweight="bold")
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Features", fontsize=12)

        # Add value labels
        for i, v in enumerate(values):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Feature importance plot saved to: {os.path.abspath(save_path)}")
        plt.close()
    else:
        print("Model doesn't support feature importance visualization.")


def print_sample_classifications(model, X_test_sample, sample_texts, actual_labels):
    """
    Print sample email classifications with predictions.

    Args:
        model: Trained classifier
        X_test_sample: Feature matrix for sample emails
        sample_texts: List of original email text samples
        actual_labels: Actual labels for the samples
    """
    predictions = model.predict(X_test_sample)
    probabilities = model.predict_proba(X_test_sample)
    class_labels = model.classes_

    print("\n" + "=" * 80)
    print("Sample Email Classifications")
    print("=" * 80)

    for i, (email_text, pred, actual) in enumerate(zip(sample_texts, predictions, actual_labels)):
        # Extract subject line for display
        lines = email_text.split("\n")
        subject = ""
        for line in lines:
            if line.startswith("Subject:"):
                subject = line.replace("Subject:", "").strip()
                break
        if not subject:
            subject = email_text[:80] + "..."

        # Get confidence
        pred_idx = list(class_labels).index(pred)
        confidence = probabilities[i][pred_idx]

        pred_icon = "🔴" if pred == "Phishing" else "🟢"
        actual_icon = "🔴" if actual == "Phishing" else "🟢"
        correct = "✅" if pred == actual else "❌"

        print(f"\n{correct} Sample {i + 1}:")
        print(f"   Subject: {subject[:70]}")
        print(f"   Predicted: {pred_icon} {pred} (confidence: {confidence:.2%})")
        print(f"   Actual:    {actual_icon} {actual}")

"""
Feature extractor module.
Extracts numerical and categorical features from email text for machine learning.
"""

import pandas as pd
from src.utils import (
    extract_urls,
    has_ip_address_url,
    has_suspicious_domain,
    is_shortened_url,
    count_suspicious_keywords,
    count_exclamation_marks,
    has_mismatched_sender,
    get_email_length,
    get_word_count,
)


def extract_features_from_email(email_text):
    """
    Extract a feature vector from a single email text.

    Args:
        email_text: Raw email text content

    Returns:
        Dictionary of extracted feature name -> value
    """
    urls = extract_urls(email_text)
    num_urls = len(urls)

    # URL-based features
    ip_url_count = sum(1 for url in urls if has_ip_address_url(url))
    suspicious_domain_count = sum(1 for url in urls if has_suspicious_domain(url))
    shortened_url_count = sum(1 for url in urls if is_shortened_url(url))

    # Text-based features
    keyword_count = count_suspicious_keywords(email_text)
    exclamation_count = count_exclamation_marks(email_text)
    sender_mismatch = has_mismatched_sender(email_text)
    email_length = get_email_length(email_text)
    word_count = get_word_count(email_text)

    features = {
        "num_urls": num_urls,
        "ip_url_count": ip_url_count,
        "suspicious_domain_count": suspicious_domain_count,
        "shortened_url_count": shortened_url_count,
        "keyword_count": keyword_count,
        "exclamation_count": exclamation_count,
        "sender_mismatch": sender_mismatch,
        "email_length": email_length,
        "word_count": word_count,
    }

    return features


def extract_features_dataframe(df):
    """
    Extract features for all emails in a DataFrame.

    Args:
        df: DataFrame with an 'email' column containing raw email text

    Returns:
        DataFrame with extracted features (plus original label if present)
    """
    feature_rows = []
    for idx, row in df.iterrows():
        features = extract_features_from_email(row["email"])
        if "label" in row:
            features["label"] = row["label"]
        feature_rows.append(features)

    feature_df = pd.DataFrame(feature_rows)
    return feature_df


def get_feature_names():
    """
    Get the list of feature names used by this extractor.

    Returns:
        List of feature column names
    """
    return [
        "num_urls",
        "ip_url_count",
        "suspicious_domain_count",
        "shortened_url_count",
        "keyword_count",
        "exclamation_count",
        "sender_mismatch",
        "email_length",
        "word_count",
    ]

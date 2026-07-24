"""
Utility functions for text preprocessing and URL analysis.
"""

import re
from urllib.parse import urlparse


def extract_urls(text):
    """
    Extract all URLs from a given text using regex.

    Args:
        text: Input text string

    Returns:
        List of URLs found in the text
    """
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|'
        r'(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.findall(text)


def has_ip_address_url(url):
    """
    Check if a URL uses an IP address instead of a domain name.

    Args:
        url: URL string

    Returns:
        True if the URL contains an IP address, False otherwise
    """
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
    return bool(ip_pattern.match(hostname))


def has_suspicious_domain(url):
    """
    Check if a URL uses a suspicious or uncommon top-level domain.

    Args:
        url: URL string

    Returns:
        True if domain is suspicious, False otherwise
    """
    suspicious_tlds = {'.xyz', '.top', '.online', '.club', '.work', '.bid', '.loan', '.download', '.review'}
    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    # Check for suspicious TLDs
    for tld in suspicious_tlds:
        if hostname.endswith(tld):
            return True

    # Check for domains that mimic legitimate ones (e.g., paypa1.com, amaz0n.com)
    suspicious_keywords = ['secure', 'login', 'verify', 'account', 'update', 'confirm', 'bank', 'paypal', 'amazon', 'netflix']
    for keyword in suspicious_keywords:
        if keyword in hostname and not hostname.endswith(f'{keyword}.com'):
            return True

    return False


def is_shortened_url(url):
    """
    Check if a URL is a shortened URL (bit.ly, tinyurl, etc.).

    Args:
        url: URL string

    Returns:
        True if URL is shortened, False otherwise
    """
    shortening_services = [
        'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly',
        'is.gd', 'buff.ly', 'shorturl.at', 'tiny.cc', 'tr.im'
    ]
    for service in shortening_services:
        if service in url:
            return True
    return False


def count_suspicious_keywords(text):
    """
    Count the number of suspicious keywords in the email text.

    These keywords are commonly found in phishing emails.

    Args:
        text: Email text string

    Returns:
        Count of suspicious keywords found
    """
    text_lower = text.lower()
    suspicious_keywords = [
        'urgent', 'immediately', 'account suspended', 'verify your account',
        'click here', 'login', 'password', 'confirm', 'bank', 'credit card',
        'social security', 'limited', 'expire', 'suspicious activity',
        'unauthorized', 'compromised', 'lottery', 'winner', 'prize',
        'free', 'congratulations', 'claim', 'million dollars', 'inheritance',
        'wire transfer', 'money', 'act now', 'security alert', 'update your',
        'deactivated', 'terminated', 'legal action', 'fbi', 'irs',
    ]
    count = 0
    for keyword in suspicious_keywords:
        if keyword in text_lower:
            count += 1
    return count


def count_exclamation_marks(text):
    """
    Count the number of exclamation marks in the text.
    Phishing emails often have excessive exclamation marks.

    Args:
        text: Email text string

    Returns:
        Number of exclamation marks
    """
    return text.count('!')


def has_mismatched_sender(text):
    """
    Check if the email mentions a sender domain mismatch (common in phishing).
    Simplified check - looks for patterns like "from Bank" with different reply-to domains.

    Args:
        text: Email text string

    Returns:
        1 if mismatch pattern detected, 0 otherwise
    """
    # Simplified heuristic: Look for @ symbol patterns that suggest mismatched domains
    at_patterns = re.findall(r'\S+@\S+', text)
    if len(at_patterns) > 0:
        # Check if any email domains look suspicious (non-standard domains)
        suspicious_domains = ['@gmail.com', '@yahoo.com', '@hotmail.com']  # personal email domains used for phishing
        for email_addr in at_patterns:
            for dom in suspicious_domains:
                if dom in email_addr:
                    return 1
    return 0


def get_email_length(text):
    """
    Get the length of the email text.

    Args:
        text: Email text string

    Returns:
        Length of the text in characters
    """
    return len(text)


def get_word_count(text):
    """
    Get the word count of the email text.

    Args:
        text: Email text string

    Returns:
        Number of words
    """
    return len(text.split())

"""
Generate a synthetic dataset of phishing and legitimate (safe) emails.
Creates a CSV file with email content and labels for training/evaluation.
"""

import csv
import os
import random

# Seed for reproducibility
random.seed(42)

# --- Phishing email templates ---
PHISHING_SUBJECTS = [
    "Urgent: Your account has been compromised!",
    "Verify your account immediately!",
    "Security Alert: Unauthorized login attempt!!!",
    "Your account has been suspended",
    "Confirm Your Account Information NOW!",
    "PayPal: Your account has been limited!",
    "IRS: You have a tax refund pending!!!",
    "FBI: Cyber attack detected on your system!",
    "Your Netflix subscription has expired!",
    "Amazon: Suspicious purchase detected!!!",
    "WINNER: You won a lottery prize!!!",
    "Bank of America: Verify your identity!",
    "Your email password will expire tomorrow!!!",
    "Important: Update your payment method!",
    "FREE iPhone 15 Pro - Claim now!!!",
]

PHISHING_BODIES = [
    "Dear user,\n\nWe detected suspicious activity on your account. "
    "Please click the link below to verify your identity immediately:\n"
    "http://bit.ly/2xVerify{}".format(random.randint(1000, 9999))
    + "\n\nFailure to verify within 24 hours will result in account suspension.\n\n"
    "Sincerely,\nAccount Security Team",

    "Hello valued customer,\n\nYour account has been compromised by a third party. "
    "To protect your information, please login and change your password immediately:\n"
    "http://192.168.1.1/login/{}".format(random.randint(1000, 9999))
    + "\n\nDo not ignore this message! \n\nSupport Team",

    "Congratulations!!! You have been selected as the winner of "
    "our annual lottery!! You have won $5,000,000 USD!!\n"
    "Click here to claim your prize: http://tinyurl.com/win-{}".format(random.randint(10000, 99999))
    + "\n\nThis is a limited time offer. Act now!\n\n"
    "Lottery Commission",

    "URGENT: Your account will be deactivated!\n\n"
    "We notice that your account information is out of date. "
    "Please update your records within 48 hours or your account will be terminated.\n"
    "Update here: http://fakebank-login.com/verify/{}".format(random.randint(1000, 9999))
    + "\n\nThank you,\nAccount Management",

    "Dear customer,\n\nWe need you to confirm your email address "
    "to continue using our services. Click the link below:\n"
    "http://secure-login.xyz/confirm?id={}".format(random.randint(10000, 99999))
    + "\n\nIf you do not confirm, your account will be closed.\n\n"
    "Support Team",

    "FBI Cyber Security Division\n\n"
    "We have detected a cyber attack originating from your IP address. "
    "You must download and run the attached security tool immediately!\n"
    "http://malware-download.xyz/security-tool.exe\n\n"
    "Failure to comply may result in legal action.\n\n"
    "Special Agent J. Mueller",

    "Your Netflix account has been suspended due to payment failure.\n"
    "Please update your billing information here:\n"
    "http://netflix-verify.{}".format(random.choice(["com", "org", "net"]))
    + "/update\n\nWe apologize for the inconvenience.\n"
    "Netflix Billing Team",

    "You have a new secure message from your bank.\n"
    "Please login to view: http://bank-secure-message.xyz\n"
    "This message will expire in 24 hours.\n\n"
    "Online Banking",

    "IMPORTANT: Your domain is expiring!\n"
    "Renew now to avoid losing {}beautiful-site{}.com!\n".format("your-", random.choice(["", "online", "shop"]))
    + "http://domain-renew.{}".format(random.choice(["xyz", "top", "online"]))
    + "/renew\n\nDomain Registrar Services",

    "Dear Sir/Madam,\n\nI am a representative of a foreign bank. "
    "I have $10,500,000 USD that I need to transfer out of my country. "
    "I need your assistance and bank account to complete this transfer. "
    "You will receive 30% of the total amount.\n\n"
    "Please contact me at: prince{}@{}mail.com\n".format(random.randint(100, 999), random.choice(["g", "y", "hot"]))
    + "\nThis is a confidential business proposal.\n\n"
    "Best regards,\nDr. {} {}".format(random.choice(["James", "Michael", "David"]), random.choice(["Smith", "Brown", "Johnson"])),
]

# --- Safe (legitimate) email templates ---
SAFE_SUBJECTS = [
    "Meeting rescheduled to Friday",
    "Your weekly report is ready",
    "Lunch tomorrow?",
    "Invoice #{} attached".format(random.randint(10000, 99999)),
    "Project update: Q3 milestones",
    "Re: Budget approval request",
    "Team building event next week",
    "Your Amazon order has shipped",
    "Password changed successfully",
    "Monthly newsletter - October edition",
    "Your appointment is confirmed",
    "Weekly standup notes",
    "New comment on your pull request",
    "Thank you for your submission",
    "Office holiday schedule",
]

SAFE_BODIES = [
    "Hi team,\n\nJust a reminder that our meeting has been moved to Friday "
    "at 2:00 PM to accommodate everyone's schedule. Please let me know "
    "if you have any conflicts.\n\nBest,\nJohn",

    "Hello,\n\nAttached is the weekly sales report for the period ending "
    "Friday. Please review and provide your feedback by EOD Wednesday.\n\n"
    "Let me know if you have any questions.\n\nThanks,\nSarah",

    "Hey!\n\nSome of us are going to the Italian place on Main Street "
    "for lunch tomorrow at 12:30. Let me know if you want to join!\n\n"
    "Cheers,\nMike",

    "Dear customer,\n\nYour order #{} has been shipped and is on its way.\n".format(random.randint(100000, 999999))
    + "Estimated delivery: 3-5 business days.\n\n"
    "You can track your package using the link provided in your account.\n\n"
    "Thank you for shopping with us!\nAmazon Customer Service",

    "Hi {},\n\n".format(random.choice(["Alice", "Bob", "Carol", "Dave"]))
    + "Here are the minutes from today's standup:\n"
    "- Completed user authentication module\n"
    "- Working on API integration\n"
    "- No blockers\n\n"
    "Please add anything I missed.\n\nThanks,\nScrum Master",

    "Your password has been changed successfully.\n"
    "If you did not make this change, please contact support immediately.\n\n"
    "This is an automated message, please do not reply.\n\n"
    "Security Team",

    "Dear team,\n\nPlease find attached the quarterly budget report "
    "for your review. We will discuss this in the next board meeting.\n\n"
    "Please submit any questions in advance.\n\n"
    "Regards,\nFinance Department",

    "Hello {},\n\n".format(random.choice(["James", "Emma", "Oliver", "Sophia"]))
    + "Your appointment with Dr. {} is confirmed for ".format(random.choice(["Smith", "Johnson", "Williams"]))
    + "{} at {}.".format(
        random.choice(["Monday", "Tuesday", "Wednesday", "Thursday"]),
        random.choice(["9:00 AM", "10:30 AM", "2:00 PM", "3:30 PM"])
    ) + "\n\nPlease arrive 15 minutes early.\n\n"
    "Thank you,\nFront Desk",

    "Hi everyone,\n\nJust a quick update on the project:\n"
    "- Frontend development is 80% complete\n"
    "- Backend APIs are being tested\n"
    "- We're on track for the release date\n\n"
    "Let me know if you have any concerns.\n\n"
    "Best,\nProject Manager",

    "Your monthly subscription has been renewed.\n"
    "Next billing date: {} {}, {}.\n\n".format(
        random.choice(["January", "March", "June", "September"]),
        random.randint(1, 28),
        random.randint(2025, 2026)
    ) + "If you have any questions, please visit our help center.\n\n"
    "Thank you for being a valued customer.\n\n"
    "Subscription Team",
]


def generate_phishing_email():
    """Generate a synthetic phishing email."""
    subject = random.choice(PHISHING_SUBJECTS)
    body = random.choice(PHISHING_BODIES)
    full_text = f"Subject: {subject}\n\n{body}"
    return full_text


def generate_safe_email():
    """Generate a synthetic legitimate email."""
    subject = random.choice(SAFE_SUBJECTS)
    body = random.choice(SAFE_BODIES)
    full_text = f"Subject: {subject}\n\n{body}"
    return full_text


def generate_dataset(num_samples=1000, phishing_ratio=0.5, output_path="data/dataset.csv"):
    """
    Generate a synthetic email dataset and save to CSV.

    Args:
        num_samples: Total number of samples to generate
        phishing_ratio: Proportion of phishing emails (0.0 - 1.0)
        output_path: Path to save the CSV file
    """
    num_phishing = int(num_samples * phishing_ratio)
    num_safe = num_samples - num_phishing

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["email", "label"])

        for _ in range(num_phishing):
            email = generate_phishing_email()
            writer.writerow([email, "Phishing"])

        for _ in range(num_safe):
            email = generate_safe_email()
            writer.writerow([email, "Safe"])

    print(f"Dataset generated: {num_samples} emails ({num_phishing} phishing, {num_safe} safe)")
    print(f"Saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    generate_dataset()

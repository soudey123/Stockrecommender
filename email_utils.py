import os, smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List

DEFAULT_SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
DEFAULT_SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))

def send_email(subject: str, body_html: str, recipients: List[str]) -> None:
    """
    Sends an HTML email using environment variables:
      EMAIL_USER, EMAIL_PASS, [optional SMTP_SERVER, SMTP_PORT]
    """
    sender = os.environ.get("EMAIL_USER")
    password = os.environ.get("EMAIL_PASS")
    if not sender or not password or not recipients:
        raise RuntimeError("Missing EMAIL_USER, EMAIL_PASS, or recipients.")
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    part = MIMEText(body_html, "html")
    msg.attach(part)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(DEFAULT_SMTP_SERVER, DEFAULT_SMTP_PORT, context=context) as server:
        server.login(sender, password)
        server.sendmail(sender, recipients, msg.as_string())

# save as test_email.py and run: python test_email.py
import smtplib
from email.mime.text import MIMEText

SMTP_EMAIL    = "tfitsquaddron@gmail.com"
SMTP_PASSWORD = "ukekrjniviedmwgb"    # your actual app password
PARENT_EMAIL  = "rachfpatil@gmail.com"

msg = MIMEText("SafeGuard test email — SMTP is working!")
msg["Subject"] = "SafeGuard Test"
msg["From"]    = SMTP_EMAIL
msg["To"]      = PARENT_EMAIL

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
    s.login(SMTP_EMAIL, SMTP_PASSWORD)
    s.sendmail(SMTP_EMAIL, PARENT_EMAIL, msg.as_string())
    print("Email sent successfully!")
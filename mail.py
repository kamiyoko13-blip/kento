import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()

def send_notification(subject: str, body: str) -> bool:
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    to_email = os.getenv("TO_EMAIL")

    message = MIMEMultipart()
    message["From"] = "ninimama27@yokopy-bot2025.conoha.email"
    message["To"] = "kamiyoko13@gmail.com"
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, message.as_string())
        print("✅ メール送信成功！")
        return True
    except Exception as e:
        print(f"❌ メール送信失敗: {e}")
        return False

# 通知を送りたいタイミングで
subject = "✅ 通知ボット：処理完了"
body = "本日の処理が正常に完了しました。"
send_notification(subject, body)



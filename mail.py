<<<<<<< HEAD
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()


def send_notification(subject: str, body: str, dry_run: bool = False) -> bool:
    """Send a notification email.

    This function is safe to import (it will not run on import). Call it
    explicitly or run the module as a script. If DRY_RUN env is set to
    1/true, the function will only simulate sending.
    """
    # allow env override
    env_dry = os.getenv("DRY_RUN", "0").lower() in ("1", "true")
    dry_run = dry_run or env_dry

    # 件名が空だった場合の保険
    if not subject:
        subject = "通知ボット"

    # 件名に応じた本文付加
    if "資金警告" in subject:
        body += "\n⚠️ 追加資金の検討をおすすめします。"
    elif "処理完了" in subject:
        body += "\n✅ 本日のBot処理は正常に終了しました。"

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    to_email = os.getenv("TO_EMAIL")

    # basic validation
    if dry_run:
        print(f"[DRY_RUN] would send mail to: {to_email} via {smtp_host}:{smtp_port}")
        print(f"[DRY_RUN] subject: {subject}\nbody:\n{body}")
        return True

    if not smtp_host or not smtp_user or not smtp_pass or not to_email:
        print("ERROR: SMTP settings incomplete. Check SMTP_HOST/SMTP_USER/SMTP_PASS/TO_EMAIL environment variables.")
        return False

    message = MIMEMultipart()
    message["From"] = smtp_user
    message["To"] = to_email
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


if __name__ == "__main__":
    # スクリプトとして実行する場合のテスト送信（本番では env を必ず設定すること）
    subject = "✅ 通知ボット：処理完了"
    body = "本日の処理が正常に完了しました。"
    # DRY_RUN 環境変数があれば実際の送信は行われません
    send_notification(subject, body)


=======
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()


def send_notification(subject: str, body: str, dry_run: bool = False) -> bool:
    """Send a notification email.

    This function is safe to import (it will not run on import). Call it
    explicitly or run the module as a script. If DRY_RUN env is set to
    1/true, the function will only simulate sending.
    """
    # allow env override
    env_dry = os.getenv("DRY_RUN", "0").lower() in ("1", "true")
    dry_run = dry_run or env_dry

    # 件名が空だった場合の保険
    if not subject:
        subject = "通知ボット"

    # 件名に応じた本文付加
    if "資金警告" in subject:
        body += "\n⚠️ 追加資金の検討をおすすめします。"
    elif "処理完了" in subject:
        body += "\n✅ 本日のBot処理は正常に終了しました。"

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    to_email = os.getenv("TO_EMAIL")

    # basic validation
    if dry_run:
        print(f"[DRY_RUN] would send mail to: {to_email} via {smtp_host}:{smtp_port}")
        print(f"[DRY_RUN] subject: {subject}\nbody:\n{body}")
        return True

    if not smtp_host or not smtp_user or not smtp_pass or not to_email:
        print("ERROR: SMTP settings incomplete. Check SMTP_HOST/SMTP_USER/SMTP_PASS/TO_EMAIL environment variables.")
        return False

    message = MIMEMultipart()
    message["From"] = smtp_user
    message["To"] = to_email
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


if __name__ == "__main__":
    # スクリプトとして実行する場合のテスト送信（本番では env を必ず設定すること）
    subject = "✅ 通知ボット：処理完了"
    body = "本日の処理が正常に完了しました。"
    # DRY_RUN 環境変数があれば実際の送信は行われません
    send_notification(subject, body)


>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d

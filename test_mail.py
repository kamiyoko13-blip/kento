# テスト用メール送信スクリプト
from ninibo1127 import send_notification, smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject

if __name__ == '__main__':
    missing = []
    for name, val in (('SMTP_HOST', smtp_host), ('SMTP_PORT', smtp_port), ('SMTP_USER', smtp_user), ('SMTP_PASSWORD', smtp_password), ('EMAIL_TO', email_to)):
        if not val:
            missing.append(name)
    if missing:
        print('メール設定が不足しています: ' + ', '.join(missing))
        print('`.env` を確認して必要な SMTP_* および EMAIL_TO を設定してください。')
    else:
        print('メール送信テストを開始します（1回）')
        send_notification(smtp_host, smtp_port, smtp_user, smtp_password, email_to, subject or 'テストメール', 'テストメール: 通知機能の確認です。')
        print('送信処理が完了しました。ログ／受信トレイを確認してください。')

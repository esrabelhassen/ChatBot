import os
import json
import re
import imaplib
import email
import smtplib
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.header import decode_header
from dotenv import load_dotenv
load_dotenv()
PENDING_QUESTIONS_FILE = "pending_questions.json"

def fetch_admin_replies():
    email_account = "fsmchatbot@gmail.com"
    password = os.getenv("BOT_EMAIL_PASSWORD")

    with imaplib.IMAP4_SSL("imap.gmail.com") as mail:
        mail.login(email_account, password)
        mail.select("inbox")

        result, data = mail.search(None, 'UNSEEN')
        for num in data[0].split():
            result, msg_data = mail.fetch(num, '(RFC822)')
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            raw_subject = msg["subject"]
            decoded_parts = decode_header(raw_subject)
            subject = ''.join([
                part.decode(charset or 'utf-8') if isinstance(part, bytes) else part
                for part, charset in decoded_parts
            ])
            print(f"✅ Decoded subject: {subject}")
            reply_to = msg["reply-to"] or msg["from"]
            body = ""

            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body += part.get_payload(decode=True).decode()
            else:
                body += msg.get_payload(decode=True).decode()

            print(f"DEBUG - Raw subject: {repr(subject)}")
            match = re.search(r"\[([a-f0-9\-]{36})\]", subject)
            if match:
                question_id = match.group(1)
                print(f"Processing reply for question ID: {question_id}")
                process_admin_reply(question_id, body, msg)
            else:
                print("No question ID found in subject.")

def process_admin_reply(question_id, admin_response, full_msg):
    with open(PENDING_QUESTIONS_FILE, "r") as f:
        data = json.load(f)

    entry = next((item for item in data if item["id"] == question_id), None)
    if not entry:
        print("No matching question found.")
        return

    user_email = entry["user_email"]

    message = MIMEMultipart()
    message["From"] = "fsmchatbot@gmail.com"
    message["To"] = user_email
    message["Subject"] = "Réponse à votre question posée au chatbot universitaire"
    message.attach(MIMEText(admin_response, "plain"))

    attachments = []

    for part in full_msg.walk():
        if part.get_content_maintype() == "multipart" or part.get('Content-Disposition') is None:
            continue
        file_data = part.get_payload(decode=True)
        file_name = part.get_filename()
        if file_data and file_name:
            encoded_content = base64.b64encode(file_data).decode('utf-8')
            attachments.append({
                "filename": file_name,
                "content": encoded_content
            })

            attachment = MIMEApplication(file_data)
            attachment.add_header('Content-Disposition', 'attachment', filename=file_name)
            message.attach(attachment)

    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "fsmchatbot@gmail.com"
    sender_password = os.getenv("BOT_EMAIL_PASSWORD")

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(message)

    print(f"✅ Sent reply to {user_email}")

    answered_file = "answered_questions.json"

    entry["response"] = admin_response.strip()
    entry["attachments"] = attachments

    if os.path.exists(answered_file):
        with open(answered_file, "r") as f:
            answered_data = json.load(f)
    else:
        answered_data = []

    answered_data.append(entry)

    with open(answered_file, "w") as f:
        json.dump(answered_data, f, indent=4)

    data = [item for item in data if item["id"] != question_id]
    with open(PENDING_QUESTIONS_FILE, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    fetch_admin_replies()
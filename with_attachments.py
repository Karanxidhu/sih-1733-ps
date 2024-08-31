import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

smtp_port = 587
smtp_server = "smtp.gmail.com"

email_from = "enforcer.899@gmail.com"
pswd = "rknyceqnmsumizzr"

subject = "SAR Image Colorization Result"

def send_emails(email_list, image_paths):
    attachments = [
        {"filename": path, "mimetype": "image/jpeg"} for path in image_paths
    ]
    
    attachment_packages = []
    for attachment in attachments:
        with open(attachment["filename"], 'rb') as file:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(attachment['filename'])}")
        part.add_header('Content-Type', attachment['mimetype'])
        attachment_packages.append(part)

    print("Connecting to server...")
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(email_from, pswd)
        print("Successfully connected to server")
        print()

        for person in email_list:
            body = f"""
Hello,

Attached are the original and colorized versions of your uploaded Synthetic Aperture Radar (SAR) image.

I hope you find them interesting!

Best regards,
{email_from}
"""

            msg = MIMEMultipart()
            msg['From'] = email_from
            msg['To'] = person
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            for attachment in attachment_packages:
                msg.attach(attachment)

            text = msg.as_string()

            print(f"Sending email to: {person}...")
            server.sendmail(email_from, person, text)
            print(f"Email sent to: {person}")
            print()

    print("All emails sent. SMTP connection closed.")


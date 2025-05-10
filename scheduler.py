import time
from send_student import fetch_admin_replies

while True:
    print("Checking for new emails...")
    fetch_admin_replies()
    time.sleep(60)  
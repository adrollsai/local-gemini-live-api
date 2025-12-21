import os
from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
from_number = os.getenv("TWILIO_PHONE_NUMBER")
to_number = os.getenv("YOUR_PHONE_NUMBER")
base_url = os.getenv("SERVER_URL")

# Validate configuration
if not all([account_sid, auth_token, from_number, to_number, base_url]):
    print("❌ Error: Missing variables in .env file.")
    exit(1)

# Ensure the URL ends with /twiml (endpoints defined in server.py)
webhook_url = f"{base_url.rstrip('/')}/twiml"

print(f"📞 Initiating call...")
print(f"   From: {from_number}")
print(f"   To:   {to_number}")
print(f"   URL:  {webhook_url}")

try:
    # Initialize Twilio Client
    client = Client(account_sid, auth_token)

    # Make the call
    call = client.calls.create(
        to=to_number,
        from_=from_number,
        url=webhook_url
    )

    print(f"\n✅ Call initiated! SID: {call.sid}")

except Exception as e:
    print(f"\n❌ Failed to make call: {e}")
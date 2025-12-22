import os
import requests
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
api_key = os.getenv("EXOTEL_API_KEY")
api_token = os.getenv("EXOTEL_API_TOKEN")
subdomain = os.getenv("EXOTEL_SUBDOMAIN")
account_sid = os.getenv("EXOTEL_SID")
flow_id = os.getenv("EXOTEL_FLOW_ID") # <--- NEW: Flow ID from Dashboard
exophone = os.getenv("EXOTEL_PHONE_NUMBER") 
user_number = os.getenv("YOUR_PHONE_NUMBER")

# Validate
if not all([api_key, api_token, subdomain, account_sid, flow_id, exophone, user_number]):
    print("❌ Error: Missing variables in .env file (Check EXOTEL_FLOW_ID).")
    exit(1)

# Exotel API Endpoint
url = f"https://{subdomain}.exotel.com/v1/Accounts/{account_sid}/Calls/connect.json"

# The logic: 
# 1. Exotel dials 'From' (Your mobile).
# 2. When you pick up, it executes the 'Url' (Your Flow/Stream).
flow_url = f"http://my.exotel.com/{account_sid}/exoml/start_voice/{flow_id}"

print(f"📞 Initiating call...")
print(f"   Dialing User: {user_number}")
print(f"   Connecting to Flow ID: {flow_id}")

try:
    payload = {
        'From': user_number,       # Call YOU first
        'CallerId': exophone,      # Show Exophone on your screen
        'Url': flow_url,           # Connect to the Stream Flow
        'CallType': "trans"        # Try Transactional to bypass DND
    }

    response = requests.post(
        url,
        auth=(api_key, api_token),
        data=payload
    )

    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Call initiated! SID: {data.get('Call', {}).get('Sid')}")
    else:
        print(f"\n❌ Failed: {response.status_code}")
        print(f"   Response: {response.text}")

except Exception as e:
    print(f"\n❌ Error: {e}")
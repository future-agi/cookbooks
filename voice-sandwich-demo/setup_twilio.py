"""
Setup script to configure Twilio phone number for voice agent streaming.

Usage:
    1. Make sure .env has TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER
    2. Start ngrok: ngrok http 8081
    3. Start server: cd components/python/src && python main.py
    4. Run: python setup_twilio.py <ngrok-url>

Example:
    python setup_twilio.py https://abcd-1234.ngrok-free.app
"""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()


def main():
    if len(sys.argv) < 2:
        print("Usage: python setup_twilio.py <ngrok-url>")
        print("Example: python setup_twilio.py https://abcd-1234.ngrok-free.app")
        sys.exit(1)

    ngrok_url = sys.argv[1].rstrip("/")
    ws_url = ngrok_url.replace("https://", "wss://") + "/ws/twilio"

    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    phone = os.environ.get("TWILIO_PHONE_NUMBER")

    if not all([sid, token, phone]):
        print("Missing env vars. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER in .env")
        sys.exit(1)

    # Find the phone number SID
    r = requests.get(
        f"https://api.twilio.com/2010-04-01/Accounts/{sid}/IncomingPhoneNumbers.json",
        params={"PhoneNumber": phone},
        auth=(sid, token),
    )
    r.raise_for_status()
    numbers = r.json()["incoming_phone_numbers"]

    if not numbers:
        print(f"Phone number {phone} not found on this account")
        sys.exit(1)

    pn_sid = numbers[0]["sid"]
    print(f"Found: {numbers[0]['phone_number']} ({pn_sid})")

    # Configure TwiML to stream audio to our WebSocket
    twiml = f'<Response><Connect><Stream url="{ws_url}" /></Connect></Response>'

    r2 = requests.post(
        f"https://api.twilio.com/2010-04-01/Accounts/{sid}/IncomingPhoneNumbers/{pn_sid}.json",
        data={
            "VoiceUrl": f"https://handler.twilio.com/twiml/inline?twiml={requests.utils.quote(twiml)}",
        },
        auth=(sid, token),
    )

    if r2.ok:
        print(f"Streaming to: {ws_url}")
        print(f"Call {phone} to talk to your agent!")
    else:
        print(f"Error: {r2.status_code} {r2.text}")
        sys.exit(1)


if __name__ == "__main__":
    main()

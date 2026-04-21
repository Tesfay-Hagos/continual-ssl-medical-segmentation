"""
One-time Google Drive OAuth2 setup — run this LOCALLY (not on Kaggle).

Steps:
  1. Go to https://console.cloud.google.com/
  2. Create a project → Enable "Google Drive API"
  3. Credentials → Create OAuth2 client ID → Desktop app
  4. Download the client_secrets JSON file
  5. Run:  python src/utils/gdrive_setup.py --secrets client_secrets.json
  6. A browser window opens — log in with your Google account
  7. credentials.json is saved in the current directory
  8. Copy its contents → Kaggle secret named  GDRIVE_CREDENTIALS
  9. Create a folder in your Drive for checkpoints
 10. Copy the folder ID from the URL → Kaggle secret named  GDRIVE_FOLDER_ID
     (the folder ID is the long alphanumeric string at the end of the Drive URL)
"""

import argparse
import json
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secrets", default="client_secrets.json",
                    help="Path to downloaded OAuth2 client secrets JSON")
    ap.add_argument("--out", default="credentials.json",
                    help="Output path for the saved credentials")
    args = ap.parse_args()

    flow = InstalledAppFlow.from_client_secrets_file(args.secrets, SCOPES)
    creds = flow.run_local_server(port=0)

    creds_dict = {
        "token":         creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri":     creds.token_uri,
        "client_id":     creds.client_id,
        "client_secret": creds.client_secret,
    }
    Path(args.out).write_text(json.dumps(creds_dict, indent=2))
    print(f"\nCredentials saved to {args.out}")
    print("Copy the contents of this file into Kaggle secret: GDRIVE_CREDENTIALS")
    print("Also set Kaggle secret GDRIVE_FOLDER_ID to your Drive folder ID.")


if __name__ == "__main__":
    main()

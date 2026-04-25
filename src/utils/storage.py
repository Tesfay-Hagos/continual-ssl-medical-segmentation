"""
Unified checkpoint persistence: WandB artifacts + Google Drive.

Priority order:
  1. WandB artifacts  — versioned, 100 GB free, auto-downloaded on resume
  2. Google Drive     — optional backup; survives WandB outages / quota issues

Google Drive one-time setup (run locally, NOT on Kaggle):
  pip install google-auth-oauthlib google-api-python-client
  python src/utils/gdrive_setup.py
  → saves credentials.json; copy its contents into Kaggle secret GDRIVE_CREDENTIALS
  → create a folder in Drive, copy its ID into Kaggle secret GDRIVE_FOLDER_ID
"""

import json
import os
from pathlib import Path

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    _GDRIVE_LIBS = True
except ImportError:
    _GDRIVE_LIBS = False


# ── WandB ─────────────────────────────────────────────────────────────────────

def wandb_upload(local_path: Path, artifact_name: str):
    """Add a file to an existing or new WandB artifact (non-fatal on error).

    All files added under the same artifact_name form one versioned artifact,
    so latest.pth and pretrain_done.json both land in pretrain-checkpoint:latest.
    """
    if not (_WANDB and wandb.run is not None):
        return
    try:
        art = wandb.Artifact(artifact_name, type="checkpoint")
        art.add_file(str(local_path), name=local_path.name)
        wandb.log_artifact(art)
    except Exception as e:
        print(f"  ⚠️  WandB upload failed ({artifact_name}): {e}")


def wandb_download(artifact_name: str, filename: str,
                   dest_dir: Path, project: str) -> bool:
    """Download latest version of a WandB artifact. Returns True on success."""
    if not _WANDB:
        return False
    if (dest_dir / filename).exists():
        return False
    try:
        api    = wandb.Api()
        entity = (wandb.run.entity if wandb.run is not None
                  else api.default_entity)
        prefix = f"{entity}/{project}" if entity else project
        art  = api.artifact(f"{prefix}/{artifact_name}:latest")
        art.get_path(filename).download(root=str(dest_dir))
        print(f"  ✅ Restored {filename} from WandB ({artifact_name})")
        return True
    except Exception as e:
        print(f"  ℹ️  WandB restore failed ({artifact_name}): {e}")
        return False


# ── Google Drive ──────────────────────────────────────────────────────────────

def _gdrive_service(credentials_json: str):
    """Build an authenticated Google Drive v3 service from stored credentials."""
    if not _GDRIVE_LIBS:
        raise RuntimeError("google-api-python-client not installed. "
                           "Run: pip install google-auth-oauthlib "
                           "google-api-python-client")
    info  = json.loads(credentials_json)
    creds = Credentials(
        token=info.get("token"),
        refresh_token=info["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=info["client_id"],
        client_secret=info["client_secret"],
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def gdrive_upload(local_path: Path, folder_id: str, credentials_json: str):
    """Upload a file to Google Drive folder (non-fatal on error)."""
    if not credentials_json or not folder_id:
        return
    try:
        svc  = _gdrive_service(credentials_json)
        name = local_path.name

        # Overwrite if already exists (keep one copy per checkpoint name)
        results = svc.files().list(
            q=f"name='{name}' and '{folder_id}' in parents and trashed=false",
            fields="files(id)").execute()
        existing = results.get("files", [])

        media = MediaFileUpload(str(local_path), resumable=True)
        if existing:
            svc.files().update(fileId=existing[0]["id"], media_body=media).execute()
        else:
            meta = {"name": name, "parents": [folder_id]}
            svc.files().create(body=meta, media_body=media,
                               fields="id").execute()
        print(f"  ☁️  Saved {name} → Google Drive")
    except Exception as e:
        print(f"  ⚠️  Google Drive upload failed: {e}")


def gdrive_download(filename: str, dest_dir: Path,
                    folder_id: str, credentials_json: str) -> bool:
    """Download a file from Google Drive. Returns True on success."""
    dest = dest_dir / filename
    if dest.exists() or not credentials_json or not folder_id:
        return False
    try:
        import io
        svc     = _gdrive_service(credentials_json)
        results = svc.files().list(
            q=f"name='{filename}' and '{folder_id}' in parents and trashed=false",
            fields="files(id)").execute()
        files = results.get("files", [])
        if not files:
            return False
        content = svc.files().get_media(fileId=files[0]["id"]).execute()
        dest.write_bytes(content)
        print(f"  ✅ Restored {filename} from Google Drive")
        return True
    except Exception as e:
        print(f"  ℹ️  Google Drive restore failed ({filename}): {e}")
        return False


# ── Unified save/restore ──────────────────────────────────────────────────────

def save_checkpoint(local_path: Path, artifact_name: str,
                    gdrive_folder_id: str = "", gdrive_creds: str = ""):
    """Save checkpoint to WandB artifact AND optionally Google Drive."""
    wandb_upload(local_path, artifact_name)
    if gdrive_folder_id and gdrive_creds:
        gdrive_upload(local_path, gdrive_folder_id, gdrive_creds)


def restore_checkpoint(filename: str, dest_dir: Path, artifact_name: str,
                        project: str, gdrive_folder_id: str = "",
                        gdrive_creds: str = "") -> bool:
    """Try WandB first, then Google Drive. Returns True if restored."""
    if wandb_download(artifact_name, filename, dest_dir, project):
        return True
    if gdrive_folder_id and gdrive_creds:
        return gdrive_download(filename, dest_dir, gdrive_folder_id, gdrive_creds)
    return False

# Save final post-process to Google Drive
import os, json, logging
from typing import Optional

# Conditional imports for Google Drive (only when needed)
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False
    # Create dummy classes for when Google Drive is not available
    class service_account:
        class Credentials:
            @staticmethod
            def from_service_account_info(*args, **kwargs):
                raise ImportError("Google Drive dependencies not available")
    
    class build:
        @staticmethod
        def build(*args, **kwargs):
            raise ImportError("Google Drive dependencies not available")
    
    class MediaFileUpload:
        def __init__(self, *args, **kwargs):
            raise ImportError("Google Drive dependencies not available")

from utils.token import get_credentials

logger = logging.getLogger("dsaver")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logger.addHandler(handler)

class DriveSaver:
    """Google Drive uploader. Prefers OAuth; optional SA fallback (Shared Drive only)."""

    def __init__(self, default_folder_id: Optional[str] = None):
        self.service = None
        self.folder_id = default_folder_id or os.getenv("GDRIVE_FOLDER_ID")
        self.supports_all_drives = os.getenv("GDRIVE_FOLDER_IS_SHARED", "false").lower() in ("1","true","yes")
        self.allow_sa_fallback = os.getenv("GDRIVE_ALLOW_SA_FALLBACK", "false").lower() in ("1","true","yes")
        
        # Check if Google Drive is available
        if not GOOGLE_DRIVE_AVAILABLE:
            logger.warning("⚠️ Google Drive dependencies not available - DriveSaver will be disabled")
            return
            
        if not self.folder_id:
            logger.warning("📁 No GDRIVE_FOLDER_ID set; uploads must provide folder_id explicitly")
        self._initialize_service()

    def _initialize_service(self):
        if not GOOGLE_DRIVE_AVAILABLE:
            logger.warning("⚠️ Google Drive dependencies not available - skipping service initialization")
            return
            
        creds = get_credentials()
        if creds:
            logger.info("✅ Using OAuth credentials")
        else:
            # Optional SA fallback — ONLY valid for Shared Drives where SA is a member
            if self.allow_sa_fallback:
                creds_env = os.getenv("GDRIVE_CREDENTIALS_JSON")
                if creds_env:
                    try:
                        info = json.loads(creds_env)
                        if info.get("type") == "service_account":
                            creds = service_account.Credentials.from_service_account_info(
                                info, scopes=["https://www.googleapis.com/auth/drive"]
                            )
                            logger.info("✅ Using Service Account credentials (fallback)")
                            if not self.supports_all_drives:
                                logger.warning("⚠️ SA fallback without Shared Drive mode will likely fail (no quota). "
                                               "Set GDRIVE_FOLDER_IS_SHARED=true and use a Shared Drive folder ID.")
                        else:
                            logger.error("❌ GDRIVE_CREDENTIALS_JSON is not a service account JSON")
                    except Exception as e:
                        logger.error(f"❌ Failed to init Service Account: {e}")
            if not creds:
                logger.error("❌ No valid Google credentials available (OAuth or SA).")
                self.service = None
                return
        # Build Drive service
        self.service = build("drive", "v3", credentials=creds)
        logger.info("✅ Google Drive service initialized")

    def upload_file_to_drive(self, file_path: str, folder_id: Optional[str] = None, mimetype: Optional[str] = None) -> bool:
        if not GOOGLE_DRIVE_AVAILABLE:
            logger.warning("⚠️ Google Drive dependencies not available - upload skipped")
            return False
            
        if not self.service:
            logger.error("❌ Drive service not initialized")
            return False
        try:
            target_folder = folder_id or self.folder_id
            name = os.path.basename(file_path)
            media = MediaFileUpload(file_path, mimetype=mimetype or "application/octet-stream")
            metadata = {"name": name, "parents": [target_folder]}
            req = self.service.files().create(
                body=metadata,
                media_body=media,
                fields="id",
                supportsAllDrives=self.supports_all_drives
            )
            req.execute()
            logger.info(f"✅ Uploaded '{name}' to Drive (folder: {target_folder})")
            return True
        except Exception as e:
            logger.error(f"❌ Drive upload failed: {e}")
            return False

    def is_service_available(self) -> bool:
        return GOOGLE_DRIVE_AVAILABLE and self.service is not None

    def set_folder_id(self, folder_id: str):
        self.folder_id = folder_id
        logger.info(f"📁 Default folder ID updated: {folder_id}")

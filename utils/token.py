# GCS credential token refresher
import os, json, logging
from typing import Optional
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request

logger = logging.getLogger("token")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
TOKEN_FILE = os.getenv("GDRIVE_TOKEN_FILE", "cache/secrets/gdrive_token.json")

def _load_oauth_client_web():
    cfg_env = os.getenv("GDRIVE_CREDENTIALS_JSON")
    if not cfg_env:
        return None
    try:
        cfg = json.loads(cfg_env)
        return cfg.get("web")
    except Exception as e:
        logger.error(f"❌ Failed to parse GDRIVE_CREDENTIALS_JSON: {e}")
        return None

def _ensure_dirs():
    base = os.path.dirname(TOKEN_FILE)
    if base and not os.path.exists(base):
        os.makedirs(base, exist_ok=True)

def get_credentials() -> Optional[Credentials]:
    # 1) Token file
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            creds = Credentials.from_authorized_user_info(data, scopes=SCOPES)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                logger.info("🔄 Refreshed access token from token file")
            return creds
        except Exception as e:
            logger.warning(f"⚠️ Failed to load token file: {e}")

    # 2) Refresh token in env
    refresh = os.getenv("GDRIVE_REFRESH_TOKEN")
    web = _load_oauth_client_web()
    if refresh and web:
        creds = Credentials(
            None,
            refresh_token=refresh,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=web.get("client_id"),
            client_secret=web.get("client_secret"),
            scopes=SCOPES,
        )
        if creds and (creds.expired or not creds.valid):
            try:
                creds.refresh(Request())
                logger.info("🔄 Refreshed access token from env refresh token")
            except Exception as e:
                logger.warning(f"⚠️ Refresh with env token failed: {e}")
        return creds

    # 3) Nothing available
    return None

def build_auth_url(redirect_uri: str) -> str:
    web = _load_oauth_client_web()
    if not web:
        raise RuntimeError("GDRIVE_CREDENTIALS_JSON missing or invalid ('web' section required)")
    flow = Flow.from_client_config({"web": web}, scopes=SCOPES, redirect_uri=redirect_uri)
    auth_url, _ = flow.authorization_url(
        prompt="consent",
        access_type="offline",
        include_granted_scopes="true"
    )
    return auth_url

def exchange_code(code: str, redirect_uri: str) -> Credentials:
    web = _load_oauth_client_web()
    if not web:
        raise RuntimeError("GDRIVE_CREDENTIALS_JSON missing or invalid ('web' section required)")
    flow = Flow.from_client_config({"web": web}, scopes=SCOPES, redirect_uri=redirect_uri)
    flow.fetch_token(code=code)
    creds: Credentials = flow.credentials

    info = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": web.get("client_id"),
        "client_secret": web.get("client_secret"),
        "scopes": SCOPES,
    }
    _ensure_dirs()
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        json.dump(info, f)
    logger.info("✅ Saved Google refresh token to %s", TOKEN_FILE)

    # also set env for current process
    if creds.refresh_token:
        os.environ["GDRIVE_REFRESH_TOKEN"] = creds.refresh_token

    return creds
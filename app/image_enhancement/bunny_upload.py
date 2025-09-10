import os
import uuid
import re
from typing import Optional, Tuple, Dict

import requests


def _normalize_host(host: str) -> str:
    host = (host or "").strip()
    # Remove trailing semicolons and slashes/spaces
    host = host.rstrip(" ;/\t\n\r")
    # Remove accidental trailing semicolon in the middle too
    if host.endswith(";"):
        host = host[:-1]
    # If protocol missing, default to https
    if host and not host.startswith(("http://", "https://")):
        host = f"https://{host}"
    return host


def _normalize_pull_zone_host(host: str) -> str:
    host = (host or "").strip()
    # Remove scheme if provided
    host = re.sub(r"^https?://", "", host, flags=re.IGNORECASE)
    # Trim slashes and semicolons
    host = host.strip(" ;/")
    return host


_RAW_BUNNY_STORAGE_HOST = os.environ.get("BUNNY_STORAGE_HOST", "")
BUNNY_STORAGE_HOST = _normalize_host(_RAW_BUNNY_STORAGE_HOST) or "https://storage.bunnycdn.com"
BUNNY_STORAGE_ZONE_NAME = os.environ.get("BUNNY_STORAGE_ZONE_NAME", "")
BUNNY_PASSWORD = os.environ.get("BUNNY_PASSWORD", "")
BUNNY_PULL_ZONE_HOST = _normalize_pull_zone_host(os.environ.get("BUNNY_PULL_ZONE_HOST", ""))


def is_bunny_configured() -> bool:
    return bool(BUNNY_STORAGE_ZONE_NAME and BUNNY_PASSWORD and BUNNY_PULL_ZONE_HOST)


def _sanitize_segment(value: str, fallback: str) -> str:
    value = (value or "").strip()
    if not value:
        return fallback
    # allow letters, numbers, dash, underscore, slash
    value = re.sub(r"[^a-zA-Z0-9_\-/]", "-", value)
    # prevent path traversal
    value = value.replace("..", "-")
    value = value.strip("/")
    return value or fallback


def _derive_extension(filename: str, content_type: str) -> str:
    name = (filename or "").lower()
    if "." in name:
        ext = name.rsplit(".", 1)[-1]
        if ext:
            return ext
    # fallback from content type
    mapping = {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/avif": "avif",
        "image/gif": "gif",
        "application/octet-stream": "bin",
    }
    return mapping.get((content_type or "").lower(), "bin")


def _build_disposition(folder: str, extension: str) -> Dict[str, str]:
    folder_sanitized = _sanitize_segment(folder or "uploads", "uploads")
    uid = uuid.uuid4().hex
    ext = (extension or "").lstrip(".") or "bin"
    final_name = f"{uid}.{ext}"
    bunny_path = f"{folder_sanitized}/{final_name}"
    return {
        "folder": folder_sanitized,
        "finalName": final_name,
        "bunnyPath": bunny_path,
    }


def upload_to_bunny(
    file_bytes: bytes,
    *,
    content_type: str = "application/octet-stream",
    folder: str = "uploads",
    filename: Optional[str] = None,
) -> Dict[str, str]:
    """Upload provided bytes to Bunny Storage.

    Returns dict with: bunny_path, file_name, folder, content_type, public_url.
    Raises requests.HTTPError on non-2xx.
    """
    if not is_bunny_configured():
        raise RuntimeError("Bunny is not configured: set BUNNY_STORAGE_ZONE_NAME, BUNNY_PASSWORD, BUNNY_PULL_ZONE_HOST")

    extension = _derive_extension(filename or "", content_type)
    disposition = _build_disposition(folder, extension)

    bunny_url = f"{BUNNY_STORAGE_HOST}/{BUNNY_STORAGE_ZONE_NAME}/{disposition['bunnyPath']}"

    resp = requests.put(
        bunny_url,
        headers={
            "AccessKey": BUNNY_PASSWORD,
            "Content-Type": content_type or "application/octet-stream",
        },
        data=file_bytes,
        timeout=60,
    )
    if not resp.ok:
        try:
            text = resp.text
        except Exception:
            text = ""
        resp.raise_for_status()

    public_url = f"https://{BUNNY_PULL_ZONE_HOST}/{disposition['bunnyPath']}"
    return {
        "bunny_path": disposition["bunnyPath"],
        "file_name": disposition["finalName"],
        "folder": disposition["folder"],
        "content_type": content_type,
        "public_url": public_url,
    }



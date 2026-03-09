from __future__ import annotations

from io import BytesIO
from typing import Optional, List, Dict
import os
import logging

import requests
from flask import Blueprint, jsonify, request, send_file
from supabase import create_client, Client

from app.convert import (
    ImageConversionError,
    convert_avif_to_jpeg,
    convert_heic_to_jpeg,
    convert_webp_to_jpeg,
)
from app.image_enhancement.bunny_upload import upload_to_bunny


image_convert_bp = Blueprint("image_convert", __name__)

MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

logger = logging.getLogger(__name__)


_SUPABASE_CLIENT: Client | None = None


def get_supabase_client() -> Client:
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is not None:
        return _SUPABASE_CLIENT

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set for process-submission")

    _SUPABASE_CLIENT = create_client(url, key)
    return _SUPABASE_CLIENT


def _get_uploaded_file():
    if "file" not in request.files:
        return None, jsonify({"error": "Missing file field 'file'"}), 400

    file = request.files["file"]
    if file.filename == "":
        return None, jsonify({"error": "Uploaded file must have a filename"}), 400

    return file, None, None


def _jpeg_response(jpeg_bytes: bytes):
    buffer = BytesIO(jpeg_bytes)
    buffer.seek(0)
    # Flask will set the correct Content-Type header
    return send_file(
        buffer,
        mimetype="image/jpeg",
        as_attachment=False,
        download_name="converted.jpg",
    )


def _detect_format_from_filename(name: str) -> Optional[str]:
    name = (name or "").lower()
    if name.endswith(".heic") or name.endswith(".heif"):
        return "heic"
    if name.endswith(".avif"):
        return "avif"
    if name.endswith(".webp"):
        return "webp"
    return None


def _detect_format_from_url_and_mime(url: str, mime: str | None) -> Optional[str]:
    """Best-effort format detection from URL and Content-Type."""
    url = (url or "").lower()
    ext = ""
    if "." in url:
        ext = url.rsplit(".", 1)[-1]

    if ext in {"heic", "heif"}:
        return "heic"
    if ext == "avif":
        return "avif"
    if ext == "webp":
        return "webp"
    if ext in {"jpg", "jpeg"}:
        return "jpeg"
    if ext == "png":
        return "png"

    mime = (mime or "").lower()
    if mime in {"image/heic", "image/heif"}:
        return "heic"
    if mime == "image/avif":
        return "avif"
    if mime == "image/webp":
        return "webp"
    if mime in {"image/jpeg", "image/jpg"}:
        return "jpeg"
    if mime == "image/png":
        return "png"

    return None


def download_image(url: str) -> tuple[bytes, Optional[str]]:
    """Download image bytes from URL with basic size limiting.

    Returns (bytes, content_type).
    """
    if not url or not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ValueError("Invalid image URL")

    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(f"Failed to download image: {exc}") from exc

    # Enforce max download size
    total = 0
    chunks: list[bytes] = []
    for chunk in resp.iter_content(chunk_size=8192):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_DOWNLOAD_BYTES:
            raise ValueError("Image exceeds maximum allowed size of 20MB")
        chunks.append(chunk)

    if not chunks:
        raise ValueError("Downloaded image is empty")

    return b"".join(chunks), resp.headers.get("Content-Type")


@image_convert_bp.route("/heic", methods=["POST"])
def convert_heic_route():
    """
    Convert an uploaded HEIC/HEIF image to JPEG.
    Exposed as: POST /api/convert/heic (see app.__init__.py).
    """
    file, error_resp, status = _get_uploaded_file()
    if error_resp is not None:
        return error_resp, status

    try:
        jpeg_bytes = convert_heic_to_jpeg(file.read())
    except ImageConversionError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "Failed to convert HEIC image"}), 500

    return _jpeg_response(jpeg_bytes)


@image_convert_bp.route("/avif", methods=["POST"])
def convert_avif_route():
    """
    Convert an uploaded AVIF image to JPEG.
    Exposed as: POST /api/convert/avif (see app.__init__.py).
    """
    file, error_resp, status = _get_uploaded_file()
    if error_resp is not None:
        return error_resp, status

    try:
        jpeg_bytes = convert_avif_to_jpeg(file.read())
    except ImageConversionError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "Failed to convert AVIF image"}), 500

    return _jpeg_response(jpeg_bytes)


@image_convert_bp.route("/webp", methods=["POST"])
def convert_webp_route():
    """
    Convert an uploaded WEBP image to JPEG.
    Exposed as: POST /api/convert/webp (see app.__init__.py).
    """
    file, error_resp, status = _get_uploaded_file()
    if error_resp is not None:
        return error_resp, status

    try:
        jpeg_bytes = convert_webp_to_jpeg(file.read())
    except ImageConversionError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "Failed to convert WEBP image"}), 500

    return _jpeg_response(jpeg_bytes)


@image_convert_bp.route("/upload", methods=["POST"])
def convert_and_upload_route():
    """
    Convert a HEIC/HEIF/AVIF/WEBP image to optimized JPEG and upload to Bunny.
    Exposed as: POST /api/convert/upload
    """
    file, error_resp, status = _get_uploaded_file()
    if error_resp is not None:
        return error_resp, status

    # Determine format from filename or mimetype
    fmt = _detect_format_from_filename(file.filename)
    if not fmt:
        mime = (file.mimetype or "").lower()
        if mime in {"image/heic", "image/heif"}:
            fmt = "heic"
        elif mime == "image/avif":
            fmt = "avif"
        elif mime == "image/webp":
            fmt = "webp"

    if fmt not in {"heic", "avif", "webp"}:
        return (
            jsonify(
                {
                    "error": "Unsupported image format. "
                    "Supported formats: HEIC, HEIF, AVIF, WEBP."
                }
            ),
            400,
        )

    data = file.read()
    if not data:
        return jsonify({"error": "Uploaded file is empty"}), 400

    try:
        if fmt == "heic":
            jpeg_bytes = convert_heic_to_jpeg(data)
        elif fmt == "avif":
            jpeg_bytes = convert_avif_to_jpeg(data)
        else:
            jpeg_bytes = convert_webp_to_jpeg(data)
    except ImageConversionError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "Failed to convert image"}), 500

    # Upload to Bunny Storage under product-images/<uuid>.jpg
    try:
        upload_info = upload_to_bunny(
            jpeg_bytes,
            content_type="image/jpeg",
            folder="product-images",
            filename="converted.jpg",
        )
    except Exception as exc:
        return jsonify({"error": f"Failed to upload to Bunny Storage: {exc}"}), 500

    path = upload_info.get("bunny_path") or ""
    file_name = upload_info.get("file_name") or ""
    public_url = upload_info.get("public_url") or ""

    return (
        jsonify(
            {
                "url": public_url,
                "path": path,
                "fileName": file_name,
            }
        ),
        200,
    )


@image_convert_bp.route("/from-url", methods=["POST"])
def convert_from_url_route():
    """
    Convert an image referenced by URL and upload optimized JPEG to Bunny.
    Exposed as: POST /api/convert/from-url

    Expected JSON body:
    {
        "imageUrl": "https://..."
    }
    """
    data = request.get_json(silent=True) or {}
    image_url = data.get("imageUrl")
    if not image_url or not isinstance(image_url, str):
        return jsonify({"success": False, "error": "Field 'imageUrl' is required"}), 400

    print("Downloading image:", image_url)

    # Download image
    try:
        image_bytes, content_type = download_image(image_url)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": f"Failed to download image: {exc}"}), 400

    fmt = _detect_format_from_url_and_mime(image_url, content_type)
    print("Conversion format:", fmt)

    # If already JPEG/PNG, skip conversion and return original URL
    if fmt in {"jpeg", "png"}:
        return (
            jsonify(
                {
                    "success": True,
                    "originalUrl": image_url,
                    "convertedUrl": image_url,
                    "url": image_url,
                    "skipped": True,
                }
            ),
            200,
        )

    if fmt not in {"heic", "avif", "webp"}:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Unsupported or unknown image format. "
                    "Supported conversion formats: HEIC, HEIF, AVIF, WEBP.",
                }
            ),
            400,
        )

    # Convert to optimized JPEG
    try:
        if fmt == "heic":
            jpeg_bytes = convert_heic_to_jpeg(image_bytes)
        elif fmt == "avif":
            jpeg_bytes = convert_avif_to_jpeg(image_bytes)
        else:
            jpeg_bytes = convert_webp_to_jpeg(image_bytes)
    except ImageConversionError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": f"Failed to convert image: {exc}"}), 500

    print("Uploading converted image to Bunny...")

    # Upload converted JPEG to Bunny under product-images/<uuid>.jpg
    try:
        upload_info = upload_to_bunny(
            jpeg_bytes,
            content_type="image/jpeg",
            folder="product-images",
            filename="converted-from-url.jpg",
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Failed to upload to Bunny Storage: {exc}",
                }
            ),
            500,
        )

    converted_url = upload_info.get("public_url") or ""

    return (
        jsonify(
            {
                "success": True,
                "originalUrl": image_url,
                "convertedUrl": converted_url,
                "url": converted_url,
            }
        ),
        200,
    )


@image_convert_bp.route("/batch", methods=["POST"])
def convert_batch_route():
    """
    Batch convert images referenced by URLs and upload optimized JPEGs to Bunny.
    Exposed as: POST /api/convert/batch

    Expected JSON body:
    {
        "imageUrls": ["https://...", "..."]
    }

    Response:
    {
        "converted": {
            "original_url": "converted_url"
        }
    }
    """
    data = request.get_json(silent=True) or {}
    image_urls = data.get("imageUrls") or []

    if not isinstance(image_urls, list) or not all(isinstance(u, str) for u in image_urls):
        return jsonify({"converted": {}, "error": "Field 'imageUrls' must be a list of strings"}), 400

    mapping: Dict[str, str] = {}

    for url in image_urls:
        original_url = url
        mapping[original_url] = original_url  # default fallback

        if not url:
            continue

        print("Processing image URL:", url)

        # Detect extension
        base = url.split("?", 1)[0].lower()
        ext = base.rsplit(".", 1)[-1] if "." in base else ""

        # Skip non-convertible formats
        if ext in {"jpg", "jpeg", "png"}:
            print("Skipping conversion for (already JPEG/PNG):", url)
            continue
        if ext not in {"heic", "heif", "avif", "webp"}:
            print("Skipping unsupported format:", url)
            continue

        # Download image
        print("Downloading image:", url)
        try:
            image_bytes, content_type = download_image(url)
        except Exception as exc:
            print("Download failed for", url, "error:", exc)
            continue

        # Determine conversion format
        if ext in {"heic", "heif"}:
            fmt = "heic"
        else:
            fmt = ext  # avif / webp

        print("Converting image format:", fmt, "for", url)

        try:
            if fmt == "heic":
                jpeg_bytes = convert_heic_to_jpeg(image_bytes)
            elif fmt == "avif":
                jpeg_bytes = convert_avif_to_jpeg(image_bytes)
            else:  # webp
                jpeg_bytes = convert_webp_to_jpeg(image_bytes)
        except ImageConversionError as exc:
            print("Conversion error for", url, ":", exc)
            continue
        except Exception as exc:
            print("Unexpected conversion error for", url, ":", exc)
            continue

        print("Uploading converted image to Bunny for", url)

        try:
            upload_info = upload_to_bunny(
                jpeg_bytes,
                content_type="image/jpeg",
                folder="product-images",
                filename="converted-batch.jpg",
            )
            converted_url = upload_info.get("public_url") or original_url
            mapping[original_url] = converted_url
            print("Conversion complete for", url, "->", converted_url)
        except Exception as exc:
            print("Upload failed for", url, ":", exc)
            # keep original URL in mapping
            continue

    return jsonify({"converted": mapping}), 200


def run_agents(submission_id: str) -> None:
    """Placeholder for downstream agent pipeline trigger."""
    logger.info("Triggering agents for submission %s", submission_id)


@image_convert_bp.route("/process-submission", methods=["POST"])
def process_submission_route():
    """
    Process a product submission:
    - Fetch submission by id
    - Convert HEIC/HEIF/AVIF/WEBP images in media array
    - Upload converted JPEGs to Bunny
    - Update product_submissions.media
    - Trigger downstream agents

    Exposed as: POST /api/convert/process-submission

    Expected JSON body:
    {
        "submissionId": "<uuid>"
    }
    """
    data = request.get_json(silent=True) or {}
    submission_id = data.get("submissionId")
    if not submission_id or not isinstance(submission_id, str):
        return jsonify({"success": False, "error": "Field 'submissionId' is required"}), 400

    logger.info("Processing submission %s", submission_id)

    try:
        supabase = get_supabase_client()
    except Exception as exc:
        logger.error("Supabase client initialization failed: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500

    # Step 1 — Fetch submission
    try:
        res = (
            supabase.table("product_submissions")
            .select("id, media")
            .eq("id", submission_id)
            .single()
            .execute()
        )
        row = (res.data or {}) if hasattr(res, "data") else res
    except Exception as exc:
        logger.error("Failed to fetch submission %s: %s", submission_id, exc)
        return jsonify({"success": False, "error": f"Failed to fetch submission: {exc}"}), 500

    if not row:
        return jsonify({"success": False, "error": "Submission not found"}), 404

    media: List[Dict] = row.get("media") or []
    if not isinstance(media, list):
        logger.warning("Submission %s has non-list media field", submission_id)
        media = []

    updated_media: List[Dict] = []

    # Step 2 & 3 — Iterate media entries and convert if needed
    for item in media:
        if not isinstance(item, dict):
            updated_media.append(item)
            continue

        if item.get("type") != "image":
            updated_media.append(item)
            continue

        url = item.get("url")
        if not isinstance(url, str) or not url:
            updated_media.append(item)
            continue

        lower_url = url.split("?", 1)[0].lower()
        ext = lower_url.rsplit(".", 1)[-1] if "." in lower_url else ""

        # Skip non-convertible formats
        if ext in {"jpg", "jpeg", "png"}:
            updated_media.append(item)
            continue

        if ext not in {"heic", "heif", "avif", "webp"}:
            updated_media.append(item)
            continue

        logger.info("Downloading image %s", url)

        # Download image from Bunny
        try:
            image_bytes, content_type = download_image(url)
        except Exception as exc:
            logger.error("Failed to download image %s: %s", url, exc)
            # Keep original URL on failure
            updated_media.append(item)
            continue

        # Determine format for conversion
        if ext in {"heic", "heif"}:
            fmt = "heic"
        else:
            fmt = ext  # avif / webp

        logger.info("Converting format %s for %s", fmt, url)

        try:
            if fmt == "heic":
                jpeg_bytes = convert_heic_to_jpeg(image_bytes)
            elif fmt == "avif":
                jpeg_bytes = convert_avif_to_jpeg(image_bytes)
            else:  # webp
                jpeg_bytes = convert_webp_to_jpeg(image_bytes)
        except ImageConversionError as exc:
            logger.error("Conversion error for %s: %s", url, exc)
            updated_media.append(item)
            continue
        except Exception as exc:
            logger.error("Unexpected conversion error for %s: %s", url, exc)
            updated_media.append(item)
            continue

        logger.info("Uploading converted image to Bunny for %s", url)

        try:
            upload_info = upload_to_bunny(
                jpeg_bytes,
                content_type="image/jpeg",
                folder="product-images",
                filename="converted-submission.jpg",
            )
            converted_url = upload_info.get("public_url") or url
        except Exception as exc:
            logger.error("Upload to Bunny failed for %s: %s", url, exc)
            updated_media.append(item)
            continue

        new_item = dict(item)
        new_item["url"] = converted_url
        updated_media.append(new_item)

    # Step 5 — Save updated media array
    try:
        _ = (
            supabase.table("product_submissions")
            .update({"media": updated_media})
            .eq("id", submission_id)
            .execute()
        )
    except Exception as exc:
        logger.error("Failed to update submission %s media: %s", submission_id, exc)
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Failed to update submission media: {exc}",
                }
            ),
            500,
        )

    # Step 6 — Trigger agents
    try:
        run_agents(submission_id)
    except Exception as exc:
        logger.error("run_agents failed for %s: %s", submission_id, exc)
        # Do not fail the whole request if agents fail

    return jsonify({"success": True, "submissionId": submission_id, "media": updated_media}), 200

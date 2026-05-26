from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Optional, List, Dict, Tuple
import os
import logging

import requests
from PIL import Image
from flask import Blueprint, jsonify, request, send_file
from supabase import create_client, Client

from app.convert import (
    ImageConversionError,
    convert_avif_to_jpeg,
    convert_dng_to_jpeg,
    convert_heic_to_jpeg,
    convert_webp_to_jpeg,
)
from app.image_enhancement.bunny_upload import upload_to_bunny


image_convert_bp = Blueprint("image_convert", __name__)

MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024       # 20 MB (default)
MAX_DOWNLOAD_BYTES_DNG = 80 * 1024 * 1024  # 80 MB (DNG/RAW files are typically 20-80MB)

logger = logging.getLogger(__name__)


HASH_THREAD_WORKERS = 20  # 20 concurrent: ~20×10MB=200MB peak, safe given real-world image sizes


def _download_and_hash(url: str) -> Tuple[str, int | None]:
    """Download a single image URL and compute its pHash. Returns (url, phash_or_None)."""
    if not url:
        return url, None
    base = url.split("?", 1)[0].lower()
    ext = base.rsplit(".", 1)[-1] if "." in base else ""
    max_bytes = MAX_DOWNLOAD_BYTES_DNG if ext == "dng" else MAX_DOWNLOAD_BYTES
    try:
        image_bytes, _ = download_image(url, max_bytes=max_bytes)
    except Exception as exc:
        print("Download failed for", url, ":", exc)
        return url, None
    return url, _compute_phash(image_bytes)


def _compute_phash(image_bytes: bytes) -> int | None:
    """Compute perceptual hash from image bytes. Returns signed int64 or None on failure."""
    try:
        import imagehash  # lazy import — keeps scipy out of worker startup memory
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        h = imagehash.phash(img, hash_size=8)  # 64-bit hash
        # Convert hex hash to signed int64 (matches PostgreSQL bigint)
        n = int(str(h), 16)
        INT64_MAX = (1 << 63) - 1
        if n > INT64_MAX:
            n -= (1 << 64)
        return n
    except Exception as exc:
        print("pHash computation failed:", exc)
        return None


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
    if name.endswith(".dng"):
        return "dng"
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
    if ext == "dng":
        return "dng"
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
    if mime in {"image/dng", "image/x-adobe-dng"}:
        return "dng"
    if mime in {"image/jpeg", "image/jpg"}:
        return "jpeg"
    if mime == "image/png":
        return "png"

    return None


def download_image(url: str, max_bytes: int = MAX_DOWNLOAD_BYTES) -> tuple[bytes, Optional[str]]:
    """Download image bytes from URL with basic size limiting.

    Returns (bytes, content_type).
    """
    if not url or not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ValueError("Invalid image URL")

    try:
        resp = requests.get(url, stream=True, timeout=(10, 30))
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
        if total > max_bytes:
            raise ValueError(f"Image exceeds maximum allowed size of {max_bytes // (1024 * 1024)}MB")
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
        elif mime in {"image/dng", "image/x-adobe-dng"}:
            fmt = "dng"

    if fmt not in {"heic", "avif", "webp", "dng"}:
        return (
            jsonify(
                {
                    "error": "Unsupported image format. "
                    "Supported formats: HEIC, HEIF, AVIF, WEBP, DNG."
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
        elif fmt == "webp":
            jpeg_bytes = convert_webp_to_jpeg(data)
        else:  # dng
            jpeg_bytes = convert_dng_to_jpeg(data)
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

    if fmt not in {"heic", "avif", "webp", "dng"}:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Unsupported or unknown image format. "
                    "Supported conversion formats: HEIC, HEIF, AVIF, WEBP, DNG.",
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
        elif fmt == "webp":
            jpeg_bytes = convert_webp_to_jpeg(image_bytes)
        else:  # dng
            jpeg_bytes = convert_dng_to_jpeg(image_bytes)
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

    def _process_one(url: str) -> Tuple[str, str, int | None]:
        """Returns (original_url, converted_url, phash_or_None)."""
        if not url:
            return url, url, None

        base = url.split("?", 1)[0].lower()
        ext = base.rsplit(".", 1)[-1] if "." in base else ""
        needs_conversion = ext in {"heic", "heif", "avif", "webp", "dng"}

        if ext not in {"jpg", "jpeg", "png"} and not needs_conversion:
            print("Skipping unsupported format:", url)
            return url, url, None

        try:
            image_bytes, _ = download_image(url, max_bytes=MAX_DOWNLOAD_BYTES_DNG if ext == "dng" else MAX_DOWNLOAD_BYTES)
        except Exception as exc:
            print("Download failed for", url, ":", exc)
            return url, url, None

        phash_value = _compute_phash(image_bytes)

        if not needs_conversion:
            print("Skipping conversion for (already JPEG/PNG):", url)
            return url, url, phash_value

        fmt = "heic" if ext in {"heic", "heif"} else ext
        try:
            if fmt == "heic":
                jpeg_bytes = convert_heic_to_jpeg(image_bytes)
            elif fmt == "avif":
                jpeg_bytes = convert_avif_to_jpeg(image_bytes)
            elif fmt == "webp":
                jpeg_bytes = convert_webp_to_jpeg(image_bytes)
            else:
                jpeg_bytes = convert_dng_to_jpeg(image_bytes)
        except Exception as exc:
            print("Conversion error for", url, ":", exc)
            return url, url, phash_value

        if phash_value is None:
            phash_value = _compute_phash(jpeg_bytes)

        try:
            upload_info = upload_to_bunny(jpeg_bytes, content_type="image/jpeg", folder="product-images", filename="converted-batch.jpg")
            converted_url = upload_info.get("public_url") or url
            print("Conversion complete for", url, "->", converted_url)
            return url, converted_url, phash_value
        except Exception as exc:
            print("Upload failed for", url, ":", exc)
            return url, url, phash_value

    mapping: Dict[str, str] = {url: url for url in image_urls if url}
    hashes: Dict[str, int] = {}

    valid_urls = [u for u in image_urls if u]
    with ThreadPoolExecutor(max_workers=min(HASH_THREAD_WORKERS, len(valid_urls))) as pool:
        futures = {pool.submit(_process_one, url): url for url in valid_urls}
        for future in as_completed(futures):
            original_url, converted_url, phash_value = future.result()
            mapping[original_url] = converted_url
            if phash_value is not None:
                hashes[original_url] = phash_value

    return jsonify({"converted": mapping, "hashes": hashes}), 200


@image_convert_bp.route("/hash/batch", methods=["POST"])
def hash_batch_route():
    """
    Compute pHash for a batch of image URLs without converting or uploading.
    Exposed as: POST /api/convert/hash/batch

    Expected JSON body:
    {
        "imageUrls": ["https://...", "..."]
    }

    Response:
    {
        "hashes": { "original_url": 1234567890 }
    }
    """
    data = request.get_json(silent=True) or {}
    image_urls = data.get("imageUrls") or []

    if not isinstance(image_urls, list) or not all(isinstance(u, str) for u in image_urls):
        return jsonify({"hashes": {}, "error": "Field 'imageUrls' must be a list of strings"}), 400

    valid_urls = [u for u in image_urls if u]
    hashes: Dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=min(HASH_THREAD_WORKERS, len(valid_urls))) as pool:
        futures = {pool.submit(_download_and_hash, url): url for url in valid_urls}
        for future in as_completed(futures):
            url, phash_value = future.result()
            if phash_value is not None:
                hashes[url] = phash_value
                print("pHash computed for", url, ":", phash_value)
            else:
                print("pHash failed for", url)

    return jsonify({"hashes": hashes}), 200


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

        if ext not in {"heic", "heif", "avif", "webp", "dng"}:
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
            fmt = ext  # avif / webp / dng

        logger.info("Converting format %s for %s", fmt, url)

        try:
            if fmt == "heic":
                jpeg_bytes = convert_heic_to_jpeg(image_bytes)
            elif fmt == "avif":
                jpeg_bytes = convert_avif_to_jpeg(image_bytes)
            elif fmt == "webp":
                jpeg_bytes = convert_webp_to_jpeg(image_bytes)
            else:  # dng
                jpeg_bytes = convert_dng_to_jpeg(image_bytes)
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

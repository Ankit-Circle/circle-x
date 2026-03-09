from __future__ import annotations

from io import BytesIO
from typing import Literal

from PIL import Image, UnidentifiedImageError
import pillow_heif
import pillow_avif  # noqa: F401  - imported to register AVIF support with Pillow


# Register HEIF/HEIC opener with Pillow
pillow_heif.register_heif_opener()


ImageFormat = Literal["heic", "avif", "webp"]
MAX_WIDTH = 2000


class ImageConversionError(Exception):
    """Raised when an image cannot be converted to JPEG."""


def _open_image(image_bytes: bytes) -> Image.Image:
    buffer = BytesIO(image_bytes)
    try:
        img = Image.open(buffer)
        img.load()
        return img
    except UnidentifiedImageError as exc:
        raise ImageConversionError("Unable to decode image data") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise ImageConversionError("Unexpected error while decoding image") from exc


def _ensure_expected_format(img: Image.Image, expected: ImageFormat) -> None:
    fmt = (img.format or "").upper()

    if expected == "heic":
        # pillow-heif typically reports "HEIF" for HEIC/HEIF containers
        if fmt not in {"HEIC", "HEIF"}:
            raise ImageConversionError(f"Uploaded image is '{fmt or 'unknown'}', expected HEIC/HEIF")
    elif expected == "avif":
        if fmt != "AVIF":
            raise ImageConversionError(f"Uploaded image is '{fmt or 'unknown'}', expected AVIF")
    elif expected == "webp":
        if fmt != "WEBP":
            raise ImageConversionError(f"Uploaded image is '{fmt or 'unknown'}', expected WEBP")


def _convert_to_jpeg_bytes(image_bytes: bytes, expected_format: ImageFormat) -> bytes:
    """Convert image bytes to optimized JPEG bytes with basic validation."""
    img = _open_image(image_bytes)
    _ensure_expected_format(img, expected_format)

    # Resize if too wide
    if img.width > MAX_WIDTH:
        ratio = MAX_WIDTH / float(img.width)
        new_height = int(img.height * ratio)
        img = img.resize((MAX_WIDTH, new_height))

    rgb = img.convert("RGB")
    out_buffer = BytesIO()

    try:
        rgb.save(
            out_buffer,
            format="JPEG",
            quality=85,
            optimize=True,
            progressive=True,
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise ImageConversionError("Failed to encode JPEG image") from exc

    return out_buffer.getvalue()


def convert_heic_to_jpeg(image_bytes: bytes) -> bytes:
    """Convert HEIC/HEIF image bytes to JPEG bytes."""
    return _convert_to_jpeg_bytes(image_bytes, expected_format="heic")


def convert_avif_to_jpeg(image_bytes: bytes) -> bytes:
    """Convert AVIF image bytes to JPEG bytes."""
    return _convert_to_jpeg_bytes(image_bytes, expected_format="avif")


def convert_webp_to_jpeg(image_bytes: bytes) -> bytes:
    """Convert WEBP image bytes to JPEG bytes."""
    return _convert_to_jpeg_bytes(image_bytes, expected_format="webp")


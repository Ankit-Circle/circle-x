import os
import json
import requests
import logging
import time
import asyncio
import concurrent.futures
import gc  # ✅ Added missing gc import for garbage collection
from datetime import datetime
from flask import Blueprint, request, jsonify
from PIL import Image, ImageEnhance, ImageOps  # ✅ Added ImageOps for EXIF correction
from io import BytesIO
import openai

import cloudinary
import cloudinary.uploader

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger = logging.getLogger(__name__)
    logger.info("✅ Environment variables loaded from .env file")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("⚠️  python-dotenv not installed, using system environment variables")

# Configuration constants for memory management
MAX_IMAGE_DIMENSION = 8192  # Maximum width/height in pixels (8K)
MAX_IMAGE_PIXELS = 100 * 1024 * 1024  # Maximum total pixels (100MP)
MAX_FILE_SIZE = 150 * 1024 * 1024  # Maximum file size (150MB)
RESIZE_THRESHOLD = 4096  # Threshold for resizing large images (4K)
EXTREME_RESIZE_THRESHOLD = 2048  # Threshold for extremely large images (2K)

# Smart processing thresholds
SMART_PROCESSING_THRESHOLDS = {
    100 * 1024 * 1024: 2048,    # > 100MP → 2K (very aggressive)
    50 * 1024 * 1024: 4096,     # > 50MP → 4K (aggressive)
    25 * 1024 * 1024: 6144,     # > 25MP → 6K (moderate)
    16 * 1024 * 1024: 8192,     # > 16MP → 8K (light)
}

logger.info(f"Memory management config: MAX_DIM={MAX_IMAGE_DIMENSION}, MAX_PIXELS={MAX_IMAGE_PIXELS/1024/1024:.1f}MP, MAX_SIZE={MAX_FILE_SIZE/1024/1024:.1f}MB")
logger.info(f"Smart processing thresholds: {SMART_PROCESSING_THRESHOLDS}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console handler
        logging.FileHandler('image_enhancement.log', encoding='utf-8')  # File handler with UTF-8 encoding
    ]
)
logger = logging.getLogger(__name__)

image_enhancement_bp = Blueprint("image_enhancement", __name__)

openai.api_key = os.environ.get("OPENAI_API_KEY")


# Configure Cloudinary
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET")
)

# Global session for connection pooling
_global_session = None

# Global exception handler
def global_exception_handler(exc_type, exc_value, exc_traceback):
    """Global exception handler to prevent crashes"""
    if issubclass(exc_type, KeyboardInterrupt):
        # Allow keyboard interrupts to pass through
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.error("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))
    
    # Force memory cleanup
    cleanup_memory_aggressive()
    
    # Log the error but don't crash
    logger.error(f"Application recovered from uncaught exception: {exc_value}")

# Set the global exception handler
import sys
sys.excepthook = global_exception_handler

def get_global_session():
    """Get or create a global session for connection pooling"""
    global _global_session
    if _global_session is None:
        _global_session = requests.Session()
        _global_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ImageEnhancement/1.0)'
        })
        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3,
            pool_block=False
        )
        _global_session.mount('http://', adapter)
        _global_session.mount('https://', adapter)
    return _global_session

def cleanup_memory():
    """Force garbage collection to free memory"""
    import gc
    gc.collect()
    logger.debug("Memory cleanup performed")

def enhance_image_pillow(img, factors):
    """Enhance image using PIL with the given factors"""
    start_time = time.time()
    logger.info(f"Starting image enhancement - Size: {img.size}, Mode: {img.mode}")
    logger.info(f"Enhancement factors: {factors}")
    
    try:
        # Create a copy of the image to avoid modifying the original
        working_img = img.copy()
        
        # Apply saturation enhancement
        logger.debug("Applying saturation enhancement...")
        working_img = ImageEnhance.Color(working_img).enhance(factors.get("saturation", 1.0))
        
        # Apply brightness enhancement
        logger.debug("Applying brightness enhancement...")
        working_img = ImageEnhance.Brightness(working_img).enhance(factors.get("brightness", 1.0))
        
        # Apply contrast enhancement
        logger.debug("Applying contrast enhancement...")
        working_img = ImageEnhance.Contrast(working_img).enhance(factors.get("contrast", 1.0))
        
        # Apply sharpness enhancement
        logger.debug("Applying sharpness enhancement...")
        working_img = ImageEnhance.Sharpness(working_img).enhance(factors.get("sharpness", 1.0))
        
        processing_time = time.time() - start_time
        logger.info(f"Image enhancement completed in {processing_time:.2f} seconds")
        logger.info(f"Enhanced image size: {working_img.size}, Mode: {working_img.mode}")
        
        return working_img
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Error enhancing image after {processing_time:.2f} seconds: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)



def analyze_image_characteristics(img):
    """Analyze basic image characteristics to provide fallback enhancement values"""
    try:
        # Memory-efficient analysis without copying the entire image
        # Use numpy arrays for faster, memory-efficient calculations
        
        # Get image data as numpy array for analysis
        import numpy as np
        
        # Convert to RGB if needed (this creates a new object but we'll manage it carefully)
        if img.mode != 'RGB':
            analysis_img = img.convert('RGB')
        else:
            analysis_img = img
        
        # Convert to numpy array for efficient analysis
        img_array = np.array(analysis_img)
        
        # Calculate average brightness from RGB values (more accurate than L conversion)
        # Use only a sample of pixels for large images to save memory
        height, width = img_array.shape[:2]
        total_pixels = height * width
        
        # For very large images, sample pixels to save memory
        if total_pixels > 1000000:  # > 1MP
            # Sample every 4th pixel
            sample_step = 4
            sampled_array = img_array[::sample_step, ::sample_step]
            avg_brightness = np.mean(sampled_array)
            # Scale back to full image average
            avg_brightness = avg_brightness * (sample_step ** 2)
        else:
            # For smaller images, use all pixels
            avg_brightness = np.mean(img_array)
        
        # Normalize brightness (0-255 to 0-1)
        normalized_brightness = avg_brightness / 255.0
        
        # Calculate saturation using RGB values (avoid HSV conversion)
        # Saturation = max(R,G,B) - min(R,G,B)
        if total_pixels > 1000000:
            # Use sampled array for large images
            max_rgb = np.max(sampled_array, axis=2)
            min_rgb = np.min(sampled_array, axis=2)
            saturation_values = max_rgb - min_rgb
            avg_saturation = np.mean(saturation_values) / 255.0
        else:
            max_rgb = np.max(img_array, axis=2)
            min_rgb = np.min(img_array, axis=2)
            saturation_values = max_rgb - min_rgb
            avg_saturation = np.mean(saturation_values) / 255.0
        
        # Clean up numpy arrays to free memory
        del img_array
        if 'sampled_array' in locals():
            del sampled_array
        if 'max_rgb' in locals():
            del max_rgb
        if 'min_rgb' in locals():
            del min_rgb
        if 'saturation_values' in locals():
            del saturation_values
        
        # Only close if we created a new image object
        if analysis_img != img:
            analysis_img.close()
        
        # Force garbage collection for numpy arrays
        gc.collect()
        
        # Determine enhancement factors based on analysis
        if normalized_brightness < 0.4:  # Dark image
            brightness_factor = 1.1
            contrast_factor = 1.2
        elif normalized_brightness > 0.7:  # Bright image
            brightness_factor = 0.95
            contrast_factor = 1.1
        else:  # Normal brightness
            brightness_factor = 1.05
            contrast_factor = 1.15
        
        if avg_saturation < 0.3:  # Muted colors
            saturation_factor = 1.15
        elif avg_saturation > 0.7:  # Very saturated
            saturation_factor = 0.95
        else:  # Normal saturation
            saturation_factor = 1.05
        
        return {
            "brightness": brightness_factor,
            "contrast": contrast_factor,
            "saturation": saturation_factor,
            "sharpness": 1.2,
            "shadow": 1.0,
            "analyzed": True
        }
    except Exception as e:
        logger.error(f"Error analyzing image characteristics: {str(e)}")
        # Force memory cleanup on error
        gc.collect()
        return {
            "brightness": 1.05,
            "contrast": 1.15,
            "saturation": 1.1,
            "sharpness": 1.2,
            "shadow": 1.0,
            "fallback": True
        }

def get_ai_suggestions_parallel(image_url, img):
    """Get AI suggestions for image enhancement factors (optimized version)"""
    start_time = time.time()
    
    # Clean the URL to remove any trailing colons or invalid characters
    clean_image_url = image_url.rstrip(':').strip()
    if clean_image_url != image_url:
        logger.warning(f"URL cleaned from '{image_url}' to '{clean_image_url}' for AI processing")
        image_url = clean_image_url
    
    logger.info(f"Starting AI suggestions for image: {image_url}")
    
    # Check if OpenAI API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found, using image analysis for fallback values")
        # Use the already downloaded image for analysis
        try:
            fallback_factors = analyze_image_characteristics(img)
            logger.info(f"Using analyzed fallback factors (no API key): {fallback_factors}")
        except Exception as analysis_error:
            logger.error(f"Failed to analyze image for fallback: {str(analysis_error)}")
            fallback_factors = {
                "brightness": 1.05,
                "contrast": 1.15,
                "saturation": 1.1,
                "sharpness": 1.2,
                "shadow": 1.0,
                "apply_blur": False,  # Default to false for fallback
                "fallback": True
            }
            logger.info(f"Using default fallback factors (no API key): {fallback_factors}")
        
        return fallback_factors
    
    try:
        prompt = """
Analyze this specific product image and provide optimal enhancement values.

First, assess the image characteristics(this is just reference, you can use it or not):
1. If the image is dark or underexposed: increase brightness (1.05-1.15) and contrast (1.2-1.3)
2. If the image is bright or overexposed: decrease brightness (0.9-1.0) and increase contrast (1.1-1.2)
3. If the image has muted colors: increase saturation (1.1-1.2)
4. If the image has vibrant colors: keep saturation normal (1.0-1.05)
5. If the image is blurry: increase sharpness (1.3-1.5)
6. If the image is sharp: keep sharpness normal (1.1-1.2)

Return only a JSON object with these exact values:
- brightness: 0.9-1.15
- contrast: 1.0-1.3
- saturation: 1.0-1.2
- sharpness: 1.0-1.5
- shadow: 1.0

Base your values on the actual image analysis, not generic defaults.
"""
        logger.debug("Sending request to OpenAI API...")
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert image enhancement specialist. Your task is to analyze each product image individually and provide specific enhancement values based on the actual image characteristics. Be precise and avoid generic responses. Always base your recommendations on the visual analysis of the provided image."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
                    ]
                }
            ],
            temperature=0.2
        )
        
        api_time = time.time() - start_time
        logger.info(f"OpenAI API response received in {api_time:.2f} seconds")
        
        content = response.choices[0].message.content.strip()
        logger.info(f"OpenAI raw response: {content}")
        
        clean_content = content.replace("```json", "").replace("```", "").strip()
        logger.info(f"Cleaned content: {clean_content}")
        factors = json.loads(clean_content)
        
        result_factors = {
            "brightness": factors.get("brightness", 1.0),
            "contrast": factors.get("contrast", 1.0),
            "saturation": factors.get("saturation", 1.0),
            "sharpness": factors.get("sharpness", 1.0),
            "shadow": factors.get("shadow", 1.0)
        }
        
        # Check if AI response seems generic (common default values)
        generic_responses = [
            {"brightness": 1.05, "contrast": 1.2, "saturation": 1.1, "sharpness": 1.3, "shadow": 1.0},
            {"brightness": 1.05, "contrast": 1.15, "saturation": 1.1, "sharpness": 1.2, "shadow": 1.0},
            {"brightness": 1.0, "contrast": 1.1, "saturation": 1.0, "sharpness": 1.1, "shadow": 1.0}
        ]
        
        is_generic = any(
            all(abs(result_factors.get(key, 0) - generic.get(key, 0)) < 0.01 for key in ["brightness", "contrast", "saturation", "sharpness", "shadow"])
            for generic in generic_responses
        )
        
        if is_generic:
            logger.warning("AI response appears to be generic, using image analysis fallback")
            try:
                # Clean the URL to remove any trailing colons or invalid characters
                clean_image_url = image_url.rstrip(':').strip()
                if clean_image_url != image_url:
                    logger.warning(f"URL cleaned from '{image_url}' to '{clean_image_url}' for fallback analysis")
                    image_url = clean_image_url
                
                img_resp = requests.get(image_url, timeout=10)
                img_resp.raise_for_status()
                img = Image.open(BytesIO(img_resp.content))
                img = ImageOps.exif_transpose(img).convert("RGB")
                analyzed_factors = analyze_image_characteristics(img)
                logger.info(f"Using analyzed factors instead of generic AI response: {analyzed_factors}")
                return analyzed_factors
            except Exception as analysis_error:
                logger.error(f"Failed to analyze image for fallback: {str(analysis_error)}")
                logger.info(f"Keeping AI response despite being generic: {result_factors}")
        
        total_time = time.time() - start_time
        logger.info(f"AI suggestions completed in {total_time:.2f} seconds")
        logger.info(f"AI suggested factors: {result_factors}")
        
        return result_factors
        
    except json.JSONDecodeError as e:
        total_time = time.time() - start_time
        error_msg = f"JSON decode error in AI response after {total_time:.2f} seconds: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Raw response content: {content}")
        
        # Try to analyze the image for fallback values
        try:
            # Clean the URL to remove any trailing colons or invalid characters
            clean_image_url = image_url.rstrip(':').strip()
            if clean_image_url != image_url:
                logger.warning(f"URL cleaned from '{image_url}' to '{clean_image_url}' for JSON error fallback")
                image_url = clean_image_url
            
            img_resp = requests.get(image_url, timeout=10)
            img_resp.raise_for_status()
            img = Image.open(BytesIO(img_resp.content))
            img = ImageOps.exif_transpose(img).convert("RGB")
            fallback_factors = analyze_image_characteristics(img)
            logger.info(f"Using analyzed fallback factors due to JSON error: {fallback_factors}")
        except Exception as analysis_error:
            logger.error(f"Failed to analyze image for fallback: {str(analysis_error)}")
            fallback_factors = {
                "brightness": 1.05,
                "contrast": 1.15,
                "saturation": 1.1,
                "sharpness": 1.2,
                "shadow": 1.0,
                "fallback": True
            }
            logger.info(f"Using default fallback factors due to JSON error: {fallback_factors}")
        
        return fallback_factors
        
    except openai.AuthenticationError as e:
        total_time = time.time() - start_time
        error_msg = f"OpenAI authentication error after {total_time:.2f} seconds: {str(e)}"
        logger.error(error_msg)
        
        # Try to analyze the image for fallback values
        try:
            # Clean the URL to remove any trailing colons or invalid characters
            clean_image_url = image_url.rstrip(':').strip()
            if clean_image_url != image_url:
                logger.warning(f"URL cleaned from '{image_url}' to '{clean_image_url}' for authentication error fallback")
                image_url = clean_image_url
            
            img_resp = requests.get(image_url, timeout=10)
            img_resp.raise_for_status()
            img = Image.open(BytesIO(img_resp.content))
            img = ImageOps.exif_transpose(img).convert("RGB")
            fallback_factors = analyze_image_characteristics(img)
            logger.info(f"Using analyzed fallback factors due to authentication error: {fallback_factors}")
        except Exception as analysis_error:
            logger.error(f"Failed to analyze image for fallback: {str(analysis_error)}")
            fallback_factors = {
                "brightness": 1.05,
                "contrast": 1.15,
                "saturation": 1.1,
                "sharpness": 1.2,
                "shadow": 1.0,
                "fallback": True
            }
            logger.info(f"Using default fallback factors due to authentication error: {fallback_factors}")
        
        return fallback_factors
        
    except openai.RateLimitError as e:
        total_time = time.time() - start_time
        error_msg = f"OpenAI rate limit error after {total_time:.2f} seconds: {str(e)}"
        logger.error(error_msg)
        
        # Try to analyze the image for fallback values
        try:
            # Clean the URL to remove any trailing colons or invalid characters
            clean_image_url = image_url.rstrip(':').strip()
            if clean_image_url != image_url:
                logger.warning(f"URL cleaned from '{image_url}' to '{clean_image_url}' for rate limit fallback")
                image_url = clean_image_url
            
            img_resp = requests.get(image_url, timeout=10)
            img_resp.raise_for_status()
            img = Image.open(BytesIO(img_resp.content))
            img = ImageOps.exif_transpose(img).convert("RGB")
            fallback_factors = analyze_image_characteristics(img)
            logger.info(f"Using analyzed fallback factors due to rate limit: {fallback_factors}")
        except Exception as analysis_error:
            logger.error(f"Failed to analyze image for fallback: {str(analysis_error)}")
            fallback_factors = {
                "brightness": 1.05,
                "contrast": 1.15,
                "saturation": 1.1,
                "sharpness": 1.2,
                "shadow": 1.0,
                "fallback": True
            }
            logger.info(f"Using default fallback factors due to rate limit: {fallback_factors}")
        
        return fallback_factors
        
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = f"Unexpected error in AI suggestions after {total_time:.2f} seconds: {str(e)}"
        logger.error(error_msg)
        
        # Try to analyze the image for fallback values
        try:
            # Clean the URL to remove any trailing colons or invalid characters
            clean_image_url = image_url.rstrip(':').strip()
            if clean_image_url != image_url:
                logger.warning(f"URL cleaned from '{image_url}' to '{clean_image_url}' for unexpected error fallback")
                image_url = clean_image_url
            
            img_resp = requests.get(image_url, timeout=10)
            img_resp.raise_for_status()
            img = Image.open(BytesIO(img_resp.content))
            img = ImageOps.exif_transpose(img).convert("RGB")
            fallback_factors = analyze_image_characteristics(img)
            logger.info(f"Using analyzed fallback factors due to unexpected error: {fallback_factors}")
        except Exception as analysis_error:
            logger.error(f"Failed to analyze image for fallback: {str(analysis_error)}")
            fallback_factors = {
                "brightness": 1.05,
                "contrast": 1.15,
                "saturation": 1.1,
                "sharpness": 1.2,
                "shadow": 1.0,
                "fallback": True
            }
            logger.info(f"Using default fallback factors due to unexpected error: {fallback_factors}")
        
        return fallback_factors

def validate_image_dimensions(img: Image.Image) -> tuple[bool, str]:
    """Validate if image dimensions are within acceptable limits"""
    try:
        width, height = img.size
        total_pixels = width * height
        
        # Check dimension limits
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            return False, f"Dimensions {width}x{height} exceed maximum {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}"
        
        # Check pixel count limits
        if total_pixels > MAX_IMAGE_PIXELS:
            return False, f"Pixel count {total_pixels/1024/1024:.1f}MP exceeds maximum {MAX_IMAGE_PIXELS/1024/1024:.1f}MP"
        
        return True, "OK"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def process_single_image(image_url):
    """Process a single image with all enhancement steps and smart memory management"""
    try:
        # Clean the URL to remove any trailing colons or invalid characters
        clean_image_url = image_url.rstrip(':').strip()
        if clean_image_url != image_url:
            logger.warning(f"URL cleaned from '{image_url}' to '{clean_image_url}' for processing")
            image_url = clean_image_url
        
        # Download image
        session = get_global_session()
        img_resp = session.get(image_url, timeout=30)
        img_resp.raise_for_status()
        
        # Check file size before processing
        file_size = len(img_resp.content)
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"File size {file_size/1024/1024:.1f}MB exceeds maximum allowed {MAX_FILE_SIZE/1024/1024:.1f}MB")
            return {
                "image_url": image_url,
                "success": False,
                "error": f"File too large ({file_size/1024/1024:.1f}MB > {MAX_FILE_SIZE/1024/1024:.1f}MB)",
                "used_fallback": True,
                "fallback_reason": "file_too_large",
                "suggestion": f"Please resize your image to under {MAX_FILE_SIZE/1024/1024:.0f}MB or use our smart processing by reducing the file size",
                "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024)
            }
        
        # Process image with smart memory management
        try:
            # Detect and convert image format if necessary
            logger.info(f"Detecting image format for {image_url}")
            converted_bytes, detected_format = detect_and_convert_image_format(img_resp.content, image_url)
            
            if converted_bytes != img_resp.content:
                logger.info(f"Image converted from {detected_format} to JPEG format")
                img = Image.open(BytesIO(converted_bytes))
            else:
                img = Image.open(BytesIO(img_resp.content))
            
            img = ImageOps.exif_transpose(img).convert("RGB")
            
            original_size = img.size
            total_pixels = img.width * img.height
            
            # Use smart processing instead of rejection
            logger.info(f"Processing image: {original_size} ({total_pixels/1024/1024:.1f}MP)")
            
            # Smart processing with intelligent resizing
            processed_img, processing_info = process_image_smartly(img, original_size)
            
            # Clear original image from memory if it was resized
            if processing_info["resized"]:
                # Only close if it's not the same as processed_img
                if img != processed_img:
                    img.close()
                    del img
                    gc.collect()
                img = processed_img  # Use the processed image for further processing
            
            # Get AI suggestions
            factors = get_ai_suggestions_parallel(image_url, img)
            
            # Enhance image
            enhanced = enhance_image_pillow(img, factors)
            
            # Prepare response with processing information
            final_image = enhanced
            used_fallback = processing_info["resized"]
            fallback_reason = processing_info["resize_reason"] if processing_info["resized"] else None
            
            # Don't close the image yet - it's still needed for the response
            # Only close if we're not using it anymore
            if img != final_image:
                img.close()
                del img
                gc.collect()
            
            return {
                "image_url": image_url,
                "success": True,
                "final_image": final_image,
                "factors": factors,
                "used_fallback": used_fallback,
                "fallback_reason": fallback_reason,
                "processing_info": processing_info
            }
            
        except MemoryError as me:
            logger.error(f"Memory error processing {image_url}: {str(me)}")
            # Try to process with a much smaller size as fallback
            try:
                logger.info(f"Attempting fallback processing with reduced size for {image_url}")
                
                # Create a very small thumbnail for fallback processing
                fallback_img = Image.open(BytesIO(img_resp.content))
                fallback_img = ImageOps.exif_transpose(fallback_img).convert("RGB")
                fallback_img = fallback_img.resize((512, 512), Image.Resampling.BOX)
                
                # Use default enhancement factors
                factors = {
                    'brightness': 1.05,
                    'contrast': 1.15,
                    'saturation': 1.15,
                    'sharpness': 1.2,
                    'shadow': 1.0,
                    'analyzed': False
                }
                
                enhanced = enhance_image_pillow(fallback_img, factors)
                # Only close if it's different from enhanced
                if fallback_img != enhanced:
                    fallback_img.close()
                    del fallback_img
                    gc.collect()
                
                return {
                    "image_url": image_url,
                    "success": True,
                    "final_image": enhanced,
                    "factors": factors,
                    "used_fallback": True,
                    "fallback_reason": "memory_error_fallback",
                    "processing_info": {
                        "resized": True,
                        "resize_reason": "memory_error_fallback",
                        "original_size": original_size,
                        "final_size": (512, 512),
                        "pixel_reduction": 99.0
                    }
                }
                
            except Exception as fallback_error:
                logger.error(f"Fallback processing also failed for {image_url}: {str(fallback_error)}")
                return {
                    "image_url": image_url,
                    "success": False,
                    "error": f"Memory error and fallback failed: {str(me)}",
                    "used_fallback": True,
                    "fallback_reason": "memory_error_fallback_failed"
                }
        
    except Exception as e:
        logger.error(f"Error processing {image_url}: {str(e)}")
        return {
            "image_url": image_url,
            "success": False,
            "error": str(e),
            "used_fallback": True,
            "fallback_reason": "processing_failed"
        }

def process_multiple_images_parallel(image_urls, request_id):
    """Process multiple images with smart parallel/sequential processing based on image sizes"""
    start_time = time.time()
    logger.info(f"[{request_id}] Starting smart processing of {len(image_urls)} images")
    
    # Categorize images by size
    session = get_global_session()
    large_images, small_images = categorize_images_by_size(image_urls, session)
    
    if large_images and small_images:
        # Mixed batch - use optimal processing
        logger.info(f"[{request_id}] Mixed batch detected: {len(large_images)} large, {len(small_images)} small images")
        results = process_mixed_batch(large_images, small_images, request_id)
    elif large_images:
        # All large images - sequential processing
        logger.info(f"[{request_id}] All images are large, using sequential processing")
        results = []
        for i, (image_url, size) in enumerate(large_images):
            logger.info(f"[{request_id}] Processing large image {i+1}/{len(large_images)}: {image_url} ({size/1024/1024:.2f}MB)")
            
            try:
                result = safe_process_image(image_url, request_id)
                results.append(result)
            except Exception as e:
                logger.error(f"[{request_id}] Error processing large image {image_url}: {str(e)}")
                results.append({
                    "image_url": image_url,
                    "success": False,
                    "error": str(e),
                    "used_fallback": True,
                    "fallback_reason": "large_image_processing_failed"
                })
            
            cleanup_memory_aggressive()
    else:
        # All small images - parallel processing
        logger.info(f"[{request_id}] All images are small, using parallel processing")
        max_workers = min(3, len(small_images))
        logger.info(f"[{request_id}] Using {max_workers} workers for {len(small_images)} small images")
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(safe_process_image, url, request_id): url for url in image_urls}
            
            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    result = future.result(timeout=300)
                    results.append(result)
                    cleanup_memory_aggressive()
                except concurrent.futures.TimeoutError:
                    url = future_to_url[future]
                    # Clean the URL to remove any trailing colons or invalid characters
                    clean_url = url.rstrip(':').strip()
                    if clean_url != url:
                        logger.warning(f"[{request_id}] URL cleaned from '{url}' to '{clean_url}' for timeout error")
                        url = clean_url
                    
                    logger.error(f"[{request_id}] Worker timed out for {url}")
                    results.append({
                        "image_url": url,
                        "success": False,
                        "error": "Processing timeout",
                        "used_fallback": True,
                        "fallback_reason": "timeout"
                    })
                    cleanup_memory_aggressive()
                except Exception as e:
                    url = future_to_url[future]
                    # Clean the URL to remove any trailing colons or invalid characters
                    clean_url = url.rstrip(':').strip()
                    if clean_url != url:
                        logger.warning(f"[{request_id}] URL cleaned from '{url}' to '{clean_url}' for worker error")
                        url = clean_url
                    
                    logger.error(f"[{request_id}] Worker failed for {url}: {str(e)}")
                    results.append({
                        "image_url": url,
                        "success": False,
                        "error": str(e),
                        "used_fallback": True,
                        "fallback_reason": "worker_failed"
                    })
                    cleanup_memory_aggressive()
    
    # Final memory cleanup
    cleanup_memory_aggressive()
    
    total_time = time.time() - start_time
    logger.info(f"[{request_id}] Smart processing of {len(image_urls)} images completed in {total_time:.2f} seconds")
    
    return results





def process_image_parallel(img, image_url, request_id):
    """Process image with parallel operations where possible"""
    start_time = time.time()
    
    # Use global session for connection pooling
    session = get_global_session()
    
    try:
        # Clean the URL to remove any trailing colons or invalid characters
        clean_image_url = image_url.rstrip(':').strip()
        if clean_image_url != image_url:
            logger.warning(f"[{request_id}] URL cleaned from '{image_url}' to '{clean_image_url}' for parallel processing")
            image_url = clean_image_url
        
        # Step 1: Get AI suggestions (this can run while we prepare other things)
        logger.info(f"[{request_id}] Getting AI suggestions in parallel...")
        ai_start = time.time()
        factors = get_ai_suggestions_parallel(image_url, img)
        ai_time = time.time() - ai_start
        logger.info(f"[{request_id}] AI suggestions completed in {ai_time:.2f} seconds")
        
        # Step 2: Enhance image
        logger.info(f"[{request_id}] Enhancing image...")
        enhance_start = time.time()
        enhanced = enhance_image_pillow(img, factors)
        enhance_time = time.time() - enhance_start
        logger.info(f"[{request_id}] Image enhancement completed in {enhance_time:.2f} seconds")
        
        # No background removal - just use enhanced image
        final_image = enhanced
        used_fallback = False
        fallback_reason = None
        
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] Parallel processing completed in {total_time:.2f} seconds")
        
        return {
            "enhanced": enhanced,
            "final_image": final_image,
            "factors": factors,
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason,
            "processing_time": total_time
        }
        
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = f"Error in parallel processing after {total_time:.2f} seconds: {str(e)}"
        logger.error(f"[{request_id}] ERROR: {error_msg}")
        raise Exception(error_msg)

def get_image_size_from_url(image_url, session=None):
    """Get image size from URL without downloading the full image"""
    try:
        # Clean the URL to remove any trailing colons or invalid characters
        clean_image_url = image_url.rstrip(':').strip()
        if clean_image_url != image_url:
            logger.warning(f"URL cleaned from '{image_url}' to '{clean_image_url}' for size check")
            image_url = clean_image_url
        
        if session:
            response = session.head(image_url, timeout=10)
        else:
            response = requests.head(image_url, timeout=10)
        
        response.raise_for_status()
        content_length = response.headers.get('content-length')
        
        if content_length:
            return int(content_length)
        else:
            # If content-length not available, download a small chunk to estimate
            if session:
                response = session.get(image_url, stream=True, timeout=10)
            else:
                response = requests.get(image_url, stream=True, timeout=10)
            
            response.raise_for_status()
            # Read first chunk to estimate size
            chunk = response.raw.read(1024)
            response.close()
            
            # Estimate based on first chunk (rough approximation)
            return len(chunk) * 100  # Rough estimate
            
    except Exception as e:
        logger.warning(f"Could not determine size for {image_url}: {str(e)}")
        return None

def should_process_sequentially(image_urls, session=None, size_threshold=5*1024*1024):
    """Check if any images are large enough to warrant sequential processing"""
    large_images = []
    
    for url in image_urls:
        # Clean the URL to remove any trailing colons or invalid characters
        clean_url = url.rstrip(':').strip()
        if clean_url != url:
            logger.warning(f"URL cleaned from '{url}' to '{clean_url}' for sequential check")
            url = clean_url
        
        size = get_image_size_from_url(url, session)
        if size and size > size_threshold:
            large_images.append((url, size))
    
    if large_images:
        logger.info(f"Found {len(large_images)} large images (>5MB), will use sequential processing")
        for url, size in large_images:
            logger.info(f"Large image: {url} ({size/1024/1024:.2f}MB)")
        return True
    
    logger.info("All images are under size threshold, using parallel processing")
    return False

def resize_image_if_large(img: Image.Image, max_size: int = None) -> Image.Image:
    """Resizes an image if its dimensions exceed max_size with memory-efficient handling."""
    if max_size is None:
        max_size = RESIZE_THRESHOLD
        
    try:
        # Check if image dimensions exceed maximum allowed
        if img.width > MAX_IMAGE_DIMENSION or img.height > MAX_IMAGE_DIMENSION:
            logger.warning(f"Image dimensions {img.size} exceed maximum allowed {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}")
            # Force resize to maximum allowed dimensions
            max_size = min(max_size, MAX_IMAGE_DIMENSION)
        
        # Check if total pixels exceed maximum allowed
        total_pixels = img.width * img.height
        if total_pixels > MAX_IMAGE_PIXELS:
            logger.warning(f"Image pixel count {total_pixels/1024/1024:.1f}MP exceeds maximum allowed {MAX_IMAGE_PIXELS/1024/1024:.1f}MP")
            # Calculate appropriate max_size to stay under pixel limit
            aspect_ratio = img.width / img.height
            if aspect_ratio > 1:  # Landscape
                max_size = min(max_size, int((MAX_IMAGE_PIXELS / aspect_ratio) ** 0.5))
            else:  # Portrait
                max_size = min(max_size, int((MAX_IMAGE_PIXELS * aspect_ratio) ** 0.5))
        
        if img.width > max_size or img.height > max_size:
            logger.info(f"Resizing image from {img.size} to {max_size}x{max_size} to prevent memory issues.")
            
            # Force garbage collection before resize operation
            import gc
            gc.collect()
            
            # Determine the new size to maintain aspect ratio
            width, height = img.size
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            # Use a more memory-efficient resize approach
            try:
                # First try to resize directly
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Image resized to {resized_img.size}")
                
                # Clear the original image from memory only if it's different
                if img != resized_img:
                    img.close()
                    del img
                    gc.collect()
                
                return resized_img
                
            except MemoryError:
                logger.warning(f"Memory error during resize, trying alternative approach for {img.size}")
                
                # If direct resize fails, try a more conservative approach
                # Calculate a smaller intermediate size
                intermediate_size = max_size // 2
                
                if width > height:
                    intermediate_width = intermediate_size
                    intermediate_height = int(height * (intermediate_size / width))
                else:
                    intermediate_height = intermediate_size
                    intermediate_width = int(width * (intermediate_size / height))
                
                # Resize in steps to reduce memory usage
                intermediate_img = img.resize((intermediate_width, intermediate_height), Image.Resampling.BOX)
                # Only close if it's different
                if img != intermediate_img:
                    img.close()
                    del img
                    gc.collect()
                
                # Now resize to final size
                final_img = intermediate_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                # Only close if it's different
                if intermediate_img != final_img:
                    intermediate_img.close()
                    del intermediate_img
                    gc.collect()
                
                logger.info(f"Image resized to {final_img.size} using intermediate step")
                return final_img
                
    except Exception as e:
        logger.error(f"Error during image resize: {str(e)}")
        # Return original image if resize fails
        return img
    
    return img

def cleanup_memory_aggressive():
    """Force garbage collection and clear session to free memory"""
    try:
        global _global_session
        import gc
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Clear session if it exists
        if _global_session:
            try:
                _global_session.close()
            except:
                pass
            _global_session = None
        
        logger.debug("Aggressive memory cleanup performed")
        
    except Exception as e:
        logger.warning(f"Error during memory cleanup: {str(e)}")

def safe_process_image(image_url, request_id):
    """Safely process an image with comprehensive error handling"""
    try:
        # Clean the URL to remove any trailing colons or invalid characters
        clean_image_url = image_url.rstrip(':').strip()
        if clean_image_url != image_url:
            logger.warning(f"[{request_id}] URL cleaned from '{image_url}' to '{clean_image_url}' for safe processing")
            image_url = clean_image_url
        
        return process_single_image(image_url)
    except MemoryError as me:
        logger.error(f"[{request_id}] Memory error processing {image_url}: {str(me)}")
        return {
            "image_url": image_url,
            "success": False,
            "error": f"Memory error: {str(me)}",
            "used_fallback": True,
            "fallback_reason": "memory_error"
        }
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error processing {image_url}: {str(e)}")
        return {
            "image_url": image_url,
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "used_fallback": True,
            "fallback_reason": "unexpected_error"
        }

def categorize_images_by_size(image_urls, session=None, size_threshold=5*1024*1024):
    """Categorize images into large and small based on file size"""
    large_images = []
    small_images = []
    
    for url in image_urls:
        # Clean the URL to remove any trailing colons or invalid characters
        clean_url = url.rstrip(':').strip()
        if clean_url != url:
            logger.warning(f"URL cleaned from '{url}' to '{clean_url}' for categorization")
            url = clean_url
        
        size = get_image_size_from_url(url, session)
        if size and size > size_threshold:
            large_images.append((url, size))
        else:
            small_images.append((url, size))
    
    return large_images, small_images

def process_mixed_batch(large_images, small_images, request_id):
    """Process a mixed batch of large and small images optimally"""
    results = []
    
    # Process large images sequentially first
    if large_images:
        logger.info(f"[{request_id}] Processing {len(large_images)} large images sequentially")
        for i, (image_url, size) in enumerate(large_images):
            # Clean the URL to remove any trailing colons or invalid characters
            clean_image_url = image_url.rstrip(':').strip()
            if clean_image_url != image_url:
                logger.warning(f"[{request_id}] URL cleaned from '{image_url}' to '{clean_image_url}' for large image processing")
                image_url = clean_image_url
            
            logger.info(f"[{request_id}] Processing large image {i+1}/{len(large_images)}: {image_url} ({size/1024/1024:.2f}MB)")
            
            try:
                # Use safe processing with timeout
                result = safe_process_image(image_url, request_id)
                results.append(result)
            except Exception as e:
                logger.error(f"[{request_id}] Error processing large image {image_url}: {str(e)}")
                results.append({
                    "image_url": image_url,
                    "success": False,
                    "error": str(e),
                    "used_fallback": True,
                    "fallback_reason": "large_image_processing_failed"
                })
            
            # Aggressive memory cleanup after each large image
            cleanup_memory_aggressive()
    
    # Process small images in parallel
    if small_images:
        logger.info(f"[{request_id}] Processing {len(small_images)} small images in parallel")
        small_urls = [url for url, _ in small_images]
        
        max_workers = min(3, len(small_urls))
        logger.info(f"[{request_id}] Using {max_workers} workers for {len(small_urls)} small images")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(safe_process_image, url, request_id): url for url in small_urls}
            
            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    result = future.result(timeout=300)
                    results.append(result)
                    cleanup_memory_aggressive()
                except concurrent.futures.TimeoutError:
                    url = future_to_url[future]
                    logger.error(f"[{request_id}] Worker timed out for {url}")
                    results.append({
                        "image_url": url,
                        "success": False,
                        "error": "Processing timeout",
                        "used_fallback": True,
                        "fallback_reason": "timeout"
                    })
                    cleanup_memory_aggressive()
                except Exception as e:
                    url = future_to_url[future]
                    logger.error(f"[{request_id}] Worker failed for {url}: {str(e)}")
                    results.append({
                        "image_url": url,
                        "success": False,
                        "error": str(e),
                        "used_fallback": True,
                        "fallback_reason": "worker_failed"
                    })
                    cleanup_memory_aggressive()
    
    return results

def get_smart_resize_strategy(total_pixels):
    """Determine the best resize strategy based on image size"""
    try:
        # Find the appropriate threshold for this image size
        for threshold_pixels, target_size in sorted(SMART_PROCESSING_THRESHOLDS.items(), reverse=True):
            if total_pixels > threshold_pixels:
                return target_size, f"resized_for_memory_efficiency_{target_size}p"
        
        # If image is under all thresholds, no resize needed
        return None, None
        
    except Exception as e:
        logger.warning(f"Error determining resize strategy: {str(e)}")
        return EXTREME_RESIZE_THRESHOLD, "fallback_resize"

def process_image_smartly(img: Image.Image, original_size: tuple) -> tuple[Image.Image, dict]:
    """Process image with smart sizing strategy instead of rejection"""
    try:
        original_width, original_height = original_size
        total_pixels = original_width * original_height
        
        # Get smart resize strategy
        target_size, resize_reason = get_smart_resize_strategy(total_pixels)
        
        if target_size is None:
            # No resize needed
            return img, {
                "resized": False,
                "resize_reason": None,
                "original_size": original_size,
                "final_size": original_size,
                "pixel_reduction": 0
            }
        
        # Calculate new dimensions maintaining aspect ratio
        if original_width > original_height:
            new_width = target_size
            new_height = int(original_height * (target_size / original_width))
        else:
            new_height = target_size
            new_width = int(original_width * (target_size / original_height))
        
        new_size = (new_width, new_height)
        new_pixels = new_width * new_height
        
        # Calculate pixel reduction percentage
        pixel_reduction = ((total_pixels - new_pixels) / total_pixels) * 100
        
        logger.info(f"Smart processing: {original_size} ({total_pixels/1024/1024:.1f}MP) → {new_size} ({new_pixels/1024/1024:.1f}MP) - {pixel_reduction:.1f}% reduction")
        
        # Resize image
        resized_img = resize_image_if_large(img, max_size=target_size)
        
        return resized_img, {
            "resized": True,
            "resize_reason": resize_reason,
            "original_size": original_size,
            "final_size": resized_img.size,
            "pixel_reduction": pixel_reduction,
            "target_size": target_size
        }
        
    except Exception as e:
        logger.error(f"Error in smart processing: {str(e)}")
        # Return original image if smart processing fails
        return img, {
            "resized": False,
            "resize_reason": "smart_processing_failed",
            "original_size": original_size,
            "final_size": original_size,
            "pixel_reduction": 0
        }

def convert_avif_to_jpeg(image_bytes):
    """Convert AVIF image bytes to JPEG format"""
    try:
        # Try to import pillow_avif
        import pillow_avif
        logger.info("AVIF support detected via pillow_avif, converting AVIF to JPEG")
        
        # Open AVIF image directly
        img = Image.open(BytesIO(image_bytes))
        img = img.convert('RGB')
        
        # Convert to JPEG bytes
        output = BytesIO()
        img.save(output, format='JPEG', quality=95)
        output.seek(0)
        
        # Only close if it's different from output
        if img != output:
            img.close()
        return output.getvalue()
        
    except ImportError:
        logger.warning("pillow_avif not available, cannot process AVIF images")
        raise ValueError("AVIF format not supported - pillow_avif library not installed")
    except Exception as e:
        logger.error(f"Error converting AVIF to JPEG: {str(e)}")
        raise ValueError(f"Failed to convert AVIF image: {str(e)}")

def detect_and_convert_image_format(image_bytes, image_url):
    """Detect image format and convert if necessary"""
    try:
        # Clean the URL to remove any trailing colons or invalid characters
        clean_image_url = image_url.rstrip(':').strip()
        if clean_image_url != image_url:
            logger.warning(f"URL cleaned from '{image_url}' to '{clean_image_url}' for format detection")
            image_url = clean_image_url
        
        # Try to open with PIL first
        img = Image.open(BytesIO(image_bytes))
        logger.info(f"Image format detected: {img.format}, size: {img.size}")
        return image_bytes, img.format
        
    except Exception as e:
        error_msg = str(e).lower()
        
        # Check if it's an AVIF format error
        if 'cannot identify image file' in error_msg or 'avif' in error_msg:
            logger.warning(f"AVIF format detected from error: {error_msg}")
            
            # Try to convert AVIF to JPEG
            try:
                converted_bytes = convert_avif_to_jpeg(image_bytes)
                logger.info("AVIF successfully converted to JPEG")
                return converted_bytes, 'JPEG'
            except ValueError as ve:
                if "pillow_avif library not installed" in str(ve):
                    logger.error(f"AVIF format not supported: {str(ve)}")
                    raise ValueError(f"AVIF format not supported. Please install pillow_avif or convert to JPEG/PNG first: {str(ve)}")
                else:
                    logger.error(f"Failed to convert AVIF image: {str(ve)}")
                    raise ValueError(f"AVIF conversion failed: {str(ve)}")
            except Exception as conv_error:
                logger.error(f"Unexpected error converting AVIF: {str(conv_error)}")
                raise ValueError(f"AVIF conversion failed: {str(conv_error)}")
        
        # Check for other unsupported formats
        elif 'webp' in error_msg:
            logger.warning("WebP format detected, attempting conversion")
            try:
                img = Image.open(BytesIO(image_bytes))
                img = img.convert('RGB')
                
                output_buffer = BytesIO()
                img.save(output_buffer, format='JPEG', quality=95)
                output_buffer.seek(0)
                
                logger.info("WebP successfully converted to JPEG")
                return output_buffer.getvalue(), 'JPEG'
                
            except Exception as webp_error:
                logger.error(f"WebP conversion failed: {str(webp_error)}")
                raise Exception("WebP format not supported. Please convert to JPEG/PNG first.")
        
        else:
            # Unknown format error
            logger.error(f"Unknown image format error: {error_msg}")
            raise Exception(f"Unsupported image format. Error: {error_msg}")
    
    return image_bytes, 'UNKNOWN'

@image_enhancement_bp.route("/", methods=["POST"], strict_slashes=False)
def enhance():
    """Enhanced image processing endpoint with comprehensive error handling"""
    request_start_time = time.time()
    request_id = f"req_{int(time.time())}_{os.getpid()}"
    
    # Wrap the entire function in a try-catch to prevent crashes
    try:
        logger.info(f"[{request_id}] Starting image enhancement request")
        logger.info(f"[{request_id}] Request method: {request.method}")
        logger.info(f"[{request_id}] Content-Type: {request.content_type}")
        
        # Step 1: Validate request format
        logger.info(f"[{request_id}] Step 1: Validating request format...")
        if not request.is_json:
            error_msg = "Content-Type must be application/json"
            logger.error(f"[{request_id}] ERROR: {error_msg}")
            return jsonify({
                "success": False,
                "error": "Invalid request format",
                "enhanced_image_url": "https://example.com/placeholder.jpg",
                "used_fallback": True,
                "fallback_reason": "invalid_request_format",
                "request_id": request_id
            }), 200

        data = request.json
        image_url = data.get("image_url")
        if not image_url:
            error_msg = "No image_url provided"
            logger.error(f"[{request_id}] ERROR: {error_msg}")
            return jsonify({
                "success": False,
                "error": "Image URL is required",
                "enhanced_image_url": "https://example.com/placeholder.jpg",
                "used_fallback": True,
                "fallback_reason": "missing_image_url",
                "request_id": request_id
            }), 200

        if not image_url.startswith(('http://', 'https://')):
            error_msg = "Invalid image URL"
            logger.error(f"[{request_id}] ERROR: {error_msg}")
            return jsonify({
                "success": False,
                "error": "Invalid image URL format",
                "enhanced_image_url": "https://example.com/placeholder.jpg",
                "used_fallback": True,
                "fallback_reason": "invalid_image_url",
                "request_id": request_id
            }), 200

        logger.info(f"[{request_id}] Request validation passed")
        logger.info(f"[{request_id}] Image URL: {image_url}")

        # Step 2: Download image
        logger.info(f"[{request_id}] Step 2: Downloading image...")
        download_start = time.time()
        
        # Clean the URL to remove any trailing colons or invalid characters
        clean_image_url = image_url.rstrip(':').strip()
        if clean_image_url != image_url:
            logger.warning(f"[{request_id}] URL cleaned from '{image_url}' to '{clean_image_url}'")
            image_url = clean_image_url
        
        try:
            img_resp = requests.get(image_url, timeout=30)
            img_resp.raise_for_status()
            
            download_time = time.time() - download_start
            logger.info(f"[{request_id}] Image downloaded in {download_time:.2f} seconds")
            logger.info(f"[{request_id}] Image size: {len(img_resp.content):,} bytes ({len(img_resp.content)/1024/1024:.2f} MB)")
            
        except requests.RequestException as e:
            error_msg = f"Failed to download image: {str(e)}"
            logger.error(f"[{request_id}] ERROR: {error_msg}")
            # Return a placeholder response instead of error
            return jsonify({
                "success": False,
                "error": "Unable to process image at this time",
                "enhanced_image_url": "https://example.com/placeholder.jpg",
                "used_fallback": True,
                "fallback_reason": "image_download_failed",
                "request_id": request_id
            }), 200

        # Step 3: Process image
        logger.info(f"[{request_id}] Step 3: Processing image...")
        process_start = time.time()
        
        try:
            # Detect and convert image format if necessary
            logger.info(f"[{request_id}] Detecting image format...")
            converted_bytes, detected_format = detect_and_convert_image_format(img_resp.content, image_url)
            
            if converted_bytes != img_resp.content:
                logger.info(f"[{request_id}] Image converted from {detected_format} to JPEG format")
                img = Image.open(BytesIO(converted_bytes))
            else:
                img = Image.open(BytesIO(img_resp.content))
            
            # Apply EXIF orientation correction safely
            img = ImageOps.exif_transpose(img).convert("RGB")
            
            # Store original size for response
            original_size = img.size
            
            process_time = time.time() - process_start
            logger.info(f"[{request_id}] Image processed in {process_time:.2f} seconds")
            logger.info(f"[{request_id}] Image dimensions: {img.size[0]}x{img.size[1]} pixels")
            logger.info(f"[{request_id}] Image mode: {img.mode}")
            
        except Exception as e:
            error_msg = f"Invalid image format: {str(e)}"
            logger.error(f"[{request_id}] ERROR: {error_msg}")
            # Return a placeholder response instead of error
            return jsonify({
                "success": False,
                "error": "Unable to process image at this time",
                "enhanced_image_url": "https://example.com/placeholder.jpg",
                "used_fallback": True,
                "fallback_reason": "image_processing_failed",
                "request_id": request_id
            }), 200

        # Step 4: Process image with optimized parallel processing
        logger.info(f"[{request_id}] Step 4: Processing image with parallel optimization...")
        process_start = time.time()
        
        try:
            # Use the optimized parallel processing function
            result = process_image_parallel(img, image_url, request_id)
            
            final_image = result["final_image"]
            factors = result["factors"]
            used_fallback = result["used_fallback"]
            fallback_reason = result["fallback_reason"]
            processing_time = result["processing_time"]
            
            process_time = time.time() - process_start
            logger.info(f"[{request_id}] Parallel processing completed in {process_time:.2f} seconds")
            logger.info(f"[{request_id}] Enhancement factors: {factors}")
            
        except Exception as process_error:
            process_time = time.time() - process_start
            error_msg = f"Parallel processing failed after {process_time:.2f} seconds: {str(process_error)}"
            logger.error(f"[{request_id}] ERROR: {error_msg}")
            logger.info(f"[{request_id}] Falling back to sequential processing...")
            
            # Fallback to sequential processing
            try:
                # Get AI suggestions
                factors = get_ai_suggestions_parallel(image_url, img)
                
                # Enhance image
                enhanced = enhance_image_pillow(img, factors)
                
                # No background removal - just use enhanced image
                final_image = enhanced
                used_fallback = False
                fallback_reason = None
                    
            except Exception as fallback_error:
                logger.error(f"[{request_id}] Fallback processing also failed: {str(fallback_error)}")
                # Use original image as final fallback
                final_image = img
                factors = {
                    "brightness": 1.0,
                    "contrast": 1.0,
                    "saturation": 1.0,
                    "sharpness": 1.0,
                    "shadow": 1.0,
                    "fallback": True
                }
                used_fallback = True
                fallback_reason = "complete_processing_failure"

        # Step 8: Save final image
        logger.info(f"[{request_id}] Step 8: Saving final image...")
        save_start = time.time()
        
        buf = BytesIO()
        final_image.save(buf, format="JPEG", quality=100)  # Maximum quality, no compression
        buf.seek(0)
        
        save_time = time.time() - save_start
        final_size = len(buf.getvalue())
        logger.info(f"[{request_id}] Final image saved in {save_time:.2f} seconds")
        logger.info(f"[{request_id}] Final image size: {final_size:,} bytes ({final_size/1024/1024:.2f} MB)")

        # Step 9: Upload to Cloudinary
        logger.info(f"[{request_id}] Step 9: Uploading to Cloudinary...")
        upload_start = time.time()
        
        try:
            # Check if Cloudinary credentials are available
            if not all([
                os.environ.get("CLOUDINARY_CLOUD_NAME"),
                os.environ.get("CLOUDINARY_API_KEY"),
                os.environ.get("CLOUDINARY_API_SECRET")
            ]):
                logger.warning(f"[{request_id}] Cloudinary credentials not found, skipping upload")
                cloudinary_url = "https://example.com/placeholder.jpg"  # Placeholder
            else:
                upload_result = cloudinary.uploader.upload(
                    buf,
                    folder="mediaenrichment/enhanced",
                    resource_type="image",
                    format="jpg"
                )
                cloudinary_url = upload_result.get("secure_url")
                if not cloudinary_url:
                    raise Exception("Failed to get Cloudinary URL")
                
                upload_time = time.time() - upload_start
                logger.info(f"[{request_id}] Cloudinary upload completed in {upload_time:.2f} seconds")
                logger.info(f"[{request_id}] Uploaded URL: {cloudinary_url}")
                
        except Exception as e:
            upload_time = time.time() - upload_start
            error_msg = f"Failed to upload to Cloudinary: {str(e)}"
            logger.error(f"[{request_id}] ERROR: {error_msg} after {upload_time:.2f} seconds")
            # Use a placeholder URL instead of throwing error
            cloudinary_url = "https://example.com/placeholder.jpg"
            logger.info(f"[{request_id}] Using placeholder URL due to upload failure")

        # Step 10: Prepare response
        logger.info(f"[{request_id}] Step 10: Preparing response...")
        
        total_time = time.time() - request_start_time
        logger.info(f"[{request_id}] Total processing time: {total_time:.2f} seconds")
        
        # Get processing information if available
        processing_info = getattr(result, 'get', lambda x, default=None: default)('processing_info', {})
        
        response_data = {
            "success": True,
            "enhancement_factors": factors,
            "enhanced_image_url": cloudinary_url,
            "original_size": original_size,
            "enhanced_size": final_image.size,
            "processing_time": total_time,
            "request_id": request_id,
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason
        }
        
        # Add processing information if available
        if processing_info:
            response_data["processing_info"] = processing_info
            
            # Add user-friendly notes
            if processing_info.get("resized"):
                pixel_reduction = processing_info.get("pixel_reduction", 0)
                original_mp = (processing_info.get("original_size", [0, 0])[0] * processing_info.get("original_size", [0, 0])[1]) / (1024 * 1024)
                final_mp = (processing_info.get("final_size", [0, 0])[0] * processing_info.get("final_size", [0, 0])[1]) / (1024 * 1024)
                
                response_data["note"] = f"Your {original_mp:.1f}MP image was intelligently resized to {final_mp:.1f}MP for optimal processing while maintaining quality. Pixel reduction: {pixel_reduction:.1f}%"
                
                if processing_info.get("target_size"):
                    response_data["quality_note"] = f"Image optimized to {processing_info['target_size']}p resolution for best enhancement results"
            else:
                response_data["note"] = "Image processed at original resolution for maximum quality"
        
        logger.info(f"[{request_id}] Request completed successfully!")
        logger.info(f"[{request_id}] Response data: {response_data}")
        
        return jsonify(response_data)
        
    except MemoryError as me:
        total_time = time.time() - request_start_time
        error_msg = f"Memory error during processing: {str(me)}"
        logger.error(f"[{request_id}] MEMORY ERROR: {error_msg} after {total_time:.2f} seconds")
        
        # Force memory cleanup
        cleanup_memory_aggressive()
        
        # Return a graceful response
        return jsonify({
            "success": False,
            "error": "Image too large to process at this time",
            "enhanced_image_url": "https://example.com/placeholder.jpg",
            "used_fallback": True,
            "fallback_reason": "memory_error",
            "request_id": request_id,
            "processing_time": total_time
        }), 200
        
    except Exception as e:
        total_time = time.time() - request_start_time
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"[{request_id}] UNEXPECTED ERROR: {error_msg} after {total_time:.2f} seconds")
        
        # Force memory cleanup
        cleanup_memory_aggressive()
        
        # Return a graceful response
        return jsonify({
            "success": False,
            "error": "Unable to process image at this time",
            "enhanced_image_url": "https://example.com/placeholder.jpg",
            "used_fallback": True,
            "fallback_reason": "unexpected_error",
            "request_id": request_id,
            "processing_time": total_time
        }), 200

@image_enhancement_bp.route("/batch", methods=["POST"], strict_slashes=False)
def enhance_batch():
    """Enhanced endpoint for processing multiple images in parallel with comprehensive error handling"""
    request_start_time = time.time()
    request_id = f"batch_{int(time.time())}_{os.getpid()}"
    
    # Wrap the entire function in a try-catch to prevent crashes
    try:
        logger.info(f"[{request_id}] Starting batch image enhancement request")
        
        # Validate request
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Invalid request format",
                "results": [],
                "request_id": request_id
            }), 200

        data = request.json
        image_urls = data.get("image_urls", [])
        
        if not image_urls or not isinstance(image_urls, list):
            return jsonify({
                "success": False,
                "error": "image_urls array is required",
                "results": [],
                "request_id": request_id
            }), 200

        if len(image_urls) > 10:  # Limit batch size
            return jsonify({
                "success": False,
                "error": "Maximum 10 images allowed per batch",
                "results": [],
                "request_id": request_id
            }), 200

        # Validate URLs and clean them
        valid_urls = []
        for url in image_urls:
            if url and isinstance(url, str) and url.startswith(('http://', 'https://')):
                # Clean the URL to remove any trailing colons or invalid characters
                clean_url = url.rstrip(':').strip()
                if clean_url != url:
                    logger.warning(f"[{request_id}] URL cleaned from '{url}' to '{clean_url}'")
                    valid_urls.append(clean_url)
                else:
                    valid_urls.append(url)
            else:
                logger.warning(f"[{request_id}] Invalid URL in batch: {url}")

        if not valid_urls:
            return jsonify({
                "success": False,
                "error": "No valid image URLs provided",
                "results": [],
                "request_id": request_id
            }), 200

        # Categorize images by size
        try:
            large_images, small_images = categorize_images_by_size(valid_urls)
            
            if large_images:
                logger.info(f"[{request_id}] Found {len(large_images)} large images (>5MB), will use mixed processing.")
                results = process_mixed_batch(large_images, small_images, request_id)
            else:
                logger.info(f"[{request_id}] All images are under size threshold, using parallel processing for smaller images.")
                results = process_multiple_images_parallel(valid_urls, request_id)
                
        except MemoryError as me:
            logger.error(f"[{request_id}] Memory error during batch processing: {str(me)}")
            # Force memory cleanup
            cleanup_memory_aggressive()
            
            # Return error response for all images
            results = []
            for url in valid_urls:
                results.append({
                    "image_url": url,
                    "success": False,
                    "error": "Memory error during batch processing",
                    "used_fallback": True,
                    "fallback_reason": "batch_memory_error"
                })
        except Exception as e:
            logger.error(f"[{request_id}] Error during batch processing: {str(e)}")
            # Return error response for all images
            results = []
            for url in valid_urls:
                results.append({
                    "image_url": url,
                    "success": False,
                    "error": f"Batch processing error: {str(e)}",
                    "request_id": request_id
                })

        # Upload results to Cloudinary
        uploaded_results = []
        for result in results:
            if result["success"]:
                try:
                    # Save image to buffer
                    buf = BytesIO()
                    result["final_image"].save(buf, format="JPEG", quality=100)
                    buf.seek(0)
                    
                    # Upload to Cloudinary
                    if all([
                        os.environ.get("CLOUDINARY_CLOUD_NAME"),
                        os.environ.get("CLOUDINARY_API_KEY"),
                        os.environ.get("CLOUDINARY_API_SECRET")
                    ]):
                        upload_result = cloudinary.uploader.upload(
                            buf,
                            folder="mediaenrichment/enhanced",
                            resource_type="image",
                            format="jpg"
                        )
                        cloudinary_url = upload_result.get("secure_url")
                    else:
                        cloudinary_url = "https://example.com/placeholder.jpg"
                    
                    uploaded_results.append({
                        "image_url": result["image_url"],
                        "success": True,
                        "enhanced_image_url": cloudinary_url,
                        "enhancement_factors": result.get("factors", {}),
                        "used_fallback": result.get("used_fallback", False),
                        "fallback_reason": result.get("fallback_reason", None),
                        "processing_info": result.get("processing_info", {})
                    })
                    
                except Exception as e:
                    logger.error(f"[{request_id}] Failed to upload {result['image_url']}: {str(e)}")
                    uploaded_results.append({
                        "image_url": result["image_url"],
                        "success": False,
                        "error": "Upload failed",
                        "enhanced_image_url": "https://example.com/placeholder.jpg",
                        "used_fallback": True,
                        "fallback_reason": "upload_failed"
                    })
            else:
                uploaded_results.append({
                    "image_url": result["image_url"],
                    "success": False,
                    "error": result.get("error", "Processing failed"),
                    "enhanced_image_url": "https://example.com/placeholder.jpg",
                    "used_fallback": result.get("used_fallback", True),
                    "fallback_reason": result.get("fallback_reason", "processing_failed")
                })

        total_time = time.time() - request_start_time
        
        response_data = {
            "success": True,
            "results": uploaded_results,
            "total_images": len(valid_urls),
            "successful_images": len([r for r in uploaded_results if r["success"]]),
            "processing_time": total_time,
            "request_id": request_id
        }
        
        logger.info(f"[{request_id}] Batch processing completed successfully!")
        return jsonify(response_data)
        
    except MemoryError as me:
        total_time = time.time() - request_start_time
        error_msg = f"Memory error during batch processing: {str(me)}"
        logger.error(f"[{request_id}] MEMORY ERROR: {error_msg} after {total_time:.2f} seconds")
        
        # Force memory cleanup
        cleanup_memory_aggressive()
        
        # Return a graceful response
        return jsonify({
            "success": False,
            "error": "Batch too large to process at this time",
            "results": [],
            "request_id": request_id,
            "processing_time": total_time
        }), 200
        
    except Exception as e:
        total_time = time.time() - request_start_time
        error_msg = f"Unexpected batch processing error: {str(e)}"
        logger.error(f"[{request_id}] UNEXPECTED ERROR: {error_msg} after {total_time:.2f} seconds")
        
        # Force memory cleanup
        cleanup_memory_aggressive()
        
        # Return a graceful response
        return jsonify({
            "success": False,
            "error": "Unable to process batch at this time",
            "results": [],
            "request_id": request_id,
            "processing_time": total_time
        }), 200

# Call demo function when module loads
if __name__ == "__main__":
    app.run(debug=True)

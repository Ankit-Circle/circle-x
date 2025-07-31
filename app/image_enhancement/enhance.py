import os
import json
import requests
import logging
import time
import asyncio
import concurrent.futures
from datetime import datetime
from flask import Blueprint, request, jsonify
from PIL import Image, ImageEnhance, ImageFilter, ImageOps  # ✅ Added ImageOps for EXIF correction
from io import BytesIO
import openai
import replicate
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
replicate.api_token = os.environ.get("REPLICATE_API_TOKEN")

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET")
)

# Global session for connection pooling
_global_session = None

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

def enhance_image_pillow(img, factors):
    """Enhance image using PIL with the given factors"""
    start_time = time.time()
    logger.info(f"Starting image enhancement - Size: {img.size}, Mode: {img.mode}")
    logger.info(f"Enhancement factors: {factors}")
    
    try:
        # Apply saturation enhancement
        logger.debug("Applying saturation enhancement...")
        img = ImageEnhance.Color(img).enhance(factors.get("saturation", 1.0))
        
        # Apply brightness enhancement
        logger.debug("Applying brightness enhancement...")
        img = ImageEnhance.Brightness(img).enhance(factors.get("brightness", 1.0))
        
        # Apply contrast enhancement
        logger.debug("Applying contrast enhancement...")
        img = ImageEnhance.Contrast(img).enhance(factors.get("contrast", 1.0))
        
        # Apply sharpness enhancement
        logger.debug("Applying sharpness enhancement...")
        img = ImageEnhance.Sharpness(img).enhance(factors.get("sharpness", 1.0))
        
        processing_time = time.time() - start_time
        logger.info(f"Image enhancement completed in {processing_time:.2f} seconds")
        logger.info(f"Enhanced image size: {img.size}, Mode: {img.mode}")
        
        return img
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Error enhancing image after {processing_time:.2f} seconds: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def blur_background_with_mask(image, alpha_channel):
    """Blur background using alpha channel as mask"""
    start_time = time.time()
    logger.info(f"Starting background blur - Image size: {image.size}, Alpha size: {alpha_channel.size}")
    
    try:
        # Convert alpha channel to grayscale and resize
        logger.debug("Converting alpha channel to grayscale...")
        mask = alpha_channel.convert("L").resize(image.size)
        
        # Apply Gaussian blur to the image
        logger.debug("Applying Gaussian blur to image...")
        blurred = image.filter(ImageFilter.GaussianBlur(radius=5))
        
        # Composite the original image with blurred background
        logger.debug("Compositing image with blurred background...")
        result = Image.composite(image, blurred, mask)
        
        processing_time = time.time() - start_time
        logger.info(f"Background blur completed in {processing_time:.2f} seconds")
        logger.info(f"Blurred image size: {result.size}, Mode: {result.mode}")
        
        return result
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Error applying background blur after {processing_time:.2f} seconds: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def analyze_image_characteristics(img):
    """Analyze basic image characteristics to provide fallback enhancement values"""
    try:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate average brightness
        img_array = img.convert('L')  # Convert to grayscale
        avg_brightness = sum(img_array.getdata()) / len(img_array.getdata())
        brightness_factor = 255  # Normalize to 0-1 scale
        normalized_brightness = avg_brightness / brightness_factor
        
        # Calculate average saturation (simplified)
        img_hsv = img.convert('HSV')
        h, s, v = img_hsv.split()
        avg_saturation = sum(s.getdata()) / len(s.getdata()) / 255
        
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

def process_multiple_images_parallel(image_urls, request_id):
    """Process multiple images in parallel using ThreadPoolExecutor"""
    start_time = time.time()
    logger.info(f"[{request_id}] Starting parallel processing of {len(image_urls)} images")
    
    results = []
    
    def process_single_image(image_url):
        try:
            # Download image
            session = get_global_session()
            img_resp = session.get(image_url, timeout=30)
            img_resp.raise_for_status()
            
            # Process image
            img = Image.open(BytesIO(img_resp.content))
            img = ImageOps.exif_transpose(img).convert("RGB")
            
            # Get AI suggestions
            factors = get_ai_suggestions_parallel(image_url, img)
            
            # Enhance image
            enhanced = enhance_image_pillow(img, factors)
            
            # Background removal
            removed_bg = remove_background_via_replicate_optimized(enhanced, session)
            
            # Background blur
            if removed_bg.mode == "RGBA" and "A" in removed_bg.getbands():
                alpha_mask = removed_bg.getchannel("A")
                final_image = blur_background_with_mask(enhanced, alpha_mask)
                used_fallback = False
                fallback_reason = None
            else:
                final_image = enhanced
                used_fallback = True
                fallback_reason = "background_removal_no_alpha"
            
            return {
                "image_url": image_url,
                "success": True,
                "final_image": final_image,
                "factors": factors,
                "used_fallback": used_fallback,
                "fallback_reason": fallback_reason
            }
            
        except Exception as e:
            logger.error(f"[{request_id}] Error processing {image_url}: {str(e)}")
            return {
                "image_url": image_url,
                "success": False,
                "error": str(e),
                "used_fallback": True,
                "fallback_reason": "processing_failed"
            }
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(image_urls), 5)) as executor:
        future_to_url = {executor.submit(process_single_image, url): url for url in image_urls}
        
        for future in concurrent.futures.as_completed(future_to_url):
            result = future.result()
            results.append(result)
    
    total_time = time.time() - start_time
    logger.info(f"[{request_id}] Parallel processing of {len(image_urls)} images completed in {total_time:.2f} seconds")
    
    return results

def remove_background_via_replicate_optimized(image: Image.Image, session=None) -> Image.Image:
    """Optimized background removal with connection pooling"""
    start_time = time.time()
    logger.info(f"Starting optimized background removal via Replicate - Image size: {image.size}")
    
    # Check if Replicate API token is available
    if not os.environ.get("REPLICATE_API_TOKEN"):
        logger.warning("REPLICATE_API_TOKEN not found, skipping background removal")
        logger.info("Returning original image without background removal")
        return image
    
    try:
        # Prepare image for Replicate API
        logger.debug("Preparing image for Replicate API...")
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        logger.debug(f"Image prepared - Size: {len(buffered.getvalue())} bytes")

        logger.info("Sending image to Replicate for background removal...")
        api_start = time.time()
        
        response = replicate.run(
            "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
            input={"image": buffered}
        )
        
        api_time = time.time() - api_start
        logger.info(f"Replicate API response received in {api_time:.2f} seconds")

        # Parse response
        logger.debug("Parsing Replicate response...")
        if isinstance(response, str):
            mask_url = response
            logger.debug("Response is string URL")
        elif isinstance(response, list) and len(response) > 0:
            mask_url = response[0]
            logger.debug("Response is list, using first item")
        elif hasattr(response, "url"):
            mask_url = response.url() if callable(response.url) else response.url
            logger.debug("Response has URL attribute")
        elif hasattr(response, "__str__"):
            mask_url = str(response)
            logger.debug("Response converted to string")
        else:
            error_msg = "Unexpected response format from Replicate"
            logger.error(f"ERROR: {error_msg}")
            raise Exception(error_msg)

        logger.info(f"Downloading processed image from: {mask_url}")
        download_start = time.time()
        
        # Use session for connection pooling if provided
        if session:
            mask_resp = session.get(mask_url, timeout=60)
        else:
            mask_resp = requests.get(mask_url, timeout=60)
        mask_resp.raise_for_status()
        
        download_time = time.time() - download_start
        logger.info(f"Downloaded processed image in {download_time:.2f} seconds")
        
        mask_img = Image.open(BytesIO(mask_resp.content))
        
        # Validate the processed image
        if mask_img.size != image.size:
            logger.warning(f"Processed image size mismatch: expected {image.size}, got {mask_img.size}")
            logger.info("Returning original image due to size mismatch")
            return image
            
        if mask_img.mode not in ["RGBA", "RGB"]:
            logger.warning(f"Processed image has unexpected mode: {mask_img.mode}")
            logger.info("Returning original image due to unexpected mode")
            return image
        
        total_time = time.time() - start_time
        logger.info(f"Background removal completed in {total_time:.2f} seconds")
        logger.info(f"Processed image size: {mask_img.size}, Mode: {mask_img.mode}")
        
        return mask_img
        
    except replicate.AuthenticationError as e:
        total_time = time.time() - start_time
        error_msg = f"Replicate authentication error after {total_time:.2f} seconds: {str(e)}"
        logger.error(error_msg)
        logger.info("Returning original image due to authentication error")
        return image
        
    except replicate.RateLimitError as e:
        total_time = time.time() - start_time
        error_msg = f"Replicate rate limit error after {total_time:.2f} seconds: {str(e)}"
        logger.error(error_msg)
        logger.info("Returning original image due to rate limit")
        return image
        
    except requests.RequestException as e:
        total_time = time.time() - start_time
        error_msg = f"Network error during background removal after {total_time:.2f} seconds: {str(e)}"
        logger.error(error_msg)
        logger.info("Returning original image due to network error")
        return image
        
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = f"Unexpected error in background removal after {total_time:.2f} seconds: {str(e)}"
        logger.error(error_msg)
        logger.info("Returning original image due to unexpected error")
        return image

def remove_background_via_replicate(image: Image.Image) -> Image.Image:
    """Legacy function for backward compatibility"""
    return remove_background_via_replicate_optimized(image)

def process_image_parallel(img, image_url, request_id):
    """Process image with parallel operations where possible"""
    start_time = time.time()
    
    # Use global session for connection pooling
    session = get_global_session()
    
    try:
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
        
        # Step 3: Background removal with optimized session
        logger.info(f"[{request_id}] Starting background removal...")
        bg_start = time.time()
        removed_bg = remove_background_via_replicate_optimized(enhanced, session)
        bg_time = time.time() - bg_start
        logger.info(f"[{request_id}] Background removal completed in {bg_time:.2f} seconds")
        
        # Step 4: Background blur (if applicable)
        logger.info(f"[{request_id}] Processing background blur...")
        blur_start = time.time()
        
        used_fallback = False
        fallback_reason = None
        
        if removed_bg.mode == "RGBA" and "A" in removed_bg.getbands():
            logger.info(f"[{request_id}] Background removal successful, proceeding with blur")
            try:
                alpha_mask = removed_bg.getchannel("A")
                final_image = blur_background_with_mask(enhanced, alpha_mask)
                blur_time = time.time() - blur_start
                logger.info(f"[{request_id}] Background blur completed in {blur_time:.2f} seconds")
            except Exception as blur_error:
                blur_time = time.time() - blur_start
                error_msg = f"Background blur failed after {blur_time:.2f} seconds: {str(blur_error)}"
                logger.warning(f"[{request_id}] WARNING: {error_msg}")
                logger.info(f"[{request_id}] Using enhanced image as fallback for blur failure")
                final_image = enhanced
                used_fallback = True
                fallback_reason = "background_blur_failed"
        else:
            logger.warning(f"[{request_id}] Background removal returned image without alpha channel, using enhanced image as fallback")
            final_image = enhanced
            used_fallback = True
            fallback_reason = "background_removal_no_alpha"
        
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

@image_enhancement_bp.route("/", methods=["POST"], strict_slashes=False)
def enhance():
    request_start_time = time.time()
    request_id = f"req_{int(time.time())}_{os.getpid()}"
    
    logger.info(f"[{request_id}] Starting image enhancement request")
    logger.info(f"[{request_id}] Request method: {request.method}")
    logger.info(f"[{request_id}] Content-Type: {request.content_type}")
    
    try:
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
            # Apply EXIF orientation correction safely
            img = Image.open(BytesIO(img_resp.content))
            img = ImageOps.exif_transpose(img).convert("RGB")
            
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
                
                # Background removal
                removed_bg = remove_background_via_replicate(enhanced)
                
                # Background blur
                if removed_bg.mode == "RGBA" and "A" in removed_bg.getbands():
                    alpha_mask = removed_bg.getchannel("A")
                    final_image = blur_background_with_mask(enhanced, alpha_mask)
                else:
                    final_image = enhanced
                    used_fallback = True
                    fallback_reason = "background_removal_fallback"
                    
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
        
        response_data = {
            "success": True,
            "enhancement_factors": factors,
            "enhanced_image_url": cloudinary_url,
            "original_size": img.size,
            "enhanced_size": final_image.size,
            "processing_time": total_time,
            "request_id": request_id,
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason
        }
        
        logger.info(f"[{request_id}] Request completed successfully!")
        logger.info(f"[{request_id}] Response data: {response_data}")
        
        return jsonify(response_data)
        
    except Exception as e:
        total_time = time.time() - request_start_time
        error_msg = f"Internal server error: {str(e)}"
        logger.error(f"[{request_id}] ERROR: {error_msg} after {total_time:.2f} seconds")
        # Return a graceful response instead of error
        return jsonify({
            "success": False,
            "error": "Unable to process image at this time",
            "enhanced_image_url": "https://example.com/placeholder.jpg",
            "used_fallback": True,
            "fallback_reason": "internal_server_error",
            "request_id": request_id,
            "processing_time": total_time
        }), 200

@image_enhancement_bp.route("/batch", methods=["POST"], strict_slashes=False)
def enhance_batch():
    """Enhanced endpoint for processing multiple images in parallel"""
    request_start_time = time.time()
    request_id = f"batch_{int(time.time())}_{os.getpid()}"
    
    logger.info(f"[{request_id}] Starting batch image enhancement request")
    
    try:
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

        # Validate URLs
        valid_urls = []
        for url in image_urls:
            if url and isinstance(url, str) and url.startswith(('http://', 'https://')):
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

        logger.info(f"[{request_id}] Processing {len(valid_urls)} images in parallel")
        
        # Process images in parallel
        results = process_multiple_images_parallel(valid_urls, request_id)
        
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
                        "enhancement_factors": result["factors"],
                        "used_fallback": result["used_fallback"],
                        "fallback_reason": result["fallback_reason"]
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
                    "used_fallback": True,
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
        
    except Exception as e:
        total_time = time.time() - request_start_time
        error_msg = f"Batch processing error: {str(e)}"
        logger.error(f"[{request_id}] ERROR: {error_msg} after {total_time:.2f} seconds")
        
        return jsonify({
            "success": False,
            "error": "Unable to process batch at this time",
            "results": [],
            "request_id": request_id,
            "processing_time": total_time
        }), 200

if __name__ == "__main__":
    app.run(debug=True)

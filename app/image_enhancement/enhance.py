import os
import json
import requests
import logging
import time
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

def get_ai_suggestions(image_url):
    """Get AI suggestions for image enhancement factors"""
    start_time = time.time()
    logger.info(f"Starting AI suggestions for image: {image_url}")
    
    # Check if OpenAI API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found, using fallback values")
        fallback_factors = {
            "brightness": 1.05,
            "contrast": 1.15,
            "saturation": 1.1,
            "sharpness": 1.2,
            "shadow": 1.0,
            "fallback": True
        }
        logger.info(f"Using fallback enhancement factors: {fallback_factors}")
        return fallback_factors
    
    try:
        prompt = """
You are an expert ecommerce product photo enhancer. 
Given a product image, return the best enhancement values in JSON format using:
- brightness (0.5 to 1.15)
- contrast (1.0 to 1.3)
- saturation (1.0 to 1.2)
- sharpness (1.0 to 1.5)

Make the product look clear, fresh, and professional. Don't increase too much brightness.
Do not over-saturate or distort the image. Return only a JSON object, no explanation.
"""
        logger.debug("Sending request to OpenAI API...")
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional image enhancement assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
                    ]
                }
            ],
            temperature=0.3
        )
        
        api_time = time.time() - start_time
        logger.info(f"OpenAI API response received in {api_time:.2f} seconds")
        
        content = response.choices[0].message.content.strip()
        logger.debug(f"OpenAI response content: {content}")
        
        clean_content = content.replace("```json", "").replace("```", "").strip()
        factors = json.loads(clean_content)
        
        result_factors = {
            "brightness": factors.get("brightness", 1.0),
            "contrast": factors.get("contrast", 1.0),
            "saturation": factors.get("saturation", 1.0),
            "sharpness": factors.get("sharpness", 1.0),
            "shadow": factors.get("shadow", 1.0)
        }
        
        total_time = time.time() - start_time
        logger.info(f"AI suggestions completed in {total_time:.2f} seconds")
        logger.info(f"AI suggested factors: {result_factors}")
        
        return result_factors
        
    except json.JSONDecodeError as e:
        total_time = time.time() - start_time
        error_msg = f"JSON decode error in AI response after {total_time:.2f} seconds: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Raw response content: {content}")
        
        fallback_factors = {
            "brightness": 1.05,
            "contrast": 1.15,
            "saturation": 1.1,
            "sharpness": 1.2,
            "shadow": 1.0,
            "fallback": True
        }
        logger.info(f"Using fallback factors due to JSON error: {fallback_factors}")
        return fallback_factors
        
    except openai.AuthenticationError as e:
        total_time = time.time() - start_time
        error_msg = f"OpenAI authentication error after {total_time:.2f} seconds: {str(e)}"
        logger.error(error_msg)
        
        fallback_factors = {
            "brightness": 1.05,
            "contrast": 1.15,
            "saturation": 1.1,
            "sharpness": 1.2,
            "shadow": 1.0,
            "fallback": True
        }
        logger.info(f"Using fallback factors due to authentication error: {fallback_factors}")
        return fallback_factors
        
    except openai.RateLimitError as e:
        total_time = time.time() - start_time
        error_msg = f"OpenAI rate limit error after {total_time:.2f} seconds: {str(e)}"
        logger.error(error_msg)
        
        fallback_factors = {
            "brightness": 1.05,
            "contrast": 1.15,
            "saturation": 1.1,
            "sharpness": 1.2,
            "shadow": 1.0,
            "fallback": True
        }
        logger.info(f"Using fallback factors due to rate limit: {fallback_factors}")
        return fallback_factors
        
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = f"Unexpected error in AI suggestions after {total_time:.2f} seconds: {str(e)}"
        logger.error(error_msg)
        
        fallback_factors = {
            "brightness": 1.05,
            "contrast": 1.15,
            "saturation": 1.1,
            "sharpness": 1.2,
            "shadow": 1.0,
            "fallback": True
        }
        logger.info(f"Using fallback factors due to unexpected error: {fallback_factors}")
        return fallback_factors

def remove_background_via_replicate(image: Image.Image) -> Image.Image:
    start_time = time.time()
    logger.info(f"Starting background removal via Replicate - Image size: {image.size}")
    
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
        
        mask_resp = requests.get(mask_url, timeout=60)
        mask_resp.raise_for_status()
        
        download_time = time.time() - download_start
        logger.info(f"Downloaded processed image in {download_time:.2f} seconds")
        
        mask_img = Image.open(BytesIO(mask_resp.content))
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
            return jsonify({"error": error_msg}), 400

        data = request.json
        image_url = data.get("image_url")
        if not image_url:
            error_msg = "No image_url provided"
            logger.error(f"[{request_id}] ERROR: {error_msg}")
            return jsonify({"error": error_msg}), 400

        if not image_url.startswith(('http://', 'https://')):
            error_msg = "Invalid image URL"
            logger.error(f"[{request_id}] ERROR: {error_msg}")
            return jsonify({"error": error_msg}), 400

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
            return jsonify({"error": error_msg}), 400

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
            return jsonify({"error": error_msg}), 400

        # Step 4: Get AI suggestions
        logger.info(f"[{request_id}] Step 4: Getting AI suggestions...")
        ai_start = time.time()
        
        factors = get_ai_suggestions(image_url)
        
        ai_time = time.time() - ai_start
        logger.info(f"[{request_id}] AI suggestions completed in {ai_time:.2f} seconds")
        logger.info(f"[{request_id}] Enhancement factors: {factors}")

        # Step 5: Enhance image
        logger.info(f"[{request_id}] Step 5: Enhancing image...")
        enhance_start = time.time()
        
        enhanced = enhance_image_pillow(img, factors)
        
        enhance_time = time.time() - enhance_start
        logger.info(f"[{request_id}] Image enhancement completed in {enhance_time:.2f} seconds")

        # Step 6: Background removal
        logger.info(f"[{request_id}] Step 6: Background removal...")
        bg_start = time.time()
        
        removed_bg = remove_background_via_replicate(enhanced)
        
        bg_time = time.time() - bg_start
        logger.info(f"[{request_id}] Background removal completed in {bg_time:.2f} seconds")

        # Step 7: Background blur (if applicable)
        logger.info(f"[{request_id}] Step 7: Background blur...")
        blur_start = time.time()
        
        if removed_bg.mode == "RGBA" and "A" in removed_bg.getbands():
            logger.info(f"[{request_id}] Applying background blur with alpha channel")
            alpha_mask = removed_bg.getchannel("A")
            final_image = blur_background_with_mask(enhanced, alpha_mask)
        else:
            logger.info(f"[{request_id}] No alpha channel found, using enhanced image")
            final_image = enhanced

        blur_time = time.time() - blur_start
        logger.info(f"[{request_id}] Background blur completed in {blur_time:.2f} seconds")

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
            return jsonify({"error": error_msg}), 500

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
            "request_id": request_id
        }
        
        logger.info(f"[{request_id}] Request completed successfully!")
        logger.info(f"[{request_id}] Response data: {response_data}")
        
        return jsonify(response_data)
        
    except Exception as e:
        total_time = time.time() - request_start_time
        error_msg = f"Internal server error: {str(e)}"
        logger.error(f"[{request_id}] ERROR: {error_msg} after {total_time:.2f} seconds")
        return jsonify({"error": error_msg}), 500

if __name__ == "__main__":
    app.run(debug=True)
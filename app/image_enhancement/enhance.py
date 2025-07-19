import os
import json
import requests
from flask import Flask, request, jsonify
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import base64
import openai
import replicate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
openai.api_key = os.environ.get("OPENAI_API_KEY")
replicate_api_token = os.environ.get("REPLICATE_API_TOKEN")
if replicate_api_token:
    os.environ["REPLICATE_API_TOKEN"] = replicate_api_token

def enhance_image_pillow(img, factors):
    try:
        img = ImageEnhance.Color(img).enhance(factors.get("saturation", 1.0))
        img = ImageEnhance.Brightness(img).enhance(factors.get("brightness", 1.0))
        img = ImageEnhance.Contrast(img).enhance(factors.get("contrast", 1.0))
        img = ImageEnhance.Sharpness(img).enhance(factors.get("sharpness", 1.0))
        return img
    except Exception as e:
        raise Exception(f"Error enhancing image: {str(e)}")

def blur_background_with_mask(image, mask_image):
    mask = mask_image.convert("L").resize(image.size)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=5))
    result = Image.composite(image, blurred, mask)
    return result

def get_ai_suggestions(image_url):
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
        content = response.choices[0].message.content.strip()
        clean_content = content.replace("```json", "").replace("```", "").strip()
        factors = json.loads(clean_content)
        return {
            "brightness": factors.get("brightness", 1.0),
            "contrast": factors.get("contrast", 1.0),
            "saturation": factors.get("saturation", 1.0),
            "sharpness": factors.get("sharpness", 1.0)
        }
    except:
        return {
            "brightness": 1.0,
            "contrast": 1.0,
            "saturation": 1.0,
            "sharpness": 1.0,
            "fallback": True
        }

def remove_background_via_replicate(image: Image.Image) -> Image.Image:
    try:
        # Convert image to bytes for upload
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)

        print("Sending image to Replicate for background removal...")
        
        # Upload image to a temporary file host if needed, or encode base64 and use Replicate blob
        response = replicate.run(
            "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
            input={"image": buffered}
        )
        
        print(f"Replicate response type: {type(response)}")
        print(f"Replicate response: {response}")
        
        # Handle different response types
        if isinstance(response, str):
            mask_url = response
        elif isinstance(response, list) and len(response) > 0:
            mask_url = response[0]
        elif hasattr(response, "url"):
            # Check if url is a method or attribute
            if callable(response.url):
                mask_url = response.url()
            else:
                mask_url = response.url
        elif hasattr(response, "__str__"):
            # For FileOutput objects, convert to string
            mask_url = str(response)
        else:
            print(f"Unexpected response format: {response}")
            raise Exception("Unexpected response from Replicate")

        print(f"Downloading from URL: {mask_url}")

        # Download output with alpha (transparent background)
        mask_resp = requests.get(mask_url)
        mask_resp.raise_for_status()
        mask_img = Image.open(BytesIO(mask_resp.content))
        
        print(f"Background removal successful! Image mode: {mask_img.mode}")
        return mask_img
    except Exception as e:
        print(f"Background removal failed: {str(e)}")
        # Return original image if background removal fails
        return image

@app.route("/api/enhance", methods=["POST"])
def enhance():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        data = request.json
        image_url = data.get("image_url")
        if not image_url or not image_url.startswith(('http://', 'https://')):
            return jsonify({"error": "Invalid image URL"}), 400

        # Download and open image
        try:
            img_resp = requests.get(image_url, timeout=30)
            img_resp.raise_for_status()
        except requests.RequestException as e:
            return jsonify({"error": f"Failed to download image: {str(e)}"}), 400
        try:
            img = Image.open(BytesIO(img_resp.content)).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Invalid image format: {str(e)}"}), 400

        # Enhance
        factors = get_ai_suggestions(image_url)
        enhanced = enhance_image_pillow(img, factors)

        # Remove background via Replicate
        removed_bg = remove_background_via_replicate(enhanced)
        
        # Check if background removal was successful
        if removed_bg.mode == "RGBA" and "A" in removed_bg.getbands():
            alpha_mask = removed_bg.getchannel("A")
            # Blur background
            final_image = blur_background_with_mask(enhanced, alpha_mask)
        else:
            # If background removal failed, just use the enhanced image
            final_image = enhanced

        # Convert final image to base64
        buf = BytesIO()
        final_image.save(buf, format="JPEG", quality=90)
        base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        return jsonify({
            "success": True,
            "enhancement_factors": factors,
            "image_base64": base64_img,
            "original_size": img.size,
            "enhanced_size": final_image.size
        })
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)

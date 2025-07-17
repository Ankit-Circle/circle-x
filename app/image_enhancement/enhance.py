# api/enhance.py
import os
import json
import requests
from flask import Blueprint, request, jsonify
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import base64
import openai

image_enhancement_bp = Blueprint("image_enhancement", __name__, strict_slashes=False)

openai.api_key = os.environ.get("OPENAI_API_KEY")

def enhance_image_pillow(img, factors):
    """Enhance image using PIL with the given factors"""
    try:
        img = ImageEnhance.Color(img).enhance(factors.get("saturation", 1.0))
        img = ImageEnhance.Brightness(img).enhance(factors.get("brightness", 1.0))
        img = ImageEnhance.Contrast(img).enhance(factors.get("contrast", 1.0))
        img = ImageEnhance.Sharpness(img).enhance(factors.get("sharpness", 1.0))
        return img
    except Exception as e:
        raise Exception(f"Error enhancing image: {str(e)}")
def get_ai_suggestions(image_url):
    """Get AI suggestions for image enhancement factors"""
    try:
        prompt = """
You are a professional e-commerce photo editor. Analyze the uploaded product image and suggest only the necessary **lighting enhancements** to make it suitable for a platform like Amazon.
Focus strictly on:
- brightness (0.5–1.3)
- contrast (0.8–1.5)
- saturation (0.8–1.2)
- sharpness (0.8–1.5)
- shadow (0.8–1.2)
Suggest only which are required 
Only return JSON if a value needs adjustment. Do not apply over-processing. Prioritize visibility, realism, and balance.
Return a valid strict JSON object with only the needed keys.
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
        # Clean up any markdown formatting
        clean_content = content.replace("```json", "").replace("```", "").strip()
        # Parse JSON safely
        factors = json.loads(clean_content)
        # Ensure all expected keys exist with defaults
        return {
            "brightness": factors.get("brightness", 1.0),
            "contrast": factors.get("contrast", 1.0),
            "saturation": factors.get("saturation", 1.0),
            "sharpness": factors.get("sharpness", 1.0),
            "shadow": factors.get("shadow", 1.0)
        }
    except json.JSONDecodeError as e:
        # Return fallback values if JSON parsing fails
        return {
            "brightness": 1.0,
            "contrast": 1.0,
            "saturation": 1.0,
            "sharpness": 1.0,
            "shadow": 1.0,
            "fallback": True
        }
    except Exception as e:
        # Return fallback values if any error occurs
        return {
            "brightness": 1.0,
            "contrast": 1.0,
            "saturation": 1.0,
            "sharpness": 1.0,
            "shadow": 1.0,
            "fallback": True
        }

@image_enhancement_bp.route("/", methods=["POST"], strict_slashes=False)
def enhance():
    """Enhance an image using AI-suggested parameters"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        image_url = data.get("image_url")
        if not image_url:
            return jsonify({"error": "No image_url provided"}), 400
        # Validate URL
        if not image_url.startswith(('http://', 'https://')):
            return jsonify({"error": "Invalid image URL"}), 400
        # Download and process image
        try:
            img_resp = requests.get(image_url, timeout=30)
            img_resp.raise_for_status()
        except requests.RequestException as e:
            return jsonify({"error": f"Failed to download image: {str(e)}"}), 400
        try:
            img = Image.open(BytesIO(img_resp.content)).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Invalid image format: {str(e)}"}), 400
        # Get AI suggestions and enhance
        factors = get_ai_suggestions(image_url)
        enhanced = enhance_image_pillow(img, factors)
        # Convert to base64
        buf = BytesIO()
        enhanced.save(buf, format="JPEG", quality=90)
        base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        return jsonify({
            "success": True,
            "enhancement_factors": factors,
            "image_base64": base64_img,
            "original_size": img.size,
            "enhanced_size": enhanced.size
        })
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# Main entry point for local testing
if __name__ == "__main__":
    app.run(debug=True)

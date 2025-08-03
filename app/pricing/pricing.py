from flask import Flask, Blueprint, request, jsonify
from supabase import create_client, Client
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
import sys
import re

# Load environment variables from .env file
load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pricing_bp = Blueprint("pricing", __name__)
pricing_db_bp = Blueprint("pricing_db_bp", __name__)

# Environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, PERPLEXITY_API_KEY]):
    raise ValueError("Missing one or more required environment variables: SUPABASE_URL, SUPABASE_KEY, PERPLEXITY_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_price_details(brand, model):
    try:
        prompt = (
            f"Please search **Amazon.in, Flipkart, Croma, Reliance Digital, Vijay Sales** for a **brand new {brand} {model}**. "
            f"Return both the **original MRP** and the **current best online price** (lowest price across all platforms) as two separate numeric values. "
            f"Also, mention the **source/platform name** where each price was found (MRP and best online price). "
            f"Format the response exactly like this:\n"
            f"MRP: <amount> INR (Source: <source>)\n"
            f"Best Price: <amount> INR (Source: <source>)\n"
            f"If you can't find the exact MRP or best price, provide your best estimate based on recent listings in India. But only return INR values."
        )

        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant estimating product prices in India in INR."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "top_p": 1,
            "max_tokens": 150
        }

        print(f"[{datetime.now()}] Starting Perplexity API call for: {brand} {model}")
        start_time = time.time()

        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload
        )

        duration = time.time() - start_time
        print(f"[{datetime.now()}] Perplexity API call completed in {duration:.2f} seconds")

        if response.status_code != 200:
            raise ValueError(f"Perplexity API error: {response.status_code} - {response.text}")

        answer = response.json()["choices"][0]["message"]["content"]
        print(f"Perplexity response: '{answer}'")

        # Extract MRP with description in brackets
        mrp_match = re.search(r"MRP\s*[:\-]?\s*₹?\s*([\d,]+)\s*INR?\s*\(([^)]+)\)", answer, re.IGNORECASE)
        best_price_match = re.search(r"Best Price\s*[:\-]?\s*.*?([\d,]+)\s*INR?\s*\(([^)]+)\)", answer, re.IGNORECASE)

        if not mrp_match or not best_price_match:
            raise ValueError("Failed to extract prices and sources from Perplexity response.")

        mrp = float(mrp_match.group(1).replace(",", ""))
        mrp_source = mrp_match.group(2).strip()

        best_price = float(best_price_match.group(1).replace(",", ""))
        best_price_source = best_price_match.group(2).strip()

        return {
            "mrp": mrp,
            "mrp_source": mrp_source,
            "best_online_price": best_price,
            "online_price_source": best_price_source
        }

    except Exception as e:
        raise ValueError(f"Failed to fetch price from Perplexity API: {e}")

def round_to_nearest_25(value):
    remainder = value % 25
    if remainder <= 12:
        return int(value - remainder)
    else:
        return int(value + (25 - remainder))

@pricing_bp.route("/", methods=["POST"], strict_slashes=False)
def process_files():
    if request.method != 'POST':
        return jsonify({'error': 'POST method required'}), 405

    start_time = time.time()
    print(f"[{datetime.now()}] /api/pricing called")

    try:
        data = request.get_json()

        category = str(data.get('Category', '')).strip()
        sub_category = str(data.get('Sub_Category', '')).strip()
        age_years = int(data.get('Product_Age_Years', 0))
        age_months = int(data.get('Product_Age_Months', 0))
        condition = str(data.get('Condition_Tier', '')).strip()
        brand = str(data.get('brand', '')).strip()
        model = str(data.get('model', '')).strip()

        # Compute total age in years (float), if your config uses years as a float
        age = age_years + age_months / 12.0

        # For configs that only use integer years:
        age_int = int(round(age))  # Round to nearest year

        # Log input
        print(f"Processing pricing for: {brand} {model} | Category: {category} / {sub_category} | Age: {age} | Condition: {condition}")

        # Step 1: Match category and subcategory in Supabase
        category_resp = supabase.table("categories").select("*").eq("value", category).eq("subcategory", sub_category).execute()
        print("Category query response:", category_resp.data)

        if not category_resp.data:
            print("No matching category/sub-category. Falling back to default category_id=54")
            category_id = 54
        else:
            category_row = category_resp.data[0]
            if not category_row.get("is_active", False):
                return jsonify({'error': f"The category '{category}' is currently inactive."}), 400
            category_id = category_row["id"]

        # Step 2: Fetch config for the given year and category_id
        pricing_resp = supabase.table("pricing_master").select("sale_val, tier_2_depreciation, tier_3_depreciation").eq("category_id", category_id).eq("year", age_int).execute()
        if not pricing_resp.data:
            return jsonify({'error': f"No pricing config found for category '{category}' and year {age_int}."}), 404

        pricing = pricing_resp.data[0]
        sale_val = float(pricing["sale_val"])
        tier_2 = float(pricing["tier_2_depreciation"] or 0)
        tier_3 = float(pricing["tier_3_depreciation"] or 0)

        # Step 3: Get Best Online Price
        price_details = fetch_price_details(brand, model)

        best_online_price = price_details['best_online_price']
        mrp = price_details['mrp']
        mrp_source = price_details['mrp_source']
        online_price_source = price_details['online_price_source']

        # Step 4: Apply pricing logic
        price_new = round_to_nearest_25(best_online_price * sale_val)
        price_excellent = round_to_nearest_25(best_online_price * (sale_val - tier_2))
        price_verygood = round_to_nearest_25(best_online_price * (sale_val - tier_3))

        if condition in ['Mint', 'Like New']:
            price_final = price_new
        elif condition == 'Excellent':
            price_final = price_excellent
        elif condition in ['Good', 'Very Good']:
            price_final = price_verygood
        else:
            price_final = None

        # Add recommended price range
        min_price = min(price_new, price_excellent, price_verygood)
        max_price = max(price_new, price_excellent, price_verygood)

        total_time = time.time() - start_time
        print(f"[{datetime.now()}] Total pricing time: {total_time:.2f} seconds")

        return jsonify({
            'price_final': price_final if price_final is not None else None,
            'recommended_price_range': {
                'min_price': min_price,
                'max_price': max_price
            },
            'estimated_best_online_price': best_online_price,
            'best_price_source': online_price_source,
            'mrp': mrp,
            'mrp_source': mrp_source,
            'category': category,
            'sub-category': sub_category,
            'Product_Age_Years': age_years,
            'Product_Age_Months': age_months,
            'Condition_Tier': condition,
            'brand': brand,
            'model': model
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

app = Flask(__name__)
CORS(app)
app.register_blueprint(pricing_bp, url_prefix="/api/pricing")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import os
import requests
import time
from datetime import datetime
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pricing_bp = Blueprint("pricing", __name__)
pricing_db_bp = Blueprint("pricing_db_bp", __name__)

# Set your Perplexity API key
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")

if not PERPLEXITY_API_KEY:
    raise ValueError("PERPLEXITY_API_KEY environment variable is not set")

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

        import re

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

        # Log input
        print(f"Processing pricing for: {brand} {model} | Category: {category} / {sub_category} | Age: {age} | Condition: {condition}")

        # Get Best Online Price using GPT
        price_details = fetch_price_details(brand, model)

        best_online_price = price_details['best_online_price']
        mrp = price_details['mrp']
        mrp_source = price_details['mrp_source']
        online_price_source = price_details['online_price_source']

        config_path = os.path.join(BASE_DIR, 'config1.csv')
        config = pd.read_csv(config_path)

        config['category'] = config['category'].astype(str).str.strip()
        config['sub-category'] = config['sub-category'].astype(str).str.strip()
        config['Year'] = pd.to_numeric(config['Year'], errors='coerce')

        match = config[
            (config['category'] == category) &
            (config['sub-category'] == sub_category) &
            (config['Year'] == age)
        ]

        if match.empty:
            match = config[
                (config['category'] == category) &
                (config['sub-category'] == 'ALL') &
                (config['Year'] == age)
            ]

        if match.empty:
            match = config[
                (config['category'] == 'Generic') &
                (config['sub-category'] == 'ALL') &
                (config['Year'] == age)
            ]

        if match.empty:
            return jsonify({'error': 'No matching config found, even with Generic fallback'}), 404

        row = match.iloc[0]
        sale_val = float(row['sale_val'])
        tier_2 = float(row['tier_2_depreciation'])
        tier_3 = float(row['tier_3_depreciation'])

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
            'Product_Age_Years1': age,
            'Condition_Tier': condition,
            'brand': brand,
            'model': model
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@pricing_db_bp.route("/", methods=["GET"], strict_slashes=False)
def process_files_db():
    try:
        db_path = os.path.join(BASE_DIR, 'db1.csv')
        config_path = os.path.join(BASE_DIR, 'config1.csv')

        df = pd.read_csv(db_path)
        config = pd.read_csv(config_path)

        df["Best_Online_Price"] = pd.to_numeric(df["Best_Online_Price"], errors='coerce')
        config["sale_val"] = pd.to_numeric(config["sale_val"], errors='coerce')

        config_unique = config[['category', 'sub-category']].drop_duplicates()
        config_dict = pd.Series(config_unique['sub-category'].values, index=config_unique['category']).to_dict()

        df['category_new'] = np.where(df['Category'].isin(config_unique['category']), df['Category'], 'Generic')
        df['sub-category_new'] = df['category_new'].apply(lambda x: config_dict.get(x, 'Generic'))

        cols = [
            'Listing_ID', 'Category', 'Sub_Category', 'Best_Online_Price',
            'Condition_Tier', 'Product_Age_Years1', 'brand_tier',
            'warranty_period', 'Functional_Status', 'Final_Listed_Price',
            'category_new', 'sub-category_new'
        ]

        df_comb = pd.merge(
            df[cols], config,
            how='left',
            left_on=['Product_Age_Years1', 'category_new', 'sub-category_new'],
            right_on=['Year', 'category', 'sub-category']
        )

        df_comb['price_new'] = df_comb['Best_Online_Price'] * df_comb['sale_val']
        df_comb['price_excellent'] = df_comb['Best_Online_Price'] * (df_comb['sale_val'] - df_comb['tier_2_depreciation'])
        df_comb['price_verygood'] = df_comb['Best_Online_Price'] * (df_comb['sale_val'] - df_comb['tier_3_depreciation'])

        df_comb['price_final'] = np.where(
            df_comb['Condition_Tier'].isin(['Mint', 'Like New']),
            df_comb['price_new'],
            np.where(
                df_comb['Condition_Tier'].isin(['Excellent']),
                df_comb['price_excellent'],
                np.where(
                    df_comb['Condition_Tier'].isin(['Good', 'Very Good']),
                    df_comb['price_verygood'],
                    np.nan
                )
            )
        )

        output_data = df_comb[['Listing_ID', 'price_final']].to_dict(orient='records')
        return jsonify({'result': output_data})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
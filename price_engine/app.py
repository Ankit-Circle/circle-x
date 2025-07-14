from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import openai

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def fetch_best_online_price(brand, model):
    try:
        prompt = (
            f"As of today, please provide the closest and most accurate current market price in Indian Rupees (INR) "
            f"for a brand new {brand} {model}. "
            f"Respond only with the price as a number without any currency symbols or additional text."
            f"\nIf you don't know the exact price, provide your best estimate based on recent trends."
        )
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant estimating product prices in India."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,
            temperature=0.2
        )
        answer = response.choices[0].message.content.strip()
        import re
        match = re.search(r"[\d,]+(?:\.\d+)?", answer)
        if match:
            price_str = match.group().replace(',', '')
            return float(price_str)
        else:
            raise ValueError("No valid price found in GPT response.")
    except Exception as e:
        raise ValueError(f"Failed to fetch price from GPT: {e}")

@app.route('/process', methods=['POST'])
def process_files():
    if request.method != 'POST':
        return jsonify({'error': 'POST method required'}), 405

    try:
        data = request.get_json()

        category = str(data.get('Category', '')).strip()
        sub_category = str(data.get('Sub_Category', '')).strip()
        age = int(data.get('Product_Age_Years1'))
        condition = str(data.get('Condition_Tier', '')).strip()
        brand = str(data.get('brand', '')).strip()
        model = str(data.get('model', '')).strip()

        # Get Best Online Price using GPT
        best_online_price = fetch_best_online_price(brand, model)

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
                (config['Year'] == age)
            ]

        if match.empty:
            match = config[
                (config['category'] == 'Generic') &
                (config['sub-category'] == 'Generic') &
                (config['Year'] == age)
            ]

        if match.empty:
            return jsonify({'error': 'No matching config found, even with Generic fallback'}), 404

        row = match.iloc[0]
        sale_val = float(row['sale_val'])
        tier_2 = float(row['tier_2_depreciation'])
        tier_3 = float(row['tier_3_depreciation'])

        price_new = best_online_price * sale_val
        price_excellent = best_online_price * (sale_val - tier_2)
        price_verygood = best_online_price * (sale_val - tier_3)

        if condition in ['Mint', 'Like New']:
            price_final = price_new
        elif condition == 'Excellent':
            price_final = price_excellent
        elif condition in ['Good', 'Very Good']:
            price_final = price_verygood
        else:
            price_final = None

        # Add recommended price range
        min_price = round(min(price_new, price_excellent, price_verygood), 2)
        max_price = round(max(price_new, price_excellent, price_verygood), 2)

        return jsonify({
            'price_final': round(price_final, 2) if price_final is not None else None,
            'recommended_price_range': {
                'min_price': min_price,
                'max_price': max_price
            },
            'estimated_best_online_price': round(best_online_price, 2),
            'category': category,
            'sub-category': sub_category,
            'Product_Age_Years1': age,
            'Condition_Tier': condition,
            'brand': brand,
            'model': model
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process-db', methods=['GET'])
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

if __name__ == '__main__':
    app.run(debug=True)

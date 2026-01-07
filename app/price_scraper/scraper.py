"""
Brand and Model scraper blueprint for Cashify.
Scrapes brand pages to extract models.
"""
from flask import Blueprint, jsonify, request
from flask_cors import cross_origin

from app.price_scraper.scraper_utils import scrape_cashify_all_brand_models

price_scraper_bp = Blueprint("price_scraper", __name__)

# List of brand URLs to scrape
BRAND_URLS = [
    # 'https://www.cashify.in/sell-old-mobile-phone/sell-apple',
    'https://www.cashify.in/sell-old-mobile-phone/sell-xiaomi',
    # 'https://www.cashify.in/sell-old-mobile-phone/sell-samsung',
    # 'https://www.cashify.in/sell-old-mobile-phone/sell-oneplus',
    # 'https://www.cashify.in/sell-old-mobile-phone/sell-nokia',
    # 'https://www.cashify.in/sell-old-mobile-phone/sell-poco',
    # 'https://www.cashify.in/sell-old-mobile-phone/sell-vivo',
    # 'https://www.cashify.in/sell-old-mobile-phone/sell-oppo',
    # 'https://www.cashify.in/sell-old-mobile-phone/sell-realme',
    # 'https://www.cashify.in/sell-old-mobile-phone/sell-motorola',
    # 'https://www.cashify.in/sell-old-mobile-phone/sell-lenovo',
    # 'https://www.cashify.in/sell-old-mobile-phone/sell-honor',
    # 'https://www.cashify.in/sell-old-mobile-phone/sell-google',
    # 'https://www.cashify.in/sell-old-mobile-phone/sell-infinix',
    # 'https://www.cashify.in/sell-old-mobile-phone/sell-tecno',
    # 'https://www.cashify.in/sell-old-mobile-phone/sell-iqoo',
    # 'https://www.cashify.in/sell-old-mobile-phone/sell-nothing',
]


@price_scraper_bp.route("/", methods=["GET", "POST", "OPTIONS"], strict_slashes=False)
@cross_origin()
def scrape_brands_and_models():
    """Scrape brand pages and extract models. Returns Brand -> Model table."""
    # Handle OPTIONS request for CORS
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    print("Starting to scrape brand pages to extract models...")

    try:
        # Get brand URLs from request or use default
        brand_urls = request.json.get('brand_urls', BRAND_URLS) if request.is_json else BRAND_URLS

        # Scrape all brand pages to get model URLs
        result = scrape_cashify_all_brand_models(brand_urls)

        if not result['success']:
            return jsonify({
                'error': 'Failed to scrape brand pages',
                'success': False
            }), 500

        print(f"Found {result['total_models']} total model URLs")

        # Build Brand -> Model table structure
        brand_model_table = []
        for brand_result in result['models']:
            brand_name = brand_result['brand_name']
            models = []
            
            for model in brand_result['model_urls']:
                models.append({
                    'name': model['name'],
                    'url': model['url'],
                })
            
            brand_model_table.append({
                'brand': brand_name,
                'brand_url': brand_result['brand_url'],
                'models': models,
                'model_count': len(models),
            })

        print(f"Completed scraping. Found {len(brand_model_table)} brands with {result['total_models']} total models.")

        return jsonify({
            'success': True,
            'total_brands': len(brand_model_table),
            'total_models': result['total_models'],
            'brand_model_table': brand_model_table,
        }), 200

    except Exception as error:
        print(f"Error in brand-model scraper: {error}")
        error_message = str(error)
        return jsonify({
            'error': error_message,
            'success': False
        }), 500


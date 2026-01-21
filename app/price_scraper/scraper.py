"""
Brand and Model scraper blueprint for Cashify.
Scrapes brand pages to extract models.
"""
import os
import time
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

from flask import Blueprint, jsonify, request
from flask_cors import cross_origin
from supabase import create_client, Client

from app.price_scraper.scraper_utils import scrape_cashify_all_brand_models

# Load environment variables from .env file
load_dotenv()

price_scraper_bp = Blueprint("price_scraper", __name__)

# Environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY]):
    raise ValueError("Missing required environment variables: SUPABASE_URL, SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_brands_from_database() -> List[str]:
    """Fetch brand URLs (cf_link) from market_brands table.
    
    Returns a list of cf_link values where cf_link is not null.
    For testing: Limited to 1 brand.
    """
    try:
        # Fetch all brands and filter for non-null cf_link values
        response = supabase.table("market_brands").select("cf_link").execute()
        
        if not response.data:
            print("No brands found in database")
            return []
        
        # Filter for non-null cf_link values
        brand_urls = [brand["cf_link"] for brand in response.data if brand.get("cf_link")]
        
        # TESTING: Limit to 1 brand only
        # TEST_LIMIT = 1
        # if len(brand_urls) > TEST_LIMIT:
        #     brand_urls = brand_urls[:TEST_LIMIT]
        #     print(f"[TEST MODE] Limited to {TEST_LIMIT} brand for testing")
        
        print(f"Fetched {len(brand_urls)} brand URLs from database (out of {len(response.data)} total brands)")
        return brand_urls
    
    except Exception as error:
        print(f"Error fetching brands from database: {error}")
        raise


def get_brand_info_by_cf_link(cf_link: str) -> Optional[Dict[str, Any]]:
    """Get brand_id and name from market_brands table by matching cf_link.
    
    Returns a dict with 'id' and 'name' if found, None otherwise.
    """
    try:
        response = supabase.table("market_brands").select("id, name").eq("cf_link", cf_link).execute()
        
        if response.data and len(response.data) > 0:
            return {
                'id': response.data[0]["id"],
                'name': response.data[0]["name"]
            }
        return None
    except Exception as error:
        print(f"Error fetching brand info for cf_link {cf_link}: {error}")
        return None


def clean_model_name(model_name: str, brand_name: str) -> str:
    """Remove brand name from model name.
    
    Removes the brand name from the beginning of the model name if present.
    Handles various formats and separators.
    """
    if not model_name or not brand_name:
        return model_name.strip() if model_name else ""
    
    model_name_clean = model_name.strip()
    brand_name_clean = brand_name.strip()
    
    # Convert to lowercase for comparison
    model_lower = model_name_clean.lower()
    brand_lower = brand_name_clean.lower()
    
    # Check if model name starts with brand name (case-insensitive)
    if model_lower.startswith(brand_lower):
        # Get the length of brand name (in lowercase, which matches the start)
        brand_length = len(brand_lower)
        
        # Remove brand name from the beginning (preserving original case of remaining part)
        remaining = model_name_clean[brand_length:].strip()
        
        # Remove leading separators (spaces, hyphens, etc.)
        remaining = remaining.lstrip(" -")
        
        # Debug: log the cleaning
        if model_name_clean != remaining:
            print(f"  Cleaned: '{model_name_clean}' -> '{remaining}' (removed brand: '{brand_name_clean}')")
        
        # If we removed the brand name and there's still content, return it
        if remaining:
            return remaining
        # If nothing left after cleaning, return original (shouldn't happen, but safety check)
        return model_name_clean
    
    return model_name_clean


def store_models_in_database(brand_model_table: List[Dict]) -> Dict[str, int]:
    """Store models in market_models table using batch operations.
    
    Args:
        brand_model_table: List of brand-model data from scraping
        
    Returns:
        Dictionary with counts: {'inserted': count, 'updated': count, 'errors': count}
    """
    CATEGORY_ID = 81  # Hardcoded category_id
    SOURCE = "cashify"  # Source identifier
    BATCH_SIZE = 100  # Insert models in batches to avoid timeout
    
    stats = {'inserted': 0, 'updated': 0, 'errors': 0, 'skipped': 0}
    
    print(f"Starting to store models in database (category_id={CATEGORY_ID}, batch_size={BATCH_SIZE})...")
    
    # Cache brand info lookups to avoid repeated database queries
    brand_info_cache = {}
    
    # Collect all models to insert
    all_models_to_insert = []
    
    for brand_data in brand_model_table:
        brand_url = brand_data.get('brand_url')
        models = brand_data.get('models', [])
        
        # Get brand_id and name from database (with caching)
        if brand_url not in brand_info_cache:
            brand_info_cache[brand_url] = get_brand_info_by_cf_link(brand_url)
        brand_info = brand_info_cache[brand_url]
        
        if not brand_info:
            print(f"Warning: Could not find brand info for brand_url: {brand_url}. Skipping {len(models)} models.")
            stats['skipped'] += len(models)
            continue
        
        brand_id = brand_info['id']
        brand_name_from_db = brand_info['name']  # Use actual brand name from database
        
        print(f"Preparing {len(models)} models for brand: {brand_name_from_db} (brand_id: {brand_id})")
        
        for model in models:
            try:
                model_name = model.get('name', '').strip()
                model_url = model.get('url', '').strip()
                
                if not model_name or not model_url:
                    stats['skipped'] += 1
                    continue
                
                # Clean model name by removing brand name (use database brand name)
                cleaned_model_name = clean_model_name(model_name, brand_name_from_db)
                
                if not cleaned_model_name:
                    stats['skipped'] += 1
                    continue
                
                # Prepare model data
                model_data = {
                    'brand_id': brand_id,
                    'category_id': CATEGORY_ID,
                    'name': cleaned_model_name,
                    'source': SOURCE,
                    'cf_link': model_url,
                }
                
                all_models_to_insert.append(model_data)
                    
            except Exception as error:
                print(f"Error preparing model {model.get('name', 'unknown')}: {error}")
                stats['errors'] += 1
    
    print(f"Prepared {len(all_models_to_insert)} models for batch insertion")
    
    # Insert models in batches using upsert
    # Upsert will: Insert if model doesn't exist, Update if model exists (based on unique constraint)
    # The unique constraint is: (brand_id, name, source)
    # This ensures we don't create duplicates and can update cf_link if it changes
    for i in range(0, len(all_models_to_insert), BATCH_SIZE):
        batch = all_models_to_insert[i:i + BATCH_SIZE]
        batch_num = i//BATCH_SIZE + 1
        try:
            # Upsert with on_conflict handles the unique constraint
            # Format: on_conflict="col1,col2,col3" (comma-separated string, not a list)
            response = supabase.table("market_models").upsert(
                batch,
                on_conflict="brand_id,name,source"
            ).execute()
            
            if response.data:
                stats['inserted'] += len(response.data)
                print(f"Batch {batch_num}: Upserted {len(batch)} models ({i+1}-{min(i+BATCH_SIZE, len(all_models_to_insert))}/{len(all_models_to_insert)})")
            else:
                stats['errors'] += len(batch)
                print(f"Batch {batch_num}: No data returned for {len(batch)} models")
                
        except Exception as error:
            error_msg = str(error)
            # Check if it's a constraint error (model already exists)
            if "duplicate key" in error_msg.lower() or "unique constraint" in error_msg.lower():
                # If batch insert fails due to duplicates, try individual inserts
                # to identify which ones are new vs existing
                print(f"Batch {batch_num}: Duplicate detected, processing individually...")
                inserted_count = 0
                for model_data in batch:
                    try:
                        single_response = supabase.table("market_models").insert([model_data]).execute()
                        if single_response.data:
                            inserted_count += 1
                    except Exception:
                        # This model already exists - skip it
                        pass
                stats['inserted'] += inserted_count
                if inserted_count < len(batch):
                    print(f"Batch {batch_num}: Inserted {inserted_count}/{len(batch)} new models ({len(batch) - inserted_count} already existed)")
            else:
                print(f"Error upserting batch {batch_num}: {error}")
                stats['errors'] += len(batch)
    
    print(f"Database storage complete: {stats['inserted']} models stored, {stats['skipped']} skipped, {stats['errors']} errors")
    return stats


@price_scraper_bp.route("/", methods=["GET", "POST", "OPTIONS"], strict_slashes=False)
@cross_origin()
def scrape_brands_and_models():
    """Scrape brand pages and extract models. Returns Brand -> Model table."""
    # Handle OPTIONS request for CORS
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    print("Starting to scrape brand pages to extract models...")
    
    start_time = time.time()

    try:
        # Get brand URLs from request, or fetch from database, or use empty list
        if request.is_json and request.json.get('brand_urls'):
            # Use brand URLs from request if provided
            brand_urls = request.json.get('brand_urls')
            print(f"Using {len(brand_urls)} brand URLs from request")
        else:
            # Fetch brand URLs from database
            brand_urls = fetch_brands_from_database()
            
            if not brand_urls:
                return jsonify({
                    'error': 'No brand URLs found in database. Please add brands with cf_link to market_brands table.',
                    'success': False
                }), 404

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

        # Store models in database
        storage_stats = store_models_in_database(brand_model_table)
        
        elapsed_time = time.time() - start_time
        print(f"Completed scraping. Found {len(brand_model_table)} brands with {result['total_models']} total models.")
        print(f"Total time taken to scrape all pages: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

        return jsonify({
            'success': True,
            'total_brands': len(brand_model_table),
            'total_models': result['total_models'],
            'brand_model_table': brand_model_table,
            'scraping_time_seconds': round(elapsed_time, 2),
            'database_stats': storage_stats,
        }), 200

    except Exception as error:
        print(f"Error in brand-model scraper: {error}")
        error_message = str(error)
        return jsonify({
            'error': error_message,
            'success': False
        }), 500


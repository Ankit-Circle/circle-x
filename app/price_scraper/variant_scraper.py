"""
Variant and Price scraper blueprint for Cashify.
Scrapes model pages to extract variants and prices.
"""
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

from flask import Blueprint, jsonify, request
from flask_cors import cross_origin
from supabase import create_client, Client

from app.price_scraper.variant_scraper_utils import (
    scrape_model_page,
    scrape_variant_page,
)

# Load environment variables from .env file
load_dotenv()

variant_scraper_bp = Blueprint("variant_scraper", __name__)

# Environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY]):
    raise ValueError("Missing required environment variables: SUPABASE_URL, SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_models_from_database(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch models from market_models table.
    
    Args:
        limit: Optional limit on number of models to fetch (for testing)
    
    Returns:
        List of model records with id, brand_id, name, cf_link
    """
    try:
        # Fetch all models and filter for non-null cf_link values
        response = supabase.table("market_models").select("id, brand_id, name, cf_link").execute()
        
        if not response.data:
            print("No models found in database")
            return []
        
        # Filter for non-null cf_link values
        models = [model for model in response.data if model.get("cf_link")]
        
        # Apply limit if specified
        if limit and len(models) > limit:
            models = models[:limit]
            print(f"[TEST MODE] Limited to {limit} models for testing")
        
        print(f"Fetched {len(models)} models from database (out of {len(response.data)} total models)")
        return models
    
    except Exception as error:
        print(f"Error fetching models from database: {error}")
        raise


def store_variants_in_database(variants: List[Dict[str, Any]]) -> Dict[str, int]:
    """Store variants in market_variants table.
    
    Args:
        variants: List of variant data to store
        
    Returns:
        Dictionary with counts: {'inserted': count, 'errors': count}
    """
    SOURCE = "cashify"  # Source identifier
    BATCH_SIZE = 100  # Insert variants in batches
    
    stats = {'inserted': 0, 'errors': 0, 'skipped': 0}
    
    print(f"Starting to store {len(variants)} variants in database (batch_size={BATCH_SIZE})...")
    
    # Filter out variants with missing required data
    valid_variants = []
    for variant in variants:
        if not variant.get('model_id') or not variant.get('cf_link'):
            stats['skipped'] += 1
            continue
        if variant.get('price') is None:
            stats['skipped'] += 1
            continue
        # name can be None (for single variant cases where extraction fails)
        # but we'll still store it - the table allows null for name in some cases
        # However, the constraint requires name to be not null, so we'll skip if name is missing
        if not variant.get('name'):
            stats['skipped'] += 1
            print(f"Skipping variant with missing name for model_id {variant.get('model_id')}")
            continue
        valid_variants.append(variant)
    
    print(f"Prepared {len(valid_variants)} valid variants for batch insertion")
    
    if not valid_variants:
        print("No valid variants to store")
        return stats
    
    # Insert variants in batches using upsert
    # Note: Adjust the on_conflict constraint based on your actual table structure
    # Common constraint would be: (model_id, variant_name, source) or (model_id, cf_link)
    for i in range(0, len(valid_variants), BATCH_SIZE):
        batch = valid_variants[i:i + BATCH_SIZE]
        batch_num = i//BATCH_SIZE + 1
        try:
            # Upsert with on_conflict
            # Unique constraint is: (model_id, name, source)
            response = supabase.table("market_variants").upsert(
                batch,
                on_conflict="model_id,name,source"
            ).execute()
            
            if response.data:
                stats['inserted'] += len(response.data)
                print(f"Batch {batch_num}: Upserted {len(batch)} variants ({i+1}-{min(i+BATCH_SIZE, len(valid_variants))}/{len(valid_variants)})")
            else:
                stats['errors'] += len(batch)
                print(f"Batch {batch_num}: No data returned for {len(batch)} variants")
                
        except Exception as error:
            error_msg = str(error)
            print(f"Error upserting batch {batch_num}: {error_msg}")
            # Try individual inserts if batch fails
            inserted_count = 0
            for variant_data in batch:
                try:
                    single_response = supabase.table("market_variants").insert([variant_data]).execute()
                    if single_response.data:
                        inserted_count += 1
                except Exception:
                    # Duplicate or other error - skip
                    pass
            stats['inserted'] += inserted_count
            if inserted_count < len(batch):
                stats['errors'] += (len(batch) - inserted_count)
    
    print(f"Database storage complete: {stats['inserted']} variants stored, {stats['skipped']} skipped, {stats['errors']} errors")
    return stats


def process_model(model: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process a single model: scrape variants/prices and return variant data.
    
    Args:
        model: Model record from database with id, brand_id, name, cf_link
        
    Returns:
        List of variant data dictionaries ready for database insertion
    """
    model_id = model['id']
    model_url = model['cf_link']
    SOURCE = "cashify"
    
    variants_to_store = []
    
    try:
        # Scrape the model page
        model_result = scrape_model_page(model_url)
        
        if not model_result['success']:
            print(f"Failed to scrape model {model_id}: {model_result.get('error')}")
            return variants_to_store
        
        if model_result['has_variants']:
            # Case 1: Multiple variants - visit each variant page
            variant_urls = model_result['variant_urls']
            print(f"Processing {len(variant_urls)} variants for model {model_id}")
            
            for variant_url in variant_urls:
                # Small delay to avoid rate limiting
                time.sleep(0.5)
                
                variant_result = scrape_variant_page(variant_url)
                
                if variant_result['success'] and variant_result['price'] is not None:
                    # variant_name is required (table column 'name' is NOT NULL)
                    variant_name = variant_result['variant_name']
                    if not variant_name:
                        print(f"Skipping variant {variant_url}: variant name is missing")
                        continue
                    
                    variant_data = {
                        'model_id': model_id,
                        'name': variant_name,  # Table column is 'name', not 'variant_name'
                        'price': variant_result['price'],
                        'cf_link': variant_result['cf_link'],
                        'source': SOURCE,
                        'currency': 'INR',  # Default currency
                    }
                    variants_to_store.append(variant_data)
                else:
                    print(f"Failed to scrape variant {variant_url}: {variant_result.get('error')}")
        else:
            # Case 2: Single variant - price is directly on model page
            if model_result['price'] is not None:
                # variant_name is required (table column 'name' is NOT NULL)
                variant_name = model_result['variant_name']
                if not variant_name:
                    print(f"Skipping model {model_id}: variant name is missing")
                else:
                    variant_data = {
                        'model_id': model_id,
                        'name': variant_name,  # Table column is 'name', not 'variant_name'
                        'price': model_result['price'],
                        'cf_link': model_url,  # Use model URL as cf_link for single variant
                        'source': SOURCE,
                        'currency': 'INR',  # Default currency
                    }
                    variants_to_store.append(variant_data)
            else:
                print(f"No price found for model {model_id}")
        
    except Exception as error:
        print(f"Error processing model {model_id}: {error}")
    
    return variants_to_store


@variant_scraper_bp.route("/", methods=["GET", "POST", "OPTIONS"], strict_slashes=False)
@cross_origin()
def scrape_variants_and_prices():
    """Scrape variants and prices from model pages."""
    # Handle OPTIONS request for CORS
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    print("Starting to scrape variants and prices from model pages...")
    
    start_time = time.time()
    
    try:
        # Get limit from request or use default
        limit = None
        if request.is_json and request.json.get('limit'):
            limit = int(request.json.get('limit'))
            print(f"Using limit: {limit} models")
        
        # TESTING: Hardcode limit to 1 for testing
        # TEST_LIMIT = 10
        # if limit is None:
        #     limit = TEST_LIMIT
        #     print(f"[TEST MODE] Limited to {TEST_LIMIT} model for testing")
        
        # Fetch models from database
        models = fetch_models_from_database(limit=limit)
        
        if not models:
            return jsonify({
                'error': 'No models found in database with cf_link.',
                'success': False
            }), 404
        
        print(f"Processing {len(models)} models...")
        
        # Thread-safe counters for tracking progress
        variants_lock = threading.Lock()
        all_variants = []  # Keep track of all variants for final stats
        variants_buffer = []  # Buffer for batch insertion (50 variants)
        processed = 0
        failed = 0
        BATCH_INSERT_SIZE = 50  # Insert every 50 variants
        
        # Cumulative database stats
        cumulative_stats = {'inserted': 0, 'errors': 0, 'skipped': 0}
        
        # Flag to control logging thread
        logging_active = threading.Event()
        logging_active.set()
        
        def log_progress():
            """Periodically log progress every 5 seconds."""
            while logging_active.is_set():
                time.sleep(5)
                if logging_active.is_set():  # Check again after sleep
                    with variants_lock:
                        total_variants = len(all_variants)
                        total_failed = failed
                    print(f"===>> total {total_variants} variants added till now")
                    print(f"===> {total_failed} failed till now")
        
        # Start progress logging thread
        log_thread = threading.Thread(target=log_progress, daemon=True)
        log_thread.start()
        
        # Process models with 2 workers in parallel
        NUM_WORKERS = 2
        print(f"Using {NUM_WORKERS} workers for parallel processing...")
        print(f"Will insert variants in batches of {BATCH_INSERT_SIZE}...")
        
        def insert_batch_if_needed():
            """Insert variants from buffer if it reaches BATCH_INSERT_SIZE."""
            nonlocal cumulative_stats
            batch_to_insert = None
            with variants_lock:
                if len(variants_buffer) >= BATCH_INSERT_SIZE:
                    batch_to_insert = variants_buffer[:BATCH_INSERT_SIZE]
                    variants_buffer[:] = variants_buffer[BATCH_INSERT_SIZE:]
            
            # Insert the batch outside the lock to avoid blocking other threads
            if batch_to_insert:
                batch_stats = store_variants_in_database(batch_to_insert)
                
                # Update cumulative stats (need lock for this)
                with variants_lock:
                    cumulative_stats['inserted'] += batch_stats['inserted']
                    cumulative_stats['errors'] += batch_stats['errors']
                    cumulative_stats['skipped'] += batch_stats['skipped']
                    total_inserted = cumulative_stats['inserted']
                
                print(f"Batch inserted: {batch_stats['inserted']} variants stored. Total inserted so far: {total_inserted}")
        
        def process_model_wrapper(model):
            """Wrapper to process model and update counters."""
            nonlocal processed, failed
            try:
                variants = process_model(model)
                with variants_lock:
                    all_variants.extend(variants)
                    variants_buffer.extend(variants)
                    processed += 1
                
                # Check if we need to insert a batch (outside the lock)
                insert_batch_if_needed()
                
                print(f"✓ Processed model {model['id']}: {len(variants)} variants found")
                return {'success': True, 'model_id': model['id'], 'variants': len(variants)}
            except Exception as error:
                with variants_lock:
                    failed += 1
                print(f"✗ Failed to process model {model['id']}: {error}")
                return {'success': False, 'model_id': model['id'], 'error': str(error)}
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all tasks
            future_to_model = {executor.submit(process_model_wrapper, model): model for model in models}
            
            # Process completed tasks
            for future in as_completed(future_to_model):
                try:
                    result = future.result()
                except Exception as exc:
                    model = future_to_model[future]
                    print(f"Model {model['id']} generated an exception: {exc}")
                    with variants_lock:
                        failed += 1
        
        # Stop logging thread
        logging_active.clear()
        time.sleep(0.1)  # Give logging thread a moment to finish
        
        # Final progress log
        print(f"===>> total {len(all_variants)} variants added till now")
        print(f"===> {failed} failed till now")
        
        # Insert any remaining variants in the buffer
        if variants_buffer:
            print(f"Inserting remaining {len(variants_buffer)} variants...")
            remaining_stats = store_variants_in_database(variants_buffer)
            cumulative_stats['inserted'] += remaining_stats['inserted']
            cumulative_stats['errors'] += remaining_stats['errors']
            cumulative_stats['skipped'] += remaining_stats['skipped']
        
        # Use cumulative stats as final storage stats
        storage_stats = cumulative_stats
        
        elapsed_time = time.time() - start_time
        print(f"Completed scraping. Processed {processed} models, found {len(all_variants)} variants.")
        print(f"Total time taken: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        return jsonify({
            'success': True,
            'models_processed': processed,
            'models_failed': failed,
            'variants_found': len(all_variants),
            'scraping_time_seconds': round(elapsed_time, 2),
            'database_stats': storage_stats,
        }), 200
        
    except Exception as error:
        print(f"Error in variant-price scraper: {error}")
        error_message = str(error)
        return jsonify({
            'error': error_message,
            'success': False
        }), 500

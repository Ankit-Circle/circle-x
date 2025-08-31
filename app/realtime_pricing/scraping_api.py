from flask import Blueprint, request, jsonify
import threading
import logging
from typing import Dict, Any
import sys
import os

# Add the current directory to Python path to import scrapers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import scrapers dynamically to avoid import issues with directory names
import importlib.util
import os

def import_scraper(module_name):
    """Dynamically import scraper modules"""
    try:
        # Add the 91mobiles directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mobiles_dir = os.path.join(current_dir, "91mobiles")
        
        if mobiles_dir not in sys.path:
            sys.path.insert(0, mobiles_dir)
        
        # Now import the module normally
        module = importlib.import_module(module_name)
        return module
    except Exception as e:
        logger.error(f"Failed to import {module_name}: {e}")
        return None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

scraping_bp = Blueprint("scraping", __name__)

def run_scraper_in_background(mobile: str, platform: str):
    """Run the appropriate scraper in a background thread"""
    try:
        logger.info(f"Starting background scraping for {mobile} on {platform}")
        
        if platform.lower() == "91mobile":
            if mobile.lower() == "apple":
                # Import Apple scraper dynamically
                apple_module = import_scraper("apple_91mob_scraper")
                if apple_module and hasattr(apple_module, 'AppleOnlyScraper'):
                    scraper = apple_module.AppleOnlyScraper(headless=True)
                    scraper.scrape_all_apple_products()
                    scraper.close()
                    logger.info("Apple scraping completed successfully")
                else:
                    logger.error("Failed to import Apple scraper")
                
            elif mobile.lower() == "samsung":
                # Import Samsung scraper dynamically
                samsung_module = import_scraper("samsung_91mob_scraper")
                if samsung_module and hasattr(samsung_module, 'SamsungOnlyScraper'):
                    scraper = samsung_module.SamsungOnlyScraper(headless=True)
                    scraper.scrape_all_samsung_products()
                    scraper.close()
                    logger.info("Samsung scraping completed successfully")
                else:
                    logger.error("Failed to import Samsung scraper")
                
            else:
                logger.error(f"Unsupported mobile brand: {mobile}")
                
        else:
            logger.error(f"Unsupported platform: {platform}")
            
    except Exception as e:
        logger.error(f"Error in background scraping: {str(e)}")

@scraping_bp.route("/scrape", methods=["POST"])
def scrape_mobiles():
    """API endpoint to start mobile scraping in background"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        mobile = data.get("mobile")
        platform = data.get("platform")
        
        if not mobile or not platform:
            return jsonify({"error": "Missing required fields: mobile and platform"}), 400
            
        # Validate mobile brand
        supported_mobiles = ["apple", "samsung"]
        if mobile.lower() not in supported_mobiles:
            return jsonify({"error": f"Unsupported mobile brand. Supported: {supported_mobiles}"}), 400
            
        # Validate platform
        supported_platforms = ["91mobile"]
        if platform.lower() not in supported_platforms:
            return jsonify({"error": f"Unsupported platform. Supported: {supported_platforms}"}), 400
            
        # Start scraping in background thread
        thread = threading.Thread(
            target=run_scraper_in_background,
            args=(mobile, platform),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Started background scraping for {mobile} on {platform}")
        
        return jsonify({
            "message": "Scraping started successfully",
            "mobile": mobile,
            "platform": platform,
            "status": "running"
        }), 200
        
    except Exception as e:
        logger.error(f"Error in scrape endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

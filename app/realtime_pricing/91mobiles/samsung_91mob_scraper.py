#!/usr/bin/env python3
"""
Samsung Mobile Phones Scraper for 91mobiles.com
Scrapes ALL Samsung products with pagination support and saves to CSV and Supabase
"""

import time
import random
import csv
import logging
import os
from typing import List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import config
import re
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SamsungOnlyScraper:
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.driver = None
        self.session_start_time = time.time()
        self.total_products = 0
        
        # Initialize Supabase client
        self.supabase_url = config.SUPABASE_URL
        self.supabase_key = config.SUPABASE_KEY
        
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            self.supabase = None
        
    def setup_driver(self):
        """Setup Chrome driver with anti-detection options"""
        try:
            chrome_options = Options()
            
            if self.headless:
                chrome_options.add_argument("--headless")
            
            # Docker/Container specific options
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--remote-debugging-port=9222")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--allow-running-insecure-content")
            
            # Anti-detection options
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Random window size
            width = random.choice([1366, 1440, 1536, 1600, 1920])
            height = random.choice([768, 900, 864, 900, 1080])
            chrome_options.add_argument(f"--window-size={width},{height}")
            
            # Random user agent
            user_agent = random.choice(config.USER_AGENTS)
            chrome_options.add_argument(f"--user-agent={user_agent}")
            
            # Additional options
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--disable-images")  # Faster loading
            
            # Create driver with webdriver-manager
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Execute script to remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Set random viewport
            self.driver.set_window_size(width, height)
            
            logger.info(f"Chrome driver setup complete - Window: {width}x{height}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Chrome driver: {str(e)}")
            return False
    
    def random_delay(self, min_delay: float = 2.0, max_delay: float = 5.0):
        """Add random delay between actions"""
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
        return delay
    
    def simulate_human_behavior(self):
        """Simulate human-like behavior"""
        try:
            # Random mouse movements (if not headless)
            if not self.headless:
                # Scroll randomly
                scroll_amount = random.randint(100, 500)
                self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                time.sleep(random.uniform(0.5, 1.5))
                
                # Scroll back up
                self.driver.execute_script(f"window.scrollBy(0, -{scroll_amount//2});")
                time.sleep(random.uniform(0.3, 0.8))
            
            # Random wait
            time.sleep(random.uniform(1, 3))
            
        except Exception as e:
            logger.warning(f"Error in human behavior simulation: {str(e)}")
    
    def generate_sku_id(self, model: str, ram: str, storage: str) -> str:
        """Generate SKU_ID for Samsung phones: SAMS24-128GB+8GB"""
        try:
            # Extract Samsung model number from model (e.g., "Galaxy S24" -> "S24")
            # Check Z-series FIRST before other Galaxy models
            if 'galaxy z' in model.lower():
                if 'fold' in model.lower():
                    fold_match = re.search(r'fold\s*(\d+)', model, re.IGNORECASE)
                    if fold_match:
                        model_num = f"ZFOLD{fold_match.group(1)}"
                    else:
                        model_num = "ZFOLD"
                elif 'flip' in model.lower():
                    flip_match = re.search(r'flip\s*(\d+)', model, re.IGNORECASE)
                    if flip_match:
                        model_num = f"ZFLIP{flip_match.group(1)}"
                    else:
                        model_num = "ZFLIP"
                else:
                    z_match = re.search(r'galaxy\s+z\s*(\w+)', model, re.IGNORECASE)
                    if z_match:
                        model_num = f"Z{z_match.group(1)}"
                    else:
                        model_num = "Z"
            elif 'galaxy a' in model.lower():
                a_match = re.search(r'galaxy\s+a(\d+)', model, re.IGNORECASE)
                if a_match:
                    model_num = f"A{a_match.group(1)}"
                else:
                    model_num = "A"
            elif 'galaxy m' in model.lower():
                m_match = re.search(r'galaxy\s+m(\d+)', model, re.IGNORECASE)
                if m_match:
                    model_num = f"M{m_match.group(1)}"
                else:
                    model_num = "M"
            elif 'galaxy f' in model.lower():
                f_match = re.search(r'galaxy\s+f(\d+)', model, re.IGNORECASE)
                if f_match:
                    model_num = f"F{f_match.group(1)}"
                else:
                    model_num = "F"
            else:
                # Handle S-series and other Galaxy models
                samsung_match = re.search(r'galaxy\s+([a-z]\d+)', model, re.IGNORECASE)
                if samsung_match:
                    model_num = samsung_match.group(1).upper()
                else:
                    # Try to extract any number from the model
                    num_match = re.search(r'(\d+)', model)
                    if num_match:
                        model_num = f"MODEL{num_match.group(1)}"
                    else:
                        model_num = "UNK"
            
            # Handle variants (Ultra, Plus, etc.)
            variant = ""
            if 'ULTRA' in model.upper():
                variant = "-ULTRA"
            elif 'PLUS' in model.upper():
                variant = "-PLUS"
            elif 'FE' in model.upper():
                variant = "-FE"
            # Removed 5G variant as it's not a meaningful differentiator
            
            # Clean up storage (remove spaces, handle TB)
            clean_storage = storage.replace(" ", "").upper() if storage else "UNK"
            if not clean_storage or clean_storage == "UNK":
                clean_storage = "UNK"
            elif "TB" in clean_storage:
                clean_storage = clean_storage.replace("TB", "TB")
            elif "GB" not in clean_storage and clean_storage.isdigit():
                clean_storage = f"{clean_storage}GB"
            
            # Clean up RAM (remove spaces)
            clean_ram = ram.replace(" ", "").upper() if ram else "UNK"
            if not clean_ram or clean_ram == "UNK":
                clean_ram = "UNK"
            elif "GB" not in clean_ram and clean_ram.isdigit():
                clean_ram = f"{clean_ram}GB"
            
            # Generate base SKU_ID
            base_sku_id = f"SAMS{model_num}{variant}-{clean_storage}+{clean_ram}"
            
            # Check if this exact SKU_ID exists
            if self._sku_exists_in_supabase(base_sku_id):
                # If duplicate exists, add more specific details
                model_lower = model.lower()
                
                # Add color variant if available
                color_variants = ['black', 'white', 'blue', 'red', 'gold', 'silver', 'purple', 'green', 'pink', 'gray']
                color_suffix = ""
                for color in color_variants:
                    if color in model_lower:
                        color_suffix = f"-{color.upper()}"
                        break
                
                # Add year if available
                year_match = re.search(r'(\d{4})', model)
                year_suffix = ""
                if year_match:
                    year_suffix = f"-{year_match.group(1)}"
                
                # Create unique SKU_ID with additional details
                unique_sku_id = f"SAMS{model_num}{variant}-{clean_storage}+{clean_ram}{color_suffix}{year_suffix}"
                
                # If still duplicate, just return the base SKU_ID - let the database handle uniqueness
                # This prevents ugly hash suffixes like "1BF8"
                return base_sku_id
            
            return base_sku_id
            
        except Exception as e:
            logger.error(f"Error generating SKU_ID: {str(e)}")
            return f"SAMS-{int(time.time())}"
    
    def _sku_exists_in_supabase(self, sku_id: str) -> bool:
        """Check if SKU_ID already exists in Supabase"""
        try:
            if not self.supabase:
                return False
            
            response = self.supabase.table('sku_master').select('sku_id').eq('sku_id', sku_id).execute()
            return len(response.data) > 0
            
        except Exception as e:
            logger.warning(f"Error checking SKU existence: {str(e)}")
            return False
    
    def insert_to_supabase(self, product: Dict) -> bool:
        """Insert product data into Supabase SKU_MASTER table only if SKU_ID doesn't exist"""
        try:
            if not self.supabase:
                logger.warning("Supabase client not available")
                return False
            
            # Extract specifications
            specs = product.get('specifications', {})
            ram = specs.get('RAM', '')
            storage = specs.get('Storage', '')
            
            # Generate SKU_ID
            sku_id = self.generate_sku_id(product.get('title', ''), ram, storage)
            
            # Check if SKU_ID already exists
            if self._sku_exists_in_supabase(sku_id):
                logger.info(f"SKU_ID {sku_id} already exists, skipping insertion")
                return True  # Return True since this is expected behavior
            
            # Prepare data for Supabase
            supabase_data = {
                'sku_id': sku_id,
                'brand': 'Samsung',
                'model': product.get('title', ''),
                'ram': ram,
                'storage': storage,
                'category': 'Mobile'
            }
            
            # Insert into Supabase
            response = self.supabase.table('sku_master').insert(supabase_data).execute()
            
            if response.data:
                logger.info(f"Successfully inserted to Supabase: {sku_id}")
                return True
            else:
                logger.error(f"Failed to insert to Supabase: {sku_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error inserting to Supabase: {str(e)}")
            return False
    
    def check_supabase_connection(self) -> bool:
        """Check if Supabase connection is working"""
        try:
            if not self.supabase:
                return False
            
            # Try to query the table to check connection
            response = self.supabase.table('sku_master').select('count', count='exact').execute()
            logger.info("Supabase connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"Supabase connection test failed: {str(e)}")
            return False
    
    def get_supabase_stats(self) -> Dict:
        """Get statistics from Supabase SKU_MASTER table"""
        try:
            if not self.supabase:
                return {"error": "Supabase client not available"}
            
            # Get total count
            count_response = self.supabase.table('sku_master').select('count', count='exact').execute()
            total_count = count_response.count if hasattr(count_response, 'count') else 0
            
            # Get Samsung products count
            samsung_response = self.supabase.table('sku_master').select('count', count='exact').eq('brand', 'Samsung').execute()
            samsung_count = samsung_response.count if hasattr(samsung_response, 'count') else 0
            
            return {
                "total_skus": total_count,
                "samsung_skus": samsung_count,
                "connection_status": "Connected"
            }
            
        except Exception as e:
            return {"error": str(e), "connection_status": "Failed"}
    
    def save_to_supabase(self, products: List[Dict]) -> int:
        """Save all products to Supabase and return count of successful insertions"""
        if not self.supabase:
            logger.warning("Supabase client not available, skipping database insertion")
            return 0
        
        successful_inserts = 0
        skipped_duplicates = 0
        price_updates = 0
        total_products = len(products)
        
        logger.info(f"Starting to process {total_products} products for Supabase insertion...")
        
        for i, product in enumerate(products, 1):
            try:
                product_title = product.get('title', '')[:50]
                logger.info(f"Processing product {i}/{total_products}: {product_title}...")
                
                # Check if this product would be a duplicate before attempting insertion
                specs = product.get('specifications', {})
                ram = specs.get('RAM', '')
                storage = specs.get('Storage', '')
                sku_id = self.generate_sku_id(product.get('title', ''), ram, storage)
                
                # Check if SKU_ID already exists in master table
                if self._sku_exists_in_supabase(sku_id):
                    logger.info(f"SKU_ID {sku_id} already exists in master table, skipping product insertion: {product_title}")
                    skipped_duplicates += 1
                    
                    # BUT still handle price updates for existing SKUs
                    if product.get('price'):
                        if self._price_exists_in_supabase(sku_id):
                            # Update existing price
                            if self.update_price_in_supabase(sku_id, product.get('price')):
                                price_updates += 1
                                logger.info(f"Updated price for existing SKU: {sku_id} - ₹{product.get('price')}")
                        else:
                            # Insert new price record
                            if self.insert_price_to_supabase(sku_id, product.get('price')):
                                price_updates += 1
                                logger.info(f"Inserted new price for existing SKU: {sku_id} - ₹{product.get('price')}")
                    
                    continue
                
                # Insert the product using the simplified method
                if self.insert_to_supabase(product):
                    successful_inserts += 1
                    logger.info(f"Successfully inserted: {sku_id} - {product_title}")
                    
                    # Also insert price into sku_price table if price exists
                    if product.get('price'):
                        if self.insert_price_to_supabase(sku_id, product.get('price')):
                            logger.info(f"Inserted price for new SKU: {sku_id} - ₹{product.get('price')}")
                else:
                    logger.error(f"Failed to insert: {sku_id} - {product_title}")
                
                # Add small delay between insertions to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing product {i}: {str(e)}")
                continue
        
        logger.info(f"Supabase processing complete: {successful_inserts} inserted, {skipped_duplicates} skipped (duplicates), {price_updates} price updates/inserts, {total_products} total")
        return successful_inserts
    
    def navigate_to_page(self, url: str) -> bool:
        """Navigate to a page with human-like behavior"""
        try:
            logger.info(f"Navigating to: {url}")
            
            # Random delay before navigation
            self.random_delay(1, 3)
            
            # Set page load timeout
            self.driver.set_page_load_timeout(30)
            
            # Navigate to page
            logger.info("Starting page navigation...")
            self.driver.get(url)
            logger.info("Page navigation completed")
            
            # Wait for page to load with shorter timeout
            logger.info("Waiting for page body to load...")
            try:
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                logger.info("Page body loaded successfully")
            except TimeoutException:
                logger.warning("Page body load timeout, but continuing...")
            
            # Get current URL and title for debugging
            current_url = self.driver.current_url
            current_title = self.driver.title
            logger.info(f"Current URL: {current_url}")
            logger.info(f"Current title: {current_title}")
            
            # Simulate human behavior
            self.simulate_human_behavior()
            
            # Check if page loaded successfully
            if "403" in current_title or "Forbidden" in current_title:
                logger.error("Page returned 403 Forbidden")
                return False
            
            if "404" in current_title or "Not Found" in current_title:
                logger.error("Page returned 404 Not Found")
                return False
            
            logger.info("Page loaded successfully")
            return True
            
        except TimeoutException as e:
            logger.error(f"Page load timeout: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Navigation error: {str(e)}")
            return False
    
    def get_total_pages(self):
        """Get the total number of pages to scrape"""
        try:
            # Wait for page to load completely
            time.sleep(3)
            
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Method 1: Look for "Last" button which shows the actual last page number
            last_button = soup.find('span', string='Last')
            if last_button:
                onclick_attr = last_button.get('onclick', '')
                # Extract page number from onclick="return submitPage('last', '17');"
                last_match = re.search(r"submitPage\('last', '(\d+)'\)", onclick_attr)
                if last_match:
                    total_pages = int(last_match.group(1))
                    logger.info(f"Found total pages from Last button: {total_pages}")
                    return total_pages
            
            # Method 2: Look for total product count and calculate pages
            all_text = soup.get_text()
            # Look for patterns like "329 Mobiles" or "329 Samsung Mobiles"
            total_count_match = re.search(r'(\d+)\s*(?:Samsung\s+)?Mobiles', all_text)
            if total_count_match:
                total_count = int(total_count_match.group(1))
                # Calculate pages: 329 products ÷ 20 per page = 16.45 → 17 pages
                calculated_pages = (total_count + 19) // 20  # Round up division
                logger.info(f"Found total count: {total_count}, calculated pages: {calculated_pages}")
                
                # Verify with Last button if available
                if last_button:
                    onclick_attr = last_button.get('onclick', '')
                    last_match = re.search(r"submitPage\('last', '(\d+)'\)", onclick_attr)
                    if last_match:
                        actual_pages = int(last_match.group(1))
                        if abs(actual_pages - calculated_pages) <= 1:  # Allow 1 page difference
                            logger.info(f"Verified: Last button shows {actual_pages} pages, calculated {calculated_pages}")
                            return actual_pages
                        else:
                            logger.warning(f"Page count mismatch: Last button shows {actual_pages}, calculated {calculated_pages}")
                            return max(actual_pages, calculated_pages)
                
                return calculated_pages
            
            # Method 3: Look for pagination text like "321 - 329 Mobiles" and calculate
            count_match = re.search(r'(\d+)\s*-\s*(\d+)\s*Mobiles', all_text)
            if count_match:
                start = int(count_match.group(1))
                end = int(count_match.group(2))
                # If we see "321 - 329 Mobiles", this suggests we're on the last page
                if start > 300:  # Likely near the end for Samsung
                    # Look for Last button to confirm
                    last_span = soup.find('span', string='Last')
                    if last_span:
                        onclick_attr = last_span.get('onclick', '')
                        last_match = re.search(r"submitPage\('last', '(\d+)'\)", onclick_attr)
                        if last_match:
                            total_pages = int(last_match.group(1))
                            logger.info(f"Found total pages from Last button: {total_pages}")
                            return total_pages
                    else:
                        # Estimate based on current range
                        estimated_page = (start - 1) // 20 + 1
                        logger.info(f"Estimated total pages based on product range {start}-{end}: {estimated_page}")
                        return estimated_page
            
            # Method 4: Look for JavaScript variable
            script_tags = soup.find_all('script')
            for script in script_tags:
                if script.string:
                    match = re.search(r"var total_pages = '(\d+)';", script.string)
                    if match:
                        total_pages = int(match.group(1))
                        # Verify this makes sense by checking if we can find pagination controls
                        pagination_controls = soup.find_all('span', class_='list-bttnn')
                        if pagination_controls:
                            logger.info(f"Found total pages from JavaScript variable: {total_pages}")
                            return total_pages
            
            # Method 5: Look for pagination elements
            pagination_selectors = [
                'div[class*="pagination"]',
                'ul[class*="pagination"]',
                'div[class*="pager"]',
                'div[class*="nav"]',
                'div[class*="next"]'
            ]
            
            for selector in pagination_selectors:
                pagination_elem = soup.select_one(selector)
                if pagination_elem:
                    # Look for page numbers in pagination
                    page_links = pagination_elem.find_all('a')
                    page_numbers = []
                    for link in page_links:
                        text = link.get_text(strip=True)
                        if text.isdigit():
                            page_numbers.append(int(text))
                    
                    if page_numbers:
                        total_pages = max(page_numbers)
                        logger.info(f"Found total pages from pagination: {total_pages}")
                        return total_pages
            
            # Method 6: Look for text patterns like "Page 1 of X"
            patterns = [
                r'Page\s+(\d+)\s+of\s+(\d+)',    # "Page 1 of 17"
                r'(\d+)\s*of\s*(\d+)',           # "1 of 17"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, all_text)
                if match:
                    if len(match.groups()) == 2:
                        total_pages = int(match.group(2))
                        logger.info(f"Found total pages from text pattern: {total_pages}")
                        return total_pages
            
            # Method 7: Estimate based on product count and products per page
            # Look for text like "1 - 20 Mobiles" to estimate
            count_match = re.search(r'(\d+)\s*-\s*(\d+)\s*Mobiles', all_text)
            if count_match:
                start = int(count_match.group(1))
                end = int(count_match.group(2))
                # If we see "1 - 20 Mobiles" and there's a "Next" button, estimate more pages
                if start == 1 and end == 20:
                    # Look for "Next" button to confirm pagination
                    next_button = soup.find('span', string='Next')
                    if next_button:
                        # For Samsung, estimate based on typical count (around 329)
                        estimated_pages = 17  # 329 ÷ 20 ≈ 17
                        logger.info(f"Estimated total pages for Samsung: {estimated_pages}")
                        return estimated_pages
            
            # Fallback: If we can't determine, assume at least 17 pages for Samsung
            logger.warning("Could not determine total pages, assuming 17 pages for Samsung (329 products)")
            return 17
            
        except Exception as e:
            logger.error(f"Error getting total pages: {str(e)}")
            # Fallback to estimated pages for Samsung
            return 17

    def find_next_page_url(self, current_page):
        """Find and click the next page button, handling dynamic content loading."""
        try:
            # Wait a bit for the page to stabilize
            time.sleep(2)
            
            # Multiple strategies to find the Next button
            next_button_selectors = [
                "//span[contains(@class, 'list-bttnn') and text()='Next']",
                "//a[contains(@class, 'list-bttnn') and text()='Next']",
                "//button[contains(@class, 'list-bttnn') and text()='Next']",
                "//div[contains(@class, 'list-bttnn') and text()='Next']",
                "//span[text()='Next']",
                "//a[text()='Next']",
                "//button[text()='Next']",
                "//div[text()='Next']",
                "//*[contains(text(), 'Next')]"
            ]
            
            next_button = None
            used_selector = None
            
            # Try to find the Next button with different selectors
            for selector in next_button_selectors:
                try:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    if elements:
                        next_button = elements[0]
                        used_selector = selector
                        logger.info(f"Found Next button using selector: {selector}")
                        break
                except Exception as e:
                    continue
            
            if not next_button:
                logger.warning("No Next button found with any selector")
                return None
            
            # Check if the button is clickable
            try:
                # Wait for element to be clickable
                WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, used_selector))
                )
            except TimeoutException:
                logger.warning("Next button not clickable, trying to scroll to it")
                # Scroll to the button
                self.driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                time.sleep(1)
            
            # Get current page content for comparison
            current_content = self.driver.page_source
            current_product_count = len(self.driver.find_elements(By.CSS_SELECTOR, "div.finder_snipet_wrap"))
            current_page_text = self.driver.find_element(By.TAG_NAME, "body").text
            
            # Look for current page indicator like "1 - 20 Mobiles"
            current_range_match = re.search(r'(\d+)\s*-\s*(\d+)\s*Mobiles', current_page_text)
            current_start = int(current_range_match.group(1)) if current_range_match else 0
            
            logger.info(f"Current page shows products {current_start}-{current_start + current_product_count - 1}")
            
            # Click the button
            logger.info("Clicking Next button to go to next page...")
            next_button.click()
            
            # Wait for content to change - longer wait for JavaScript pagination
            time.sleep(5)
            
            # Check if content actually changed using multiple methods
            new_content = self.driver.page_source
            new_product_count = len(self.driver.find_elements(By.CSS_SELECTOR, "div.finder_snipet_wrap"))
            new_page_text = self.driver.find_element(By.TAG_NAME, "body").text
            
            # Look for new page indicator
            new_range_match = re.search(r'(\d+)\s*-\s*(\d+)\s*Mobiles', new_page_text)
            new_start = int(new_range_match.group(1)) if new_range_match else 0
            
            logger.info(f"After clicking Next, page shows products {new_start}-{new_start + new_product_count - 1}")
            
            # Check if we're on a new page
            content_changed = False
            
            # Method 1: Check if product range changed
            if new_start > current_start:
                content_changed = True
                logger.info(f"Product range changed from {current_start}-{current_start + current_product_count - 1} to {new_start}-{new_start + new_product_count - 1}")
            
            # Method 2: Check if product count changed significantly
            elif abs(new_product_count - current_product_count) > 2:
                content_changed = True
                logger.info(f"Product count changed from {current_product_count} to {new_product_count}")
            
            # Method 3: Check if page content hash changed significantly
            elif len(new_content) != len(current_content) or new_content[:2000] != current_content[:2000]:
                content_changed = True
                logger.info("Page content changed significantly")
            
            # Method 4: Check if we can find new products that weren't there before
            if not content_changed:
                # Look for any new product titles
                current_titles = set()
                new_titles = set()
                
                try:
                    current_products = BeautifulSoup(current_content, 'html.parser').find_all('div', class_='finder_snipet_wrap')
                    new_products = BeautifulSoup(new_content, 'html.parser').find_all('div', class_='finder_snipet_wrap')
                    
                    for product in current_products:
                        title_elem = product.find('a', class_='name')
                        if title_elem:
                            current_titles.add(title_elem.get_text(strip=True))
                    
                    for product in new_products:
                        title_elem = product.find('a', class_='name')
                        if title_elem:
                            new_titles.add(title_elem.get_text(strip=True))
                    
                    # Check if we have new titles
                    new_unique_titles = new_titles - current_titles
                    if len(new_unique_titles) > 0:
                        content_changed = True
                        logger.info(f"Found {len(new_unique_titles)} new product titles")
                        
                except Exception as e:
                    logger.debug(f"Error comparing product titles: {e}")
            
            if content_changed:
                logger.info("Page content changed successfully after clicking Next")
                return "CLICKED_NEXT"
            else:
                logger.warning("Page content did not change after clicking Next")
                # Try to check if we're at the last page
                try:
                    last_button = self.driver.find_element(By.XPATH, "//span[text()='Last']")
                    if last_button:
                        logger.info("Found Last button, may be at the end of results")
                except:
                    pass
                return None
                
        except StaleElementReferenceException:
            logger.warning("Stale element reference, trying to re-find Next button...")
            # Try to re-find and click
            try:
                time.sleep(2)
                # Re-find with the most reliable selector
                elements = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Next')]")
                if elements:
                    next_button = elements[0]
                    next_button.click()
                    time.sleep(5)
                    logger.info("Successfully clicked Next button after re-finding")
                    return "CLICKED_NEXT"
            except Exception as e:
                logger.error(f"Failed to re-find and click Next button: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error clicking Next button: {e}")
            return None
    
    def extract_products_from_current_page(self) -> List[Dict]:
        """Extract all products from the current page"""
        products = []
        
        try:
            # Get page source and parse with BeautifulSoup
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Debug: Print page title to verify we're on the right page
            title = soup.find('title')
            if title:
                logger.info(f"Page title: {title.get_text(strip=True)}")
            
            # COMPREHENSIVE APPROACH: Look for product containers using multiple strategies
            logger.info("Looking for product containers using comprehensive approach...")
            
            # Strategy 1: Look for 'finder_snipet_wrap' class (primary method)
            product_containers = soup.find_all('div', class_='finder_snipet_wrap')
            logger.info(f"Found {len(product_containers)} containers with 'finder_snipet_wrap' class")
            
            # Strategy 2: If not enough found, look for containers with mobile-specific classes
            if len(product_containers) < 15:
                logger.info("Looking for additional mobile containers...")
                mobile_containers = soup.find_all('div', class_=lambda x: x and any(word in x.lower() for word in ['mobile', 'product', 'item', 'card', 'box']))
                logger.info(f"Found {len(mobile_containers)} additional mobile containers")
                
                # Add unique containers that contain both Samsung and price
                for container in mobile_containers:
                    if container not in product_containers:
                        text = container.get_text()
                        if 'Samsung' in text and any(pattern in text for pattern in ['Rs.', 'Rs ', '₹']):
                            if len(text) > 200:  # Reasonable size for a product container
                                product_containers.append(container)
                                logger.debug(f"Added mobile container with text: {text[:100]}...")
            
            # Strategy 3: Look for any div that contains both Samsung and price (fallback)
            if len(product_containers) < 15:
                logger.info("Looking for any div with Samsung + price...")
                all_divs = soup.find_all('div')
                for div in all_divs:
                    if div not in product_containers:
                        text = div.get_text()
                        if 'Samsung' in text and any(pattern in text for pattern in ['Rs.', 'Rs ', '₹']):
                            if len(text) > 200 and len(text) < 5000:  # Reasonable size range
                                product_containers.append(div)
                                logger.debug(f"Added div container with text: {text[:100]}...")
            
            # Strategy 4: Look for specific product patterns in the page
            if len(product_containers) < 15:
                logger.info("Looking for product patterns in page text...")
                all_text = soup.get_text()
                # Count Samsung mentions to estimate expected products
                samsung_count = all_text.count('Samsung')
                logger.info(f"Found {samsung_count} Samsung mentions in page text")
                
                if samsung_count > len(product_containers):
                    logger.warning(f"Expected {samsung_count} Samsung products but only found {len(product_containers)} containers")
            
            # Strategy 5: Look for any product-like elements (emergency fallback)
            if len(product_containers) < 10:
                logger.warning("Very few product containers found, trying emergency fallback...")
                # Look for any elements that might contain product information
                potential_products = soup.find_all(['div', 'article'], class_=lambda x: x and any(word in x.lower() for word in ['product', 'item', 'mobile', 'phone']))
                logger.info(f"Found {len(potential_products)} potential product elements in emergency search")
                
                for elem in potential_products:
                    if elem not in product_containers:
                        text = elem.get_text()
                        if 'Samsung' in text and len(text) > 100:
                            product_containers.append(elem)
                            logger.debug(f"Added emergency container with text: {text[:100]}...")
            
            logger.info(f"Total product containers found: {len(product_containers)}")
            
            # Filter containers to avoid duplicates and select the best ones
            filtered_containers = self._filter_best_containers(product_containers)
            logger.info(f"Filtered to {len(filtered_containers)} best containers")
            
            # Extract product information
            for i, container in enumerate(filtered_containers):
                try:
                    logger.debug(f"Processing container {i+1}/{len(filtered_containers)}")
                    product = self._extract_single_product(container)
                    if product:
                        # Check if this product is already extracted (deduplication)
                        if not self._is_duplicate_product(product, products):
                            products.append(product)
                            logger.debug(f"Successfully extracted product: {product['title'][:50]}...")
                        else:
                            logger.debug(f"Skipped duplicate product: {product['title'][:50]}...")
                    else:
                        logger.debug(f"Failed to extract product from container {i+1}")
                except Exception as e:
                    logger.warning(f"Error extracting product {i+1}: {str(e)}")
                    continue
            
            logger.info(f"Successfully extracted {len(products)} unique products from current page")
            
            # Additional debugging: show some product titles
            if products:
                logger.info("Sample products found:")
                for i, product in enumerate(products[:3]):  # Show first 3 products
                    logger.info(f"  {i+1}. {product.get('title', 'No title')[:60]}...")
            
        except Exception as e:
            logger.error(f"Error extracting products: {str(e)}")
        
        return products

    def _filter_best_containers(self, containers: List) -> List:
        """Filter containers to select the best ones and avoid duplicates"""
        if not containers:
            return []
        
        filtered = []
        seen_titles = set()
        
        for container in containers:
            try:
                # Extract title to check for duplicates
                title_elem = container.find('a', class_='name')
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    if title and title not in seen_titles:
                        # Check if this container has good product information
                        text = container.get_text()
                        has_price = any(pattern in text for pattern in ['Rs.', 'Rs ', '₹'])
                        has_specs = any(keyword in text for keyword in ['GB', 'MP', 'mAh', 'inches'])
                        
                        if has_price and has_specs:
                            filtered.append(container)
                            seen_titles.add(title)
                            logger.debug(f"Added container for: {title}")
            except Exception as e:
                logger.debug(f"Error filtering container: {str(e)}")
                continue
        
        return filtered
    
    def _is_duplicate_product(self, new_product: Dict, existing_products: List[Dict]) -> bool:
        """Check if a product is a duplicate of an existing one"""
        if not existing_products:
            return False
        
        new_title = new_product.get('title', '').lower()
        new_price = new_product.get('price')
        
        for existing in existing_products:
            existing_title = existing.get('title', '').lower()
            existing_price = existing.get('price')
            
            # Check if titles are very similar (allowing for minor variations)
            if new_title == existing_title:
                # Check if prices are the same (within 100 rupees)
                if new_price and existing_price:
                    if abs(new_price - existing_price) < 100:
                        return True
                else:
                    return True
        
        return False
    
    def _extract_single_product(self, container) -> Optional[Dict]:
        """Extract information from a single product container"""
        try:
            # Product title
            title_elem = container.find('a', class_='name')
            title = title_elem.get_text(strip=True) if title_elem else None
            
            # Product URL
            product_url = ""
            link_elem = container.find('a', class_='name')
            if link_elem and link_elem.get('href'):
                href = link_elem.get('href')
                if href.startswith('http'):
                    product_url = href
                else:
                    product_url = f"https://www.91mobiles.com{href}"
            
            # Price
            price = self._extract_price_enhanced(container)
            
            # Rating
            rating = self._extract_rating(container)
            
            # Review count
            review_count = self._extract_review_count(container)
            
            # Image URL
            image_url = ""
            img_elem = container.find('img', class_='finder_pro_image')
            if img_elem and img_elem.get('src'):
                src = img_elem.get('src')
                if src.startswith('http'):
                    image_url = src
                else:
                    image_url = f"https://www.91mobiles.com{src}"
            
            # Specifications - Enhanced extraction
            specs = self._extract_specifications_enhanced(container)
            
            # Add storage from the dedicated div if not already in specs
            storage_elem = container.find('div', class_='finder_icon_storage_text')
            if storage_elem and 'Storage' not in specs:
                specs['Storage'] = storage_elem.get_text(strip=True)
            
            # OS
            os_elem = container.find('div', class_='os_icon_cat')
            os_text = os_elem.get_text(strip=True) if os_elem else None
            if os_text and 'OS' not in specs:
                specs['OS'] = os_text.replace('Android', 'Android ').strip()
            
            # Debug logging
            logger.debug(f"Extracted product: {title[:50] if title else 'No title'}... | Price: {price} | Specs: {len(specs)}")
            
            if not title or len(title) < 5:
                return None
            
            return {
                "title": title,
                "price": price,
                "rating": rating,
                "review_count": review_count,
                "image_url": image_url,
                "product_url": product_url,
                "brand": "Samsung",
                "specifications": specs,
                "scraped_at": time.time(),
                "source": "91mobiles_selenium"
            }
            
        except Exception as e:
            logger.warning(f"Error extracting single product: {str(e)}")
            return None

    def _extract_price_enhanced(self, container) -> Optional[float]:
        """Enhanced price extraction for 91mobiles"""
        try:
            # Look for price elements with multiple strategies
            price_selectors = [
                'span[class*="price"]',
                'div[class*="price"]',
                'span[class*="amount"]',
                'div[class*="amount"]',
                'span[class*="cost"]',
                'div[class*="cost"]',
                'span[class*="rupee"]',
                'div[class*="rupee"]'
            ]
            
            price_text = None
            
            # Strategy 1: Look for specific price elements
            for selector in price_selectors:
                price_elem = container.select_one(selector)
                if price_elem:
                    price_text = price_elem.get_text(strip=True)
                    if price_text and any(pattern in price_text for pattern in ['Rs.', 'Rs ', '₹', 'INR']):
                        break
            
            # Strategy 2: Look for the price text after the WebRupee span
            if not price_text:
                rupee_span = container.find('span', class_='WebRupee')
                if rupee_span:
                    # Get the next text node after the rupee span
                    next_sibling = rupee_span.next_sibling
                    if next_sibling and hasattr(next_sibling, 'string'):
                        price_text = next_sibling.string.strip()
                    else:
                        # Look for the parent element's text and extract price
                        parent_text = rupee_span.parent.get_text(strip=True) if rupee_span.parent else ""
                        if parent_text:
                            # Extract price after "Rs." or "₹"
                            import re
                            price_match = re.search(r'(?:Rs\.|₹)\s*([\d,]+(?:\+)?)', parent_text)
                            if price_match:
                                price_text = price_match.group(0)
            
            # Strategy 3: Look for any text containing price patterns
            if not price_text:
                all_text = container.get_text()
                import re
                price_match = re.search(r'(?:Rs\.|₹)\s*([\d,]+(?:\+)?)', all_text)
                if price_match:
                    price_text = price_match.group(0)
            
            if price_text:
                # Parse the price
                parsed_price = self._parse_price_text(price_text)
                if parsed_price:
                    logger.debug(f"Extracted price: {price_text} -> {parsed_price}")
                    return parsed_price
            
            logger.warning(f"No price found in container")
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting price: {str(e)}")
            return None
    
    def _parse_price_text(self, text: str) -> Optional[float]:
        """Parse price from text string"""
        try:
            import re
            # Multiple price patterns for Indian Rupees
            patterns = [
                r'Rs\.\s*([\d,]+)',        # Rs. 45,999
                r'Rs\s*([\d,]+)',          # Rs 45,999
                r'₹\s*([\d,]+)',           # ₹ 45,999
                r'INR\s*([\d,]+)',         # INR 45,999
                r'([\d,]+)\s*Rs\.',        # 45,999 Rs.
                r'([\d,]+)\s*Rs',          # 45,999 Rs
                r'([\d,]+)\s*₹',           # 45,999 ₹
                r'([\d,]+)\s*INR',         # 45,999 INR
                r'Price:\s*Rs\.\s*([\d,]+)', # Price: Rs. 45,999
                r'Price:\s*([\d,]+)',      # Price: 45,999
                r'([\d,]+)\s*/-',          # 45,999 /-
                r'([\d,]+)',               # Just the number
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    price_str = match.group(1).replace(',', '')
                    try:
                        return float(price_str)
                    except ValueError:
                        continue
            
            return None
        except:
            return None
    
    def _extract_rating(self, container) -> Optional[float]:
        """Extract rating from the container"""
        try:
            # Look for rating in the rating_box_new_list div
            rating_elem = container.find('div', class_='rating_box_new_list')
            if rating_elem:
                rating_text = rating_elem.get_text(strip=True)
                # Extract percentage from text like "94%"
                import re
                rating_match = re.search(r'(\d+)%', rating_text)
                if rating_match:
                    rating_percentage = int(rating_match.group(1))
                    # Convert percentage to 5-star scale
                    rating_stars = (rating_percentage / 100) * 5
                    return round(rating_stars, 1)
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting rating: {str(e)}")
            return None
    
    def _extract_review_count(self, container) -> Optional[int]:
        """Extract review count from container"""
        try:
            review_elem = container.find(['span', 'div'], string=lambda x: x and 'review' in x.lower())
            if review_elem:
                review_text = review_elem.get_text(strip=True)
                import re
                review_match = re.search(r'(\d+)', review_text)
                if review_match:
                    return int(review_match.group(1))
            return None
        except:
            return None
    
    def _extract_specifications_enhanced(self, container) -> Dict:
        """Enhanced specifications extraction for Samsung mobiles"""
        specs = {}
        try:
            spec_sections = container.find_all('div', class_='left specs_li')
            all_spec_text = ""
            for section in spec_sections:
                spec_list_div = section.find('div', class_='a filter-list-text')
                if spec_list_div:
                    labels = spec_list_div.find_all('label')
                    for label in labels:
                        spec_detail = label.get('title', label.get_text(strip=True))
                        if spec_detail:
                            all_spec_text += spec_detail + "; " # Concatenate all spec details for comprehensive parsing

            all_spec_text_lower = all_spec_text.lower()
            import re

            # RAM pattern: "8 GB RAM" or "6 GB RAM"
            ram_match = re.search(r'(\d+)\s*GB\s*RAM', all_spec_text, re.IGNORECASE)
            if ram_match:
                specs['RAM'] = f"{ram_match.group(1)} GB"

            # Storage pattern: "128 GB" or "256 GB" or "1TB"
            # Prioritize the dedicated storage text if available
            storage_elem = container.find('div', class_='finder_icon_storage_text')
            if storage_elem:
                specs['Storage'] = storage_elem.get_text(strip=True)
            else:
                storage_match = re.search(r'(\d+)\s*(GB|TB)', all_spec_text, re.IGNORECASE)
                if storage_match:
                    specs['Storage'] = f"{storage_match.group(1)} {storage_match.group(2)}"

            # Camera pattern: "48 MP + 12 MP Dual Primary Cameras"
            camera_match = re.search(r'(\d+)\s*MP(?:\s*\+\s*(\d+)\s*MP)?(?:\s*\+\s*(\d+)\s*MP)?\s*(?:Dual|Triple)?\s*Primary\s*Cameras', all_spec_text, re.IGNORECASE)
            if camera_match:
                camera_str = f"{camera_match.group(1)} MP"
                if camera_match.group(2):
                    camera_str += f" + {camera_match.group(2)} MP"
                if camera_match.group(3):
                    camera_str += f" + {camera_match.group(3)} MP"
                specs['Camera'] = camera_str
            
            # Front Camera
            front_camera_match = re.search(r'(\d+)\s*MP\s*Front\s*Camera', all_spec_text, re.IGNORECASE)
            if front_camera_match:
                specs['Front_Camera'] = f"{front_camera_match.group(1)} MP"

            # Battery pattern: "5000 mAh"
            battery_match = re.search(r'(\d+)\s*mAh', all_spec_text, re.IGNORECASE)
            if battery_match:
                specs['Battery'] = f"{battery_match.group(1)} mAh"

            # Display pattern: "6.1 inches (15.49 cm)"
            display_match = re.search(r'(\d+\.?\d*)\s*inches', all_spec_text, re.IGNORECASE)
            if display_match:
                specs['Display'] = f"{display_match.group(1)} inches"

            # Processor pattern: "Exynos 2400" or "Snapdragon 8 Gen 3"
            processor_match = re.search(r'(Exynos\s+\d+|Snapdragon\s+\d+\s+Gen\s+\d+|Snapdragon\s+\d+)', all_spec_text, re.IGNORECASE)
            if processor_match:
                specs['Processor'] = processor_match.group(1)
            else: # Also look for Octa Core
                octa_core_match = re.search(r'Octa Core \(.+?\)', all_spec_text, re.IGNORECASE)
                if octa_core_match:
                    specs['Processor_Details'] = octa_core_match.group(0)

            # OS pattern: "Android v14" or "Android v13"
            os_match = re.search(r'Android\s+v(\d+)', all_spec_text, re.IGNORECASE)
            if os_match:
                specs['OS'] = f"Android v{os_match.group(1)}"

            # Port pattern: "USB Type-C Port"
            if 'usb type-c port' in all_spec_text_lower:
                specs['Port'] = 'USB Type-C'

            # Flash pattern: "Dual-color LED Flash" or "Dual LED Flash"
            if 'dual-color led flash' in all_spec_text_lower:
                specs['Flash'] = 'Dual-color LED Flash'
            elif 'dual led flash' in all_spec_text_lower:
                specs['Flash'] = 'Dual LED Flash'
            elif 'quad led' in all_spec_text_lower:
                specs['Flash'] = 'Quad LED Flash'
            elif 'led flash' in all_spec_text_lower:
                specs['Flash'] = 'LED Flash'

            # Refresh Rate pattern: "120 Hz Refresh Rate" or "60 Hz Refresh Rate"
            refresh_match = re.search(r'(\d+)\s*Hz\s*Refresh\s*Rate', all_spec_text, re.IGNORECASE)
            if refresh_match:
                specs['Refresh_Rate'] = f"{refresh_match.group(1)} Hz"

        except Exception as e:
            logger.debug(f"Error extracting specs: {str(e)}")
        return specs

    def scrape_all_samsung_products(self):
        """Scrape all Samsung products from all available pages."""
        all_products = []
        current_page = 1
        total_supabase_inserts = 0
        
        try:
            # Setup Chrome driver first
            if not self.setup_driver():
                logger.error("Failed to setup Chrome driver")
                return all_products
            
            # Start with the main Samsung page
            samsung_url = "https://www.91mobiles.com/samsung-mobile-price-list-in-india"
            
            if not self.navigate_to_page(samsung_url):
                logger.error("Failed to navigate to Samsung page")
                return all_products
            
            # Get total pages
            total_pages = self.get_total_pages()
            logger.info(f"Total pages to scrape: {total_pages}")
            
            # Add a safety check for expected Samsung product count
            expected_products = 329
            expected_pages = (expected_products + 19) // 20  # Round up division
            if total_pages < expected_pages:
                logger.warning(f"Detected {total_pages} pages but expected {expected_pages} for {expected_products} products")
                logger.info(f"Will attempt to scrape at least {expected_pages} pages")
                total_pages = max(total_pages, expected_pages)
            
            while current_page <= total_pages:
                logger.info(f"Scraping page {current_page}/{total_pages}")
                print(f"📄 Processing page {current_page}/{total_pages}...")
                
                # Scroll down and wait for lazy-loaded content before extracting products
                logger.info(f"Page {current_page}: Scrolling to load all content...")
                products_before_scroll = len(self.driver.find_elements(By.CSS_SELECTOR, "div.finder_snipet_wrap"))
                final_product_count = self.scroll_and_wait_for_content()
                
                if final_product_count > products_before_scroll:
                    logger.info(f"Page {current_page}: Lazy loading successful! Products increased from {products_before_scroll} to {final_product_count}")
                else:
                    logger.info(f"Page {current_page}: No additional products found during scroll")
                
                # Extract products from current page (now with all lazy-loaded content)
                page_products = self.extract_products_from_current_page()
                
                if page_products:
                    logger.info(f"Page {current_page}: Found {len(page_products)} products")
                    all_products.extend(page_products)
                    logger.info(f"Total products so far: {len(all_products)}")
                    
                    # INSERT TO SUPABASE IMMEDIATELY AFTER EACH PAGE
                    logger.info(f"Page {current_page}: Processing {len(page_products)} products for Supabase...")
                    print(f"💾 Processing {len(page_products)} products for Supabase...")
                    page_inserts = self.save_to_supabase(page_products)
                    total_supabase_inserts += page_inserts
                    logger.info(f"Page {current_page}: Supabase processing complete - {page_inserts} new products inserted")
                    print(f"✅ Page {current_page}: Supabase processing complete - {page_inserts} new products inserted")
                    print(f"📊 Total new Supabase inserts: {total_supabase_inserts}")
                    
                    # Check if we're approaching the expected total
                    if len(all_products) >= expected_products * 0.9:  # 90% of expected
                        logger.info(f"Reached {len(all_products)} products ({len(all_products)/expected_products*100:.1f}% of expected {expected_products})")
                else:
                    logger.warning(f"Page {current_page}: No products found")
                
                # Check if we've reached the last page
                if self.is_last_page():
                    logger.info("Reached the last page, stopping pagination")
                    break
                
                # Try to go to next page
                if current_page < total_pages:
                    logger.info(f"Attempting to navigate to page {current_page + 1}")
                    
                    # Try to find and click Next button
                    next_result = self.find_next_page_url(current_page)
                    
                    if next_result == "CLICKED_NEXT":
                        logger.info(f"Successfully navigated to page {current_page + 1}")
                        current_page += 1
                        
                        # Wait for new content to load
                        time.sleep(3)
                        
                        # Verify we're on a new page by checking if content changed
                        try:
                            # Look for any product containers to confirm new page loaded
                            containers = self.driver.find_elements(By.CSS_SELECTOR, "div.finder_snipet_wrap")
                            if containers:
                                logger.info(f"Page {current_page} loaded successfully with {len(containers)} product containers")
                            else:
                                logger.warning(f"Page {current_page} may not have loaded properly")
                        except Exception as e:
                            logger.error(f"Error verifying page {current_page}: {e}")
                    else:
                        logger.warning(f"Could not navigate to next page from page {current_page}")
                        # Check if we're actually on the last page
                        if self.is_last_page():
                            logger.info("Confirmed we're on the last page, stopping")
                            break
                        else:
                            logger.error("Navigation failed and we're not on the last page - stopping")
                            break
                else:
                    logger.info("Reached last page, stopping pagination")
                    break
                
                # Add delay between pages
                time.sleep(2)
                
                # Show page completion summary
                print(f"🎯 Page {current_page} completed! Products: {len(page_products) if page_products else 0}, Total so far: {len(all_products)}")
                print("-" * 50)
            
            logger.info(f"Scraping complete! Total products: {len(all_products)}")
            logger.info(f"Total Supabase inserts: {total_supabase_inserts}")
            print(f"🎉 All pages completed! Total products: {len(all_products)}, Total Supabase inserts: {total_supabase_inserts}")
            return all_products
            
            # Final verification
            if len(all_products) < expected_products * 0.8:  # Less than 80% of expected
                logger.warning(f"Scraped {len(all_products)} products but expected around {expected_products}")
                logger.warning("This might indicate pagination issues or site changes")
                
                # Try to force continue scraping
                if len(all_products) < expected_products * 0.9:  # Less than 90% of expected
                    logger.info("Attempting to force continue scraping...")
                    all_products = self.force_continue_scraping(all_products, current_page, total_pages)
                    logger.info(f"After force continue: Total products: {len(all_products)}")
            
            return all_products
            
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            return all_products
    
    def save_to_csv(self, products: List[Dict], filename: str = "samsung_mobiles_91mobiles.csv"):
        """Save products to CSV file"""
        try:
            if not products:
                logger.warning("No products to save")
                return
            
            # Define CSV headers
            headers = [
                'Title', 'Price (₹)', 'Rating', 'Review Count', 'Image URL',
                'Product URL', 'Brand', 'RAM', 'Storage', 'Camera', 'Front_Camera', 'Battery',
                'Display', 'Processor', 'Processor_Details', 'OS', 'Port', 'Flash', 'Refresh_Rate',
                'Scraped At', 'Source'
            ]
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                
                for product in products:
                    # Prepare row data
                    row = {
                        'Title': product.get('title', ''),
                        'Price (₹)': product.get('price', ''),
                        'Rating': product.get('rating', ''),
                        'Review Count': product.get('review_count', ''),
                        'Image URL': product.get('image_url', ''),
                        'Product URL': product.get('product_url', ''),
                        'Brand': product.get('brand', ''),
                        'RAM': product.get('specifications', {}).get('RAM', ''),
                        'Storage': product.get('specifications', {}).get('Storage', ''),
                        'Camera': product.get('specifications', {}).get('Camera', ''),
                        'Front_Camera': product.get('specifications', {}).get('Front_Camera', ''),
                        'Battery': product.get('specifications', {}).get('Battery', ''),
                        'Display': product.get('specifications', {}).get('Display', ''),
                        'Processor': product.get('specifications', {}).get('Processor', ''),
                        'Processor_Details': product.get('specifications', {}).get('Processor_Details', ''),
                        'OS': product.get('specifications', {}).get('OS', ''),
                        'Port': product.get('specifications', {}).get('Port', ''),
                        'Flash': product.get('specifications', {}).get('Flash', ''),
                        'Refresh_Rate': product.get('specifications', {}).get('Refresh_Rate', ''),
                        'Scraped At': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(product.get('scraped_at', time.time()))),
                        'Source': product.get('source', '')
                    }
                    writer.writerow(row)
            
            logger.info(f"Successfully saved {len(products)} products to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {str(e)}")
    
    def get_status(self) -> Dict:
        """Get current scraper status"""
        return {
            "total_products": self.total_products,
            "session_duration": time.time() - self.session_start_time,
            "driver_active": self.driver is not None,
            "headless_mode": self.headless
        }
    
    def close(self):
        """Close the browser driver"""
        if self.driver:
            self.driver.quit()
            logger.info("Browser driver closed")

    def is_last_page(self):
        """Check if we're on the last page by looking for pagination controls"""
        try:
            # Get current page text to see product range
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            range_match = re.search(r'(\d+)\s*-\s*(\d+)\s*Mobiles', page_text)
            
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                logger.info(f"Current page shows products {start}-{end}")
                
                # Check if we can find a Next button
                next_button = self.driver.find_elements(By.XPATH, "//span[text()='Next']")
                
                if not next_button:
                    logger.info(f"No Next button found on page showing products {start}-{end}")
                    # Check if this is actually the last page by looking for total count
                    total_count_match = re.search(r'(\d+)\s*Mobiles', page_text)
                    if total_count_match:
                        total_count = int(total_count_match.group(1))
                        if end >= total_count:
                            logger.info(f"Reached last page: showing products {start}-{end} out of {total_count} total")
                            return True
                        else:
                            logger.warning(f"Missing Next button but not at end: showing {start}-{end} out of {total_count}")
                            return False
                    else:
                        # If we can't determine total count, assume this is the last page
                        logger.info(f"No Next button and can't determine total count, assuming last page")
                        return True
                else:
                    # Next button exists, so we're not on the last page
                    logger.info(f"Next button found, not on last page")
                    return False
            
            # Fallback: check if we can find any pagination controls
            try:
                last_button = self.driver.find_element(By.XPATH, "//span[text()='Last']")
                if last_button:
                    # Check if the Last button is disabled or if we're on the last page
                    onclick_attr = last_button.get_attribute('onclick')
                    if onclick_attr:
                        # Extract the last page number from onclick
                        last_match = re.search(r"submitPage\('last', '(\d+)'\)", onclick_attr)
                        if last_match:
                            last_page_num = int(last_match.group(1))
                            current_page_num = self.get_current_page_number()
                            if current_page_num >= last_page_num:
                                logger.info(f"On last page: current={current_page_num}, last={last_page_num}")
                                return True
            except:
                pass
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking if last page: {e}")
            return False
    
    def get_current_page_number(self):
        """Get the current page number based on product range"""
        try:
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            range_match = re.search(r'(\d+)\s*-\s*(\d+)\s*Mobiles', page_text)
            if range_match:
                start = int(range_match.group(1))
                # Calculate page number: if showing products 1-20, it's page 1; if 21-40, it's page 2, etc.
                page_number = (start - 1) // 20 + 1
                return page_number
            return 1
        except Exception as e:
            logger.debug(f"Error getting current page number: {e}")
            return 1

    def scroll_and_wait_for_content(self):
        """Scroll down the page to trigger lazy loading and wait for content to appear"""
        try:
            logger.info("Scrolling down to trigger lazy loading...")
            
            # Get initial product count
            initial_count = len(self.driver.find_elements(By.CSS_SELECTOR, "div.finder_snipet_wrap"))
            logger.info(f"Initial product count: {initial_count}")
            
            # Scroll down in steps to trigger lazy loading
            page_height = self.driver.execute_script("return document.body.scrollHeight")
            viewport_height = self.driver.execute_script("return window.innerHeight")
            
            # Scroll down in multiple steps
            scroll_steps = 5
            for step in range(scroll_steps):
                scroll_position = (step + 1) * (page_height // scroll_steps)
                self.driver.execute_script(f"window.scrollTo(0, {scroll_position});")
                time.sleep(1)  # Wait a bit after each scroll
                
                # Check if new products appeared
                current_count = len(self.driver.find_elements(By.CSS_SELECTOR, "div.finder_snipet_wrap"))
                if current_count > initial_count:
                    logger.info(f"New products found during scroll: {current_count} (was {initial_count})")
                    initial_count = current_count
            
            # Scroll to bottom to ensure all content is loaded
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Scroll back to top for consistent starting point
            self.driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)
            
            # Final count check
            final_count = len(self.driver.find_elements(By.CSS_SELECTOR, "div.finder_snipet_wrap"))
            logger.info(f"Final product count after scrolling: {final_count}")
            
            # Wait 6 seconds as requested for any remaining lazy loading
            logger.info("Waiting 6 seconds for any remaining lazy-loaded content...")
            time.sleep(6)
            
            return final_count
            
        except Exception as e:
            logger.error(f"Error during scroll and wait: {e}")
            return 0

    def force_continue_scraping(self, all_products, current_page, total_pages):
        """Force continue scraping by manually navigating to remaining pages"""
        logger.info(f"Pagination may have failed. Attempting to force continue from page {current_page}")
        
        # Try to manually navigate to remaining pages
        for page_num in range(current_page + 1, total_pages + 1):
            try:
                logger.info(f"Force navigating to page {page_num}")
                
                # Try to construct URL for the page
                page_url = f"https://www.91mobiles.com/samsung-mobile-price-list-in-india?page={page_num}"
                if not self.navigate_to_page(page_url):
                    # Fallback: try to find and click page numbers
                    try:
                        page_link = self.driver.find_element(By.XPATH, f"//span[text()='{page_num}']")
                        if page_link:
                            page_link.click()
                            time.sleep(3)
                        else:
                            logger.warning(f"Could not find page {page_num} link")
                            continue
                    except:
                        logger.warning(f"Could not navigate to page {page_num}")
                        continue
                
                # Extract products from this page
                page_products = self.extract_products_from_current_page()
                if page_products:
                    logger.info(f"Page {page_num}: Found {len(page_products)} products")
                    all_products.extend(page_products)
                    logger.info(f"Total products so far: {len(all_products)}")
                else:
                    logger.warning(f"Page {page_num}: No products found")
                
                # Add delay between pages
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error on page {page_num}: {e}")
                continue
        
        return all_products

    def insert_price_to_supabase(self, sku_id: str, price: float) -> bool:
        """Insert price data into Supabase sku_price table"""
        try:
            if not self.supabase:
                logger.warning("Supabase client not available")
                return False
            
            # Prepare price data for Supabase
            price_data = {
                'sku_id': sku_id,
                'price_91': price
            }
            
            # Insert into sku_price table
            response = self.supabase.table('sku_price').insert(price_data).execute()
            
            if response.data:
                logger.info(f"Successfully inserted price to Supabase: {sku_id} - ₹{price}")
                return True
            else:
                logger.error(f"Failed to insert price to Supabase: {sku_id} - ₹{price}")
                return False
                
        except Exception as e:
            logger.error(f"Error inserting price to Supabase: {str(e)}")
            return False
    
    def _price_exists_in_supabase(self, sku_id: str) -> bool:
        """Check if price already exists in Supabase sku_price table"""
        try:
            if not self.supabase:
                return False
            
            response = self.supabase.table('sku_price').select('sku_id').eq('sku_id', sku_id).execute()
            return len(response.data) > 0
            
        except Exception as e:
            logger.warning(f"Error checking price existence: {str(e)}")
            return False
    
    def update_price_in_supabase(self, sku_id: str, price: float) -> bool:
        """Update existing price data in Supabase sku_price table"""
        try:
            if not self.supabase:
                logger.warning("Supabase client not available")
                return False
            
            # Update price in sku_price table
            response = self.supabase.table('sku_price').update({'price_91': price}).eq('sku_id', sku_id).execute()
            
            if response.data:
                logger.info(f"Successfully updated price in Supabase: {sku_id} - ₹{price}")
                return True
            else:
                logger.error(f"Failed to update price in Supabase: {sku_id} - ₹{price}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating price in Supabase: {str(e)}")
            return False

def main():
    """Main function to run the Samsung-only scraper"""
    print("📱 Starting Samsung Mobile Phones Scraper")
    print("📱 Scraping ALL Samsung products with pagination")
    print("💾 Saving results to CSV format")
    print("=" * 60)
    
    # Ask user preference
    headless_input = input("Run in headless mode? (y/n, default: n): ").strip().lower()
    headless = headless_input == 'y'
    
    scraper = SamsungOnlyScraper(headless=headless)
    
    try:
        # Setup driver
        print("🔧 Setting up Chrome driver...")
        if not scraper.setup_driver():
            print("❌ Failed to setup Chrome driver")
            return
        
        print("✅ Chrome driver setup complete")
        
        # Check Supabase connection
        print("🗄️  Testing Supabase connection...")
        if scraper.check_supabase_connection():
            print("✅ Supabase connection successful")
            
            # Show existing data stats
            stats = scraper.get_supabase_stats()
            if "error" not in stats:
                print(f"📊 Database stats: {stats['total_skus']} total SKUs, {stats['samsung_skus']} Samsung products")
            else:
                print(f"⚠️  Could not get database stats: {stats['error']}")
        else:
            print("❌ Supabase connection failed - will continue with CSV export only")
        
        # Show status
        status = scraper.get_status()
        print(f"📊 Status: Driver active: {status['driver_active']}, Headless: {status['headless_mode']}")
        print("=" * 60)
        
        # Scrape all Samsung products
        print("📱 Starting Samsung products scraping...")
        products = scraper.scrape_all_samsung_products()
        
        if products:
            # Save results to CSV
            scraper.save_to_csv(products)
            
            # Print summary
            print("\n" + "=" * 60)
            print("📊 SAMSUNG SCRAPING RESULTS SUMMARY")
            print("=" * 60)
            
            print(f"📱 Total Samsung products scraped: {len(products)}")
            print(f"🗄️  New products inserted to Supabase: {len(products)} (duplicates automatically skipped)")
            
            # Price analysis
            prices = [p['price'] for p in products if p['price']]
            if prices:
                avg_price = sum(prices) / len(prices)
                min_price = min(prices)
                max_price = max(prices)
                print(f"💰 Price Analysis:")
                print(f"   Average price: ₹{avg_price:,.0f}")
                print(f"   Minimum price: ₹{min_price:,.0f}")
                print(f"   Maximum price: ₹{max_price:,.0f}")
            
            # Show final status
            final_status = scraper.get_status()
            print(f"\n🔍 Final Status:")
            print(f"📈 Total products: {final_status['total_products']}")
            print(f"⏱️  Session duration: {final_status['session_duration']:.1f} seconds")
            
            print(f"\n💾 Data saved to: samsung_mobiles_91mobiles.csv")
            print(f"🗄️  Data saved to: Supabase SKU_MASTER table (duplicates automatically skipped)")
            
        else:
            print("❌ No products found. The site may be blocking access.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Scraping interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during scraping: {str(e)}")
    finally:
        scraper.close()

if __name__ == "__main__":
    main()

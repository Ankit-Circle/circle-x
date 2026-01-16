"""
Utility functions for scraping Cashify phone prices.
Converted from TypeScript Supabase Edge Function.
"""
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


def get_browser_headers() -> Dict[str, str]:
    """Browser-like headers for web scraping to avoid detection."""
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }


def fetch_with_browser_headers(url: str, **kwargs) -> requests.Response:
    """Fetch a URL with browser-like headers."""
    headers = get_browser_headers()
    # Merge with any existing headers
    if 'headers' in kwargs:
        headers.update(kwargs['headers'])
    kwargs['headers'] = headers
    return requests.get(url, **kwargs)


def parse_html(html: str) -> Optional[BeautifulSoup]:
    """Parse HTML string into a BeautifulSoup document."""
    try:
        return BeautifulSoup(html, 'lxml')
    except Exception as error:
        print(f'Error parsing HTML: {error}')
        return None


def extract_model_urls_from_selenium(
    driver: webdriver.Chrome,
    base_url: str = 'https://www.cashify.in'
) -> List[Dict[str, str]]:
    """Extract model URLs and names directly from Selenium WebDriver.
    
    This is more reliable than parsing HTML, especially for dynamic content.
    Only extracts models from the specific parent container.
    """
    models: List[Dict[str, str]] = []
    url_set = set()
    
    try:
        # First, try to find the parent container
        # Look for div with class "bg-surface" that contains model links
        parent_container = None
        
        # Try to find container with bg-surface class that has many model links
        bg_surface_divs = driver.find_elements(By.CSS_SELECTOR, 'div.bg-surface')
        
        for div in bg_surface_divs:
            # Check if this div contains model links
            test_links = div.find_elements(By.CSS_SELECTOR, 'a[href*="/sell-old-mobile-phone/used-"]')
            if len(test_links) > 5:  # Reasonable threshold for a brand page
                parent_container = div
                print(f'Found parent container with {len(test_links)} model links')
                break
        
        # If parent container found, extract from it; otherwise extract from entire page
        if parent_container:
            model_links = parent_container.find_elements(By.CSS_SELECTOR, 'a[href*="/sell-old-mobile-phone/used-"]')
            print(f'Extracting {len(model_links)} models from parent container')
        else:
            # Fallback: find all model links, but filter by checking if they're in model divs
            # Model divs have classes like "basis-full" and contain the links
            model_links = driver.find_elements(By.CSS_SELECTOR, 'div.basis-full a[href*="/sell-old-mobile-phone/used-"]')
            if not model_links:
                # Last resort: all links
                model_links = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/sell-old-mobile-phone/used-"]')
            print(f'Extracting {len(model_links)} models (parent container not found, using filtered search)')
        
        duplicates_count = 0
        no_name_count = 0
        
        for link_elem in model_links:
            try:
                href = link_elem.get_attribute('href')
                title = link_elem.get_attribute('title') or ''
                
                if href and '/sell-old-mobile-phone/used-' in href:
                    # Convert to absolute URL if needed
                    full_url = href if href.startswith('http') else urljoin(base_url, href)
                    
                    # Avoid duplicates
                    if full_url not in url_set:
                        url_set.add(full_url)
                        
                        # Get model name from title attribute (most reliable)
                        if title:
                            # Remove "Sell Old " prefix if present
                            model_name = title.replace('Sell Old ', '').strip()
                        else:
                            # Fallback to link text
                            model_name = link_elem.text.strip()
                        
                        if model_name:
                            models.append({
                                'url': full_url,
                                'name': model_name,
                            })
                        else:
                            no_name_count += 1
                    else:
                        duplicates_count += 1
            except Exception as e:
                print(f'Error extracting from link element: {e}')
                continue
        
        if duplicates_count > 0:
            print(f'Filtered out {duplicates_count} duplicate URLs')
        if no_name_count > 0:
            print(f'Filtered out {no_name_count} links without model names')
                
    except Exception as e:
        print(f'Error extracting models via Selenium: {e}')
    
    print(f'Extracted {len(models)} unique models via Selenium')
    return models


def extract_model_urls(
    document: BeautifulSoup,
    base_url: str = 'https://www.cashify.in'
) -> List[Dict[str, str]]:
    """Extract model URLs and names from a brand page HTML.
    
    First finds the parent container, then extracts models from within it.
    """
    models: List[Dict[str, str]] = []
    url_set = set()

    # Find the parent container step by step to avoid issues with escaped CSS selectors
    # Start from #__csh > main, then navigate down
    parent_container = None
    
    try:
        # Step 1: Find the main container
        main_elem = document.select_one('#__csh main')
        if main_elem:
            # Step 2: Find div with bg-surface class (the product grid container)
            # Look for div with classes: bg-surface, rounded-lg, and contains model links
            bg_surface_divs = main_elem.select('div.bg-surface')
            
            for div in bg_surface_divs:
                # Check if this div contains model links
                test_links = div.select('a[href*="/sell-old-mobile-phone/used-"]')
                if len(test_links) > 0:
                    # This is likely our parent container
                    parent_container = div
                    print(f'Found parent container with {len(test_links)} model links')
                    break
            
            # If not found, try finding the direct parent of model links
            if not parent_container:
                # Find a model link first
                sample_link = main_elem.select_one('a[href*="/sell-old-mobile-phone/used-"]')
                if sample_link:
                    # Navigate up to find the container div
                    # Look for parent with classes containing "bg-surface" or "flex flex-col"
                    current = sample_link.parent
                    for _ in range(10):  # Max 10 levels up
                        if current and current.name == 'div':
                            classes = current.get('class', [])
                            class_str = ' '.join(classes) if classes else ''
                            if 'bg-surface' in class_str or ('flex' in class_str and 'flex-col' in class_str):
                                # Check if this container has multiple model links
                                links_in_container = current.select('a[href*="/sell-old-mobile-phone/used-"]')
                                if len(links_in_container) > 5:  # Reasonable threshold
                                    parent_container = current
                                    print(f'Found parent container by navigating up, contains {len(links_in_container)} links')
                                    break
                        if current:
                            current = current.parent
                        else:
                            break
    except Exception as e:
        print(f'Error finding parent container: {e}')
    
    if not parent_container:
        print('Warning: Could not find parent container, falling back to page-wide search')
        # Fallback to original method if parent container not found
        model_links = document.select('a[href*="/sell-old-mobile-phone/used-"]')
    else:
        print(f'Found parent container, searching for models within it...')
        # Find all model divs within the parent container
        # Each model is in a div with classes like "basis-full sm:rounded-lg border-b..."
        # and contains an <a> tag with href="/sell-old-mobile-phone/used-..."
        model_links = parent_container.select('a[href*="/sell-old-mobile-phone/used-"]')
    
    print(f'Found {len(model_links)} model links')

    for link in model_links:
        href = link.get('href')
        if href and '/sell-old-mobile-phone/used-' in href:
            # Convert relative URLs to absolute URLs
            full_url = href if href.startswith('http') else urljoin(base_url, href)
            
            # Avoid duplicates
            if full_url not in url_set:
                url_set.add(full_url)
                
                # Get model name from title attribute (most reliable)
                # Format: "Sell Old Xiaomi Redmi Note 6 Pro" -> "Xiaomi Redmi Note 6 Pro"
                link_title = link.get('title') or ''
                if link_title:
                    # Remove "Sell Old " prefix if present
                    model_name = link_title.replace('Sell Old ', '').strip()
                else:
                    # Fallback to link text
                    link_text = link.get_text(strip=True) or ''
                    model_name = link_text
                
                if model_name:
                    models.append({
                        'url': full_url,
                        'name': model_name,
                    })

    print(f'Extracted {len(models)} unique models')
    return models


def extract_brand_from_url(brand_url: str) -> str:
    """Extract brand name from brand URL.
    
    Example: "https://www.cashify.in/sell-old-mobile-phone/sell-apple" -> "Apple"
    """
    try:
        parsed = urlparse(brand_url)
        path_parts = [p for p in parsed.path.split('/') if p]
        
        # Find the brand part (usually starts with "sell-")
        brand_part = next((p for p in path_parts if p.startswith('sell-')), None)
        if not brand_part:
            return 'Unknown'

        # Remove "sell-" prefix and capitalize
        brand_slug = brand_part.replace('sell-', '')
        brand_name = ' '.join(
            word.capitalize() for word in brand_slug.split('-')
        )
        
        return brand_name
    except Exception as error:
        print(f'Error extracting brand from URL: {error}')
        return 'Unknown'


def scroll_page_completely(driver: webdriver.Chrome, max_scrolls: int = 100) -> None:
    """Scroll down the page completely to load all lazy-loaded content.
    
    Uses slow, incremental scrolling to ensure all content loads properly.
    """
    # Get initial page height
    last_height = driver.execute_script("return document.body.scrollHeight")
    current_position = 0
    scroll_count = 0
    scroll_increment = 500  # Scroll 500px at a time for slow scrolling
    no_change_count = 0  # Track consecutive times with no height change
    
    print(f'Starting scroll: initial page height = {last_height}')
    
    while scroll_count < max_scrolls:
        # Get current scroll position
        current_position = driver.execute_script("return window.pageYOffset || document.documentElement.scrollTop")
        viewport_height = driver.execute_script("return window.innerHeight")
        max_scroll = driver.execute_script("return document.body.scrollHeight - window.innerHeight")
        
        # Slow incremental scroll: scroll down by increment
        new_position = min(current_position + scroll_increment, max_scroll)
        
        # Smooth scroll to new position
        driver.execute_script(f"window.scrollTo({{top: {new_position}, behavior: 'smooth'}});")
        
        # Wait for smooth scroll to complete and content to load
        time.sleep(0.2)  # Reduced wait time for faster scrolling
        
        # Also wait a bit more for lazy-loaded content
        time.sleep(0.1)
        
        # Calculate new scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        # Check if we've reached the bottom
        current_position_after = driver.execute_script("return window.pageYOffset || document.documentElement.scrollTop")
        is_at_bottom = current_position_after >= (new_height - viewport_height - 100)  # 100px threshold
        
        if new_height > last_height:
            # Content loaded, reset no_change_count
            no_change_count = 0
            print(f'Scrolled to {current_position_after}px, page height increased to {new_height}px')
        else:
            no_change_count += 1
        
        # If we're at the bottom and no new content loaded for a few scrolls, try to trigger more
        if is_at_bottom:
            if no_change_count >= 3:
                # Try scrolling a bit more and waiting longer
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(0.3)
                
                # Try scrolling up a bit and back down to trigger lazy loading
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 500);")
                time.sleep(0.2)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(0.3)
                
                # Check final height
                final_height = driver.execute_script("return document.body.scrollHeight")
                if final_height == new_height:
                    # No more content after multiple attempts
                    print(f'No more content to load after {scroll_count} scrolls')
                    break
                else:
                    new_height = final_height
                    no_change_count = 0
        
        last_height = new_height
        scroll_count += 1
        
        # If we're at bottom and no change for many scrolls, break
        if is_at_bottom and no_change_count >= 5:
            print(f'Reached bottom with no new content after {no_change_count} attempts')
            break
    
    # Final scroll to very bottom to ensure everything is loaded
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(0.3)
    
    # Scroll back to top slowly
    print('Scrolling back to top...')
    driver.execute_script("window.scrollTo({top: 0, behavior: 'smooth'});")
    time.sleep(0.3)
    
    final_height = driver.execute_script("return document.body.scrollHeight")
    print(f'Scroll complete: final page height = {final_height}, total scrolls = {scroll_count}')


def scrape_cashify_brand_models(
    brand_url: str
) -> Dict[str, Any]:
    """Scrape a single brand page using Selenium to handle lazy loading.
    
    Opens the page in a browser, scrolls completely to load all content,
    then extracts model URLs with names.
    """
    driver = None
    try:
        print(f'Opening brand page in browser: {brand_url}...')
        
        # Setup Chrome options for headless browsing
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Initialize the driver
        # Always use system chromedriver (works reliably, avoids webdriver-manager issues)
        try:
            # Try system chromedriver first (uses PATH or common locations)
            driver = webdriver.Chrome(options=chrome_options)
            print('Successfully initialized ChromeDriver using system chromedriver')
        except Exception as e:
            # If system chromedriver not found, try common installation paths
            common_paths = [
                '/usr/local/bin/chromedriver',  # Docker/common Linux
                '/usr/bin/chromedriver',         # System installation
                '/opt/chromedriver',             # Alternative location
            ]
            
            driver_initialized = False
            for chromedriver_path in common_paths:
                if os.path.exists(chromedriver_path) and os.access(chromedriver_path, os.X_OK):
                    try:
                        service = Service(chromedriver_path)
                        driver = webdriver.Chrome(service=service, options=chrome_options)
                        print(f'Successfully initialized ChromeDriver using {chromedriver_path}')
                        driver_initialized = True
                        break
                    except Exception:
                        continue
            
            if not driver_initialized:
                raise Exception(f"Failed to initialize ChromeDriver. System chromedriver not found. Error: {e}")
        driver.get(brand_url)
        
        # Wait for initial page load
        print('Waiting for page to load...')
        time.sleep(1)
        
        # Scroll down completely to load all lazy-loaded content
        print('Scrolling page to load all content...')
        scroll_page_completely(driver)
        
        # Save the scraped page data to a txt file
        try:
            # Create scraped_data directory if it doesn't exist
            scraped_dir = 'scraped_data'
            os.makedirs(scraped_dir, exist_ok=True)
            
            # Generate filename from brand URL and timestamp
            brand_slug = extract_brand_from_url(brand_url).lower().replace(' ', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{scraped_dir}/{brand_slug}_{timestamp}.txt'
            
            # Get page source and save to file
            page_source = driver.page_source
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(page_source)
            
            print(f'Saved scraped page data to {filename}')
        except Exception as save_error:
            print(f'Warning: Could not save page source to file: {save_error}')
        
        # Extract models directly from Selenium (more reliable than parsing HTML)
        print('Extracting models from page...')
        models = extract_model_urls_from_selenium(driver)
        
        if not models:
            # Fallback to HTML parsing if Selenium extraction fails
            print('Selenium extraction returned no models, falling back to HTML parsing...')
            html = driver.page_source
            document = parse_html(html)
            if document:
                models = extract_model_urls(document)
        
        print(f'Extracted {len(models)} unique models from {brand_url}')

        return {
            'success': True,
            'models': models,
        }
    except Exception as error:
        error_message = str(error)
        print(f'Error scraping brand page {brand_url}: {error_message}')
        return {
            'success': False,
            'models': [],
            'error': error_message,
        }
    finally:
        # Always close the browser
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


def scrape_cashify_all_brand_models(
    brand_urls: List[str]
) -> Dict[str, Any]:
    """Scrape multiple brand pages and extract all model URLs with brand and model names."""
    results: List[Dict[str, Any]] = []
    total_models = 0

    print(f'Starting to scrape {len(brand_urls)} brand pages...')

    # Process all brand URLs
    for brand_url in brand_urls:
        brand_name = extract_brand_from_url(brand_url)
        result = scrape_cashify_brand_models(brand_url)
        
        if result['success']:
            results.append({
                'brand_url': brand_url,
                'brand_name': brand_name,
                'model_urls': result['models'],
            })
            total_models += len(result['models'])
        else:
            results.append({
                'brand_url': brand_url,
                'brand_name': brand_name,
                'model_urls': [],
                'error': result.get('error'),
            })

    print(f'Completed scraping. Total models found: {total_models}')

    return {
        'success': True,
        'models': results,
        'total_models': total_models,
    }


def parse_price(price_text: str) -> Optional[float]:
    """Parse price string (e.g., "₹2,270") to numeric value."""
    try:
        # Remove ₹, commas, and whitespace, then parse
        cleaned = re.sub(r'[₹,\s]', '', price_text).strip()
        price = float(cleaned)
        return price if not (price != price) else None  # Check for NaN
    except Exception as error:
        print(f'Error parsing price: {error}')
        return None


def extract_price(document: BeautifulSoup) -> Optional[float]:
    """Extract price from model page HTML."""
    try:
        # Try to find the price element using the selector pattern
        # The price is in: <span class="display5 text-error">₹2,270</span>
        price_element = document.select_one('span.display5.text-error')
        
        if price_element:
            price_text = price_element.get_text(strip=True)
            return parse_price(price_text)

        # Fallback: look for any element with "display5" class containing price
        display5_elements = document.select('.display5')
        for el in display5_elements:
            text = el.get_text(strip=True)
            if '₹' in text:
                price = parse_price(text)
                if price is not None:
                    return price

        return None
    except Exception as error:
        print(f'Error extracting price: {error}')
        return None


def extract_variant_urls(
    document: BeautifulSoup,
    base_url: str = 'https://www.cashify.in'
) -> List[str]:
    """Extract variant URLs from model page HTML."""
    variant_urls: List[str] = []

    try:
        # Look for the variants container: ul inside div with "flex-col mb-2 border-2..."
        variant_container = document.select_one('div.flex-col.mb-2.border-2.rounded-lg ul')
        
        if variant_container:
            variant_links = variant_container.select('a[href*="/sell-old-mobile-phone/used-"]')
            
            for link in variant_links:
                href = link.get('href')
                if href:
                    full_url = href if href.startswith('http') else urljoin(base_url, href)
                    variant_urls.append(full_url)
    except Exception as error:
        print(f'Error extracting variant URLs: {error}')

    return variant_urls


def extract_variant_from_url(url: str) -> Optional[str]:
    """Extract variant from URL (only used for variant pages).
    
    Example: https://www.cashify.in/sell-old-mobile-phone/used-iphone-13-128-gb
    Returns: "128 GB" or "3 GB/32 GB"
    """
    try:
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split('/') if p]
        
        # Find the model part (usually contains "used-")
        model_part = next((p for p in path_parts if 'used-' in p), None)
        if not model_part:
            return None

        # Remove "used-" prefix
        model_slug = model_part.replace('used-', '')

        # Try to extract variant patterns (RAM/Storage)
        variant_patterns = [
            r'(\d+)\s*gb\s*/\s*(\d+)\s*gb',  # "3 GB/32 GB"
            r'(\d+)\s*gb\s+(\d+)\s*gb',        # "3 GB 32 GB"
            r'(\d+)\s*gb',                      # "128 GB"
        ]

        for pattern in variant_patterns:
            match = re.search(pattern, model_slug, re.IGNORECASE)
            if match:
                if len(match.groups()) >= 2 and match.group(2):
                    # Has both RAM and storage
                    return f"{match.group(1)} GB/{match.group(2)} GB"
                else:
                    # Only storage
                    return f"{match.group(1)} GB"

        return None
    except Exception as error:
        print(f'Error extracting variant from URL: {error}')
        return None


def scrape_model_price(
    model_url: str
) -> Dict[str, Any]:
    """Scrape a single model page and extract price.
    
    Handles both direct price and variant pages.
    """
    try:
        print(f'Fetching model page: {model_url}...')

        response = fetch_with_browser_headers(model_url)
        response.raise_for_status()

        html = response.text
        document = parse_html(html)
        
        if not document:
            raise Exception("Failed to parse HTML")

        # Check for variants first
        variant_urls = extract_variant_urls(document)
        
        if variant_urls:
            print(f'Found {len(variant_urls)} variants for {model_url}')
            return {
                'success': True,
                'price': None,  # Price will be fetched from variant pages
                'variant_urls': variant_urls,
            }

        # No variants, try to extract price directly
        price = extract_price(document)
        
        if price is None:
            print(f'Could not extract price from {model_url}')
        else:
            print(f'Extracted price ₹{price} from {model_url}')

        return {
            'success': True,
            'price': price,
            'variant_urls': [],
        }
    except Exception as error:
        error_message = str(error)
        print(f'Error scraping model page {model_url}: {error_message}')
        return {
            'success': False,
            'price': None,
            'variant_urls': [],
            'error': error_message,
        }


def scrape_variant_prices(
    variant_urls: List[str]
) -> List[Dict[str, Any]]:
    """Scrape all variants and get prices for each."""
    results: List[Dict[str, Any]] = []

    for variant_url in variant_urls:
        result = scrape_model_price(variant_url)
        
        if result['success'] and result['price'] is not None:
            results.append({
                'url': variant_url,
                'price': result['price'],
            })
        else:
            results.append({
                'url': variant_url,
                'price': None,
                'error': result.get('error', 'Price not found'),
            })
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)

    return results


def process_model_url(
    model_url: str,
    brand: str,
    model_name: str,
    category_id: int
) -> Dict[str, Any]:
    """Process a single model URL: scrape price and return results.
    
    Note: Database saving is excluded as per user request.
    """
    try:
        # Scrape the model page
        model_result = scrape_model_price(model_url)

        if not model_result['success']:
            return {
                'success': False,
                'saved': 0,
                'error': model_result.get('error'),
                'scraped_data': [],
            }

        scraped_data = []
        saved_count = 0

        # If there are variants, process each variant
        if model_result['variant_urls']:
            print(f'Processing {len(model_result["variant_urls"])} variants for {model_url}')
            
            variant_prices = scrape_variant_prices(model_result['variant_urls'])

            for variant_data in variant_prices:
                # Extract variant from variant URL
                variant = extract_variant_from_url(variant_data['url'])
                
                item_data = {
                    'category_id': category_id,
                    'brand': brand,
                    'model': model_name,
                    'variant': variant,
                    'page_url': variant_data['url'],
                    'cashify_price': variant_data['price'],
                    'source': 'cashify',
                }
                
                scraped_data.append(item_data)
                
                if variant_data['price'] is not None:
                    saved_count += 1
        else:
            # No variants, save the main model
            if model_result['price'] is not None:
                item_data = {
                    'category_id': category_id,
                    'brand': brand,
                    'model': model_name,
                    'variant': None,
                    'page_url': model_url,
                    'cashify_price': model_result['price'],
                    'source': 'cashify',
                }
                scraped_data.append(item_data)
                saved_count += 1

        return {
            'success': True,
            'saved': saved_count,
            'scraped_data': scraped_data,
        }
    except Exception as error:
        error_message = str(error)
        print(f'Error processing model URL {model_url}: {error_message}')
        return {
            'success': False,
            'saved': 0,
            'error': error_message,
            'scraped_data': [],
        }


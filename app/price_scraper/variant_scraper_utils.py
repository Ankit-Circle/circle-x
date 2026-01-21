"""
Utility functions for scraping Cashify variant prices.
Uses simple HTTP requests (no Playwright needed).
"""
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL = 'https://www.cashify.in'


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


def fetch_page(url: str) -> Optional[requests.Response]:
    """Fetch a page with browser-like headers."""
    try:
        headers = get_browser_headers()
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response
    except Exception as error:
        print(f'Error fetching {url}: {error}')
        return None


def parse_html(html: str) -> Optional[BeautifulSoup]:
    """Parse HTML string into a BeautifulSoup document."""
    try:
        return BeautifulSoup(html, 'lxml')
    except Exception as error:
        print(f'Error parsing HTML: {error}')
        return None


def has_variants(document: BeautifulSoup) -> bool:
    """Check if the page has multiple variants.
    
    Looks for the "Choose a variant" text.
    """
    try:
        # Look for the variant selector text
        variant_text = document.find('span', class_='subtitle5 mb-2', string='Choose a variant')
        return variant_text is not None
    except Exception:
        return False


def extract_variant_urls(document: BeautifulSoup, base_url: str = BASE_URL) -> List[str]:
    """Extract variant URLs from the model page.
    
    Uses the selector: #__csh > main > div > div:nth-child(3) > div:nth-child(1) > div > div.flex.flex-col.sm\:flex-row.bg-surface.rounded-lg.p-3.sm\:p-6.mt-4.sm\:mt-5.shadow-md > div.w-full.sm\:w-2\/3.flex.flex-col.items-start > div.flex-col.mb-2.border-2.rounded-lg.p-4.border-primary\/50.bg-surface-light.w-full.flex > ul
    """
    variant_urls = []
    
    try:
        # Use the specific selector for variant container
        selector = '#__csh > main > div > div:nth-child(3) > div:nth-child(1) > div > div.flex.flex-col.sm\\:flex-row.bg-surface.rounded-lg.p-3.sm\\:p-6.mt-4.sm\\:mt-5.shadow-md > div.w-full.sm\\:w-2\\/3.flex.flex-col.items-start > div.flex-col.mb-2.border-2.rounded-lg.p-4.border-primary\\/50.bg-surface-light.w-full.flex > ul'
        
        variant_container = document.select_one(selector)
        
        if variant_container:
            # Find all links within the variant container
            variant_links = variant_container.select('a[href]')
            
            for link in variant_links:
                href = link.get('href')
                if href:
                    # Convert to absolute URL
                    full_url = href if href.startswith('http') else urljoin(base_url, href)
                    if full_url not in variant_urls:
                        variant_urls.append(full_url)
        
        print(f'Extracted {len(variant_urls)} variant URLs')
    except Exception as error:
        print(f'Error extracting variant URLs: {error}')
    
    return variant_urls


def extract_price(document: BeautifulSoup) -> Optional[float]:
    """Extract price from the page.
    
    Uses selector: #__csh > main > div > div:nth-child(3) > div:nth-child(1) > div > div.flex.flex-col.sm\:flex-row.bg-surface.rounded-lg.p-3.sm\:p-6.mt-4.sm\:mt-5.shadow-md > div.w-full.sm\:w-2\/3.flex.flex-col.items-start > div.flex.flex-row.mt-5.mb-2 > div.flex.flex-col.flex-1 > div:nth-child(2) > span
    
    Price is in element: <span class="display5 text-error">₹5,950</span>
    """
    try:
        # Use the specific selector for price
        selector = '#__csh > main > div > div:nth-child(3) > div:nth-child(1) > div > div.flex.flex-col.sm\\:flex-row.bg-surface.rounded-lg.p-3.sm\\:p-6.mt-4.sm\\:mt-5.shadow-md > div.w-full.sm\\:w-2\\/3.flex.flex-col.items-start > div.flex.flex-row.mt-5.mb-2 > div.flex.flex-col.flex-1 > div:nth-child(2) > span'
        
        price_element = document.select_one(selector)
        
        if price_element:
            price_text = price_element.get_text(strip=True)
            # Parse price: remove ₹, commas, and whitespace
            cleaned = re.sub(r'[₹,\s]', '', price_text).strip()
            price = float(cleaned)
            return price if not (price != price) else None  # Check for NaN
        
        # Fallback: look for any element with display5 text-error class
        price_elements = document.select('span.display5.text-error')
        for element in price_elements:
            price_text = element.get_text(strip=True)
            if '₹' in price_text:
                cleaned = re.sub(r'[₹,\s]', '', price_text).strip()
                try:
                    price = float(cleaned)
                    return price if not (price != price) else None
                except ValueError:
                    continue
        
        return None
    except Exception as error:
        print(f'Error extracting price: {error}')
        return None


def extract_variant_name(document: BeautifulSoup) -> Optional[str]:
    """Extract variant name from the page.
    
    Uses selector: #__csh > main > div > div:nth-child(3) > div:nth-child(1) > div > div.flex.flex-col.sm\:flex-row.bg-surface.rounded-lg.p-3.sm\:p-6.mt-4.sm\:mt-5.shadow-md > div.w-full.sm\:w-2\/3.flex.flex-col.items-start > div.flex.flex-row.mt-5.mb-2 > div.flex.flex-col.flex-1 > div.pb-4.flex.flex-col > h2
    
    Example: <h2 class="body1">OPPO F21s Pro 5G (8 GB/128 GB)</h2>
    Extracts: "8 GB/128 GB"
    """
    try:
        # Use the specific selector for variant name
        selector = '#__csh > main > div > div:nth-child(3) > div:nth-child(1) > div > div.flex.flex-col.sm\\:flex-row.bg-surface.rounded-lg.p-3.sm\\:p-6.mt-4.sm\\:mt-5.shadow-md > div.w-full.sm\\:w-2\\/3.flex.flex-col.items-start > div.flex.flex-row.mt-5.mb-2 > div.flex.flex-col.flex-1 > div.pb-4.flex.flex-col > h2'
        
        variant_element = document.select_one(selector)
        
        if variant_element:
            variant_text = variant_element.get_text(strip=True)
            # Extract content within brackets: "OPPO F21s Pro 5G (8 GB/128 GB)" -> "8 GB/128 GB"
            match = re.search(r'\(([^)]+)\)', variant_text)
            if match:
                variant_name = match.group(1).strip()
                return variant_name
        
        # Fallback: look for any h2 with body1 class
        h2_elements = document.select('h2.body1')
        for element in h2_elements:
            text = element.get_text(strip=True)
            match = re.search(r'\(([^)]+)\)', text)
            if match:
                variant_name = match.group(1).strip()
                # Check if it contains GB (likely a variant)
                if 'gb' in variant_name.lower():
                    return variant_name
        
        return None
    except Exception as error:
        print(f'Error extracting variant name: {error}')
        return None


def scrape_model_page(model_url: str) -> Dict[str, Any]:
    """Scrape a model page to determine if it has variants or direct price.
    
    Returns:
        {
            'success': bool,
            'has_variants': bool,
            'variant_urls': List[str],  # If has_variants is True
            'price': Optional[float],   # If has_variants is False
            'variant_name': Optional[str],  # If has_variants is False
            'error': Optional[str]
        }
    """
    try:
        print(f'Fetching model page: {model_url}...')
        
        response = fetch_page(model_url)
        if not response:
            return {
                'success': False,
                'has_variants': False,
                'variant_urls': [],
                'price': None,
                'variant_name': None,
                'error': 'Failed to fetch page'
            }
        
        document = parse_html(response.text)
        if not document:
            return {
                'success': False,
                'has_variants': False,
                'variant_urls': [],
                'price': None,
                'variant_name': None,
                'error': 'Failed to parse HTML'
            }
        
        # Check if page has variants
        if has_variants(document):
            # Extract variant URLs
            variant_urls = extract_variant_urls(document)
            print(f'Found {len(variant_urls)} variants for {model_url}')
            
            return {
                'success': True,
                'has_variants': True,
                'variant_urls': variant_urls,
                'price': None,
                'variant_name': None,
            }
        else:
            # No variants, extract price and variant name directly
            price = extract_price(document)
            variant_name = extract_variant_name(document)
            
            print(f'No variants found. Price: ₹{price if price else "N/A"}, Variant: {variant_name or "N/A"}')
            
            return {
                'success': True,
                'has_variants': False,
                'variant_urls': [],
                'price': price,
                'variant_name': variant_name,
            }
            
    except Exception as error:
        error_message = str(error)
        print(f'Error scraping model page {model_url}: {error_message}')
        return {
            'success': False,
            'has_variants': False,
            'variant_urls': [],
            'price': None,
            'variant_name': None,
            'error': error_message,
        }


def scrape_variant_page(variant_url: str) -> Dict[str, Any]:
    """Scrape a variant page to extract price and variant name.
    
    Returns:
        {
            'success': bool,
            'price': Optional[float],
            'variant_name': Optional[str],
            'cf_link': str,  # The variant URL
            'error': Optional[str]
        }
    """
    try:
        print(f'Fetching variant page: {variant_url}...')
        
        # Ensure URL is absolute
        if not variant_url.startswith('http'):
            variant_url = urljoin(BASE_URL, variant_url)
        
        response = fetch_page(variant_url)
        if not response:
            return {
                'success': False,
                'price': None,
                'variant_name': None,
                'cf_link': variant_url,
                'error': 'Failed to fetch page'
            }
        
        document = parse_html(response.text)
        if not document:
            return {
                'success': False,
                'price': None,
                'variant_name': None,
                'cf_link': variant_url,
                'error': 'Failed to parse HTML'
            }
        
        # Extract price and variant name
        price = extract_price(document)
        variant_name = extract_variant_name(document)
        
        print(f'Variant: {variant_name or "N/A"}, Price: ₹{price if price else "N/A"}')
        
        return {
            'success': True,
            'price': price,
            'variant_name': variant_name,
            'cf_link': variant_url,
        }
        
    except Exception as error:
        error_message = str(error)
        print(f'Error scraping variant page {variant_url}: {error_message}')
        return {
            'success': False,
            'price': None,
            'variant_name': None,
            'cf_link': variant_url,
            'error': error_message,
        }

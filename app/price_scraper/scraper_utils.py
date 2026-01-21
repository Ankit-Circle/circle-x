"""
Utility functions for scraping Cashify phone prices.
Converted from TypeScript Supabase Edge Function.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

# Memory tracking
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    try:
        import resource
        HAS_RESOURCE = True
    except ImportError:
        HAS_RESOURCE = False


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        if HAS_PSUTIL:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        elif HAS_RESOURCE:
            # Linux resource module - ru_maxrss is peak memory, not current
            # For current memory approximation, we'll use peak (best we can do without psutil)
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB
        else:
            return 0.0
    except Exception:
        return 0.0


def get_memory_usage_string() -> str:
    """Get current memory usage as formatted string."""
    mem_mb = get_memory_usage_mb()
    if mem_mb > 0:
        return f"{mem_mb:.2f} MB"
    return "N/A"


def parse_html(html: str) -> Optional[BeautifulSoup]:
    """Parse HTML string into a BeautifulSoup document."""
    try:
        return BeautifulSoup(html, 'lxml')
    except Exception as error:
        print(f'Error parsing HTML: {error}')
        return None


def extract_model_urls_from_playwright(
    page: Page,
    base_url: str = 'https://www.cashify.in'
) -> List[Dict[str, str]]:
    """Extract model URLs and names directly from Playwright Page.
    
    This is more reliable than parsing HTML, especially for dynamic content.
    Only extracts models from the specific parent container.
    """
    models: List[Dict[str, str]] = []
    url_set = set()
    
    try:
        # Find the specific target container div
        # Selector: #__csh > main > div > div.min-w-0... > div:nth-child(1) > div > div > div > div.bg-surface... > div > div
        target_selector = '#__csh > main > div > div.min-w-0.flex.flex-row.flex-wrap.sm\\:flex.md\\:flex-wrap.content-start.md\\:m-auto.basis-full.md\\:basis-full.z-auto > div:nth-child(1) > div > div > div > div.bg-surface.rounded-lg.sm\\:rounded-none.mb-3.sm\\:mb-0.mt-4.sm\\:mt-5.overflow-hidden.flex.flex-col > div > div'
        
        parent_container = None
        try:
            parent_container_locator = page.locator(target_selector)
            if parent_container_locator.count() > 0:
                parent_container = parent_container_locator.first
                print(f'Found target container div')
            else:
                print('Could not find target container div')
                print('No models will be extracted')
                return models
        except Exception as e:
            print(f'Could not find target container div: {e}')
            print('No models will be extracted')
            return models
        
        # Extract models only from this specific container
        if parent_container:
            model_links = parent_container.locator('a[href*="/sell-old-mobile-phone/used-"]')
            link_count = model_links.count()
            print(f'Extracting {link_count} models from target container')
        else:
            print('Target container not found, no models extracted')
            return models
        
        duplicates_count = 0
        no_name_count = 0
        
        for i in range(link_count):
            try:
                link_elem = model_links.nth(i)
                href = link_elem.get_attribute('href') or ''
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
                            model_name = link_elem.inner_text().strip()
                        
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
        print(f'Error extracting models via Playwright: {e}')
    
    print(f'Extracted {len(models)} unique models via Playwright')
    return models


def extract_model_urls(
    document: BeautifulSoup,
    base_url: str = 'https://www.cashify.in'
) -> List[Dict[str, str]]:
    """Extract model URLs and names from a brand page HTML.
    
    Only extracts models from the specific target container div.
    """
    models: List[Dict[str, str]] = []
    url_set = set()

    # Find the specific target container div
    # Selector: #__csh > main > div > div.min-w-0... > div:nth-child(1) > div > div > div > div.bg-surface... > div > div
    target_selector = '#__csh > main > div > div.min-w-0.flex.flex-row.flex-wrap.sm\\:flex.md\\:flex-wrap.content-start.md\\:m-auto.basis-full.md\\:basis-full.z-auto > div:nth-child(1) > div > div > div > div.bg-surface.rounded-lg.sm\\:rounded-none.mb-3.sm\\:mb-0.mt-4.sm\\:mt-5.overflow-hidden.flex.flex-col > div > div'
    
    parent_container = None
    
    try:
        parent_container = document.select_one(target_selector)
        if parent_container:
            print(f'Found target container div')
        else:
            print('Could not find target container div, no models will be extracted')
            return models
    except Exception as e:
        print(f'Error finding target container: {e}')
        return models
    
    # Extract models only from this specific container
    if parent_container:
        print(f'Searching for models within target container...')
        model_links = parent_container.select('a[href*="/sell-old-mobile-phone/used-"]')
    else:
        print('Target container not found, no models extracted')
        return models
    
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


def create_browser_context(playwright_instance) -> Tuple[Browser, BrowserContext, Page]:
    """Create and return a Playwright browser, context, and page instance.
    
    Returns a tuple of (browser, context, page) for proper resource management.
    """
    browser = playwright_instance.chromium.launch(
        headless=True,
        args=[
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu',
        ]
    )
    
    context = browser.new_context(
        viewport={'width': 1920, 'height': 1080},
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )
    
    page = context.new_page()
    
    return browser, context, page


def scroll_page_completely(page: Page, max_scrolls: int = 100) -> None:
    """Scroll down the page completely to load all lazy-loaded content.
    
    Uses slow, incremental scrolling to ensure all content loads properly.
    Converted from Selenium version to work with Playwright.
    """
    # Get initial page height
    last_height = page.evaluate("() => document.body.scrollHeight")
    current_position = 0
    scroll_count = 0
    scroll_increment = 500  # Scroll 500px at a time for slow scrolling
    no_change_count = 0  # Track consecutive times with no height change
    
    print(f'Starting scroll: initial page height = {last_height}')
    
    while scroll_count < max_scrolls:
        # Get current scroll position
        current_position = page.evaluate("() => window.pageYOffset || document.documentElement.scrollTop")
        viewport_height = page.evaluate("() => window.innerHeight")
        max_scroll = page.evaluate("() => document.body.scrollHeight - window.innerHeight")
        
        # Slow incremental scroll: scroll down by increment
        new_position = min(current_position + scroll_increment, max_scroll)
        
        # Smooth scroll to new position
        page.evaluate(f"() => window.scrollTo({{top: {new_position}, behavior: 'smooth'}})")
        
        # Wait for smooth scroll to complete and content to load
        time.sleep(0.2)  # Reduced wait time for faster scrolling
        
        # Also wait a bit more for lazy-loaded content
        time.sleep(0.1)
        
        # Calculate new scroll height
        new_height = page.evaluate("() => document.body.scrollHeight")
        
        # Check if we've reached the bottom
        current_position_after = page.evaluate("() => window.pageYOffset || document.documentElement.scrollTop")
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
                page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(0.3)
                
                # Try scrolling up a bit and back down to trigger lazy loading
                page.evaluate("() => window.scrollTo(0, document.body.scrollHeight - 500)")
                time.sleep(0.2)
                page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(0.3)
                
                # Check final height
                final_height = page.evaluate("() => document.body.scrollHeight")
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
    page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(0.3)
    
    # Scroll back to top slowly
    print('Scrolling back to top...')
    page.evaluate("() => window.scrollTo({top: 0, behavior: 'smooth'})")
    time.sleep(0.3)
    
    final_height = page.evaluate("() => document.body.scrollHeight")
    print(f'Scroll complete: final page height = {final_height}, total scrolls = {scroll_count}')


def scrape_cashify_brand_models(
    brand_url: str,
    page: Optional[Page] = None,
    playwright_instance = None
) -> Dict[str, Any]:
    """Scrape a single brand page using Playwright to handle lazy loading.
    
    Opens the page in a browser, scrolls completely to load all content,
    then extracts model URLs with names.
    
    Args:
        brand_url: URL of the brand page to scrape
        page: Optional existing Playwright page to reuse. If None, creates a new one.
        playwright_instance: Playwright instance (required if page is None).
    """
    should_close_browser = page is None
    browser = None
    context = None
    page_start_time = time.time()
    
    try:
        print(f'Opening brand page in browser: {brand_url}...')
        
        # Create page if not provided
        if page is None:
            if playwright_instance is None:
                raise Exception("playwright_instance is required when page is not provided")
            browser, context, page = create_browser_context(playwright_instance)
            print('Successfully initialized Playwright browser')
        
        # Navigate to page - use 'domcontentloaded' for infinite scroll pages
        # (networkidle would wait forever on infinite scroll)
        print('Loading page...')
        page.goto(brand_url, wait_until='domcontentloaded', timeout=30000)
        print(f'Memory after browser init: {get_memory_usage_string()}')
        
        # Wait for initial page structure to be ready
        print('Waiting for initial page structure...')
        time.sleep(1)
        
        # Scroll down completely to load all lazy-loaded/infinite scroll content
        # This is critical for infinite scroll pages - scrolling triggers content loading
        scroll_page_completely(page)
        
        # Wait after scrolling to ensure all lazy-loaded content is fully rendered
        print('Waiting for all lazy-loaded content to render...')
        time.sleep(1.5)
        print(f'Memory after scrolling: {get_memory_usage_string()}')
        
        # Extract models directly from Playwright (more reliable than parsing HTML)
        print('Extracting models from page...')
        models = extract_model_urls_from_playwright(page)
        
        if not models:
            # Fallback to HTML parsing if Playwright extraction fails
            print('Playwright extraction returned no models, falling back to HTML parsing...')
            html = page.content()
            document = parse_html(html)
            if document:
                models = extract_model_urls(document)
        
        page_elapsed_time = time.time() - page_start_time
        print(f'Extracted {len(models)} unique models from {brand_url}')
        print(f'Memory after extraction: {get_memory_usage_string()}')
        print(f'Time taken to scrape {brand_url}: {page_elapsed_time:.2f} seconds')

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
        # Only close the browser if we created it
        if should_close_browser:
            try:
                if context:
                    context.close()
                if browser:
                    browser.close()
            except Exception:
                pass


def _worker_scrape_brands(brand_urls: List[str]) -> List[Dict[str, Any]]:
    """Worker function that processes multiple brands with a single Playwright browser.
    
    This reuses the same browser/page for all brands in the chunk, significantly
    improving performance by avoiding browser launch/quit overhead.
    """
    playwright_instance = None
    browser = None
    context = None
    page = None
    results = []
    
    try:
        # Create one Playwright instance and browser for this worker thread
        playwright_instance = sync_playwright().start()
        browser, context, page = create_browser_context(playwright_instance)
        print(f'Worker initialized Playwright browser for {len(brand_urls)} brands')
        
        # Process all brands in this chunk with the same page
        for brand_url in brand_urls:
            brand_name = extract_brand_from_url(brand_url)
            result = scrape_cashify_brand_models(brand_url, page=page)
            
            results.append({
                'brand_url': brand_url,
                'brand_name': brand_name,
                'result': result,
            })
    
    except Exception as e:
        print(f'Error in worker thread: {e}')
        # Add error results for any unprocessed brands
        for brand_url in brand_urls:
            brand_name = extract_brand_from_url(brand_url)
            # Check if this brand was already processed
            if not any(r['brand_url'] == brand_url for r in results):
                results.append({
                    'brand_url': brand_url,
                    'brand_name': brand_name,
                    'result': {
                        'success': False,
                        'models': [],
                        'error': str(e),
                    },
                })
    finally:
        # Close the browser and playwright instance when done with all brands in this chunk
        try:
            if context:
                context.close()
            if browser:
                browser.close()
            if playwright_instance:
                playwright_instance.stop()
        except Exception:
            pass
    
    return results


def scrape_cashify_all_brand_models(
    brand_urls: List[str],
    max_workers: int = 2
) -> Dict[str, Any]:
    """Scrape multiple brand pages and extract all model URLs with brand and model names.
    
    Processes brands in parallel (default: 2 at a time) to speed up scraping.
    Each worker thread reuses a single Playwright browser/page for multiple brands, which is
    much faster than launching/quitting the browser for each brand.
    """
    results: List[Dict[str, Any]] = []
    total_models = 0
    
    # Track peak memory usage
    peak_memory_mb = get_memory_usage_mb()
    initial_memory = peak_memory_mb
    
    # Track total scraping time
    scraping_start_time = time.time()

    print(f'Starting to scrape {len(brand_urls)} brand pages in parallel (max {max_workers} workers)...')
    print(f'Initial memory usage: {get_memory_usage_string()}')

    # Split brand URLs into chunks for each worker
    # Each worker will process multiple brands with one driver
    # Distribute brands evenly across workers
    num_workers = min(max_workers, len(brand_urls))
    chunk_size = (len(brand_urls) + num_workers - 1) // num_workers  # Ceiling division
    brand_chunks = [
        brand_urls[i:i + chunk_size] 
        for i in range(0, len(brand_urls), chunk_size)
    ]
    
    print(f'Split {len(brand_urls)} brands into {len(brand_chunks)} chunks for {num_workers} workers')

    # Process brand chunks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks
        future_to_chunk = {
            executor.submit(_worker_scrape_brands, chunk): chunk 
            for chunk in brand_chunks
        }
        
        # Process completed chunks as they finish
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                chunk_results = future.result()
                
                # Process each brand result from the chunk
                for brand_data in chunk_results:
                    result = brand_data['result']
                    
                    # Check memory after each brand completes
                    current_memory = get_memory_usage_mb()
                    if current_memory > peak_memory_mb:
                        peak_memory_mb = current_memory
                    
                    if result['success']:
                        results.append({
                            'brand_url': brand_data['brand_url'],
                            'brand_name': brand_data['brand_name'],
                            'model_urls': result['models'],
                        })
                        total_models += len(result['models'])
                        print(f'✓ Completed {brand_data["brand_name"]}: {len(result["models"])} models found')
                    else:
                        results.append({
                            'brand_url': brand_data['brand_url'],
                            'brand_name': brand_data['brand_name'],
                            'model_urls': [],
                            'error': result.get('error'),
                        })
                        print(f'✗ Failed {brand_data["brand_name"]}: {result.get("error", "Unknown error")}')
            except Exception as e:
                # Handle any exceptions from the parallel execution
                print(f'✗ Exception processing chunk: {e}')
                # Add error results for brands in this chunk
                for brand_url in chunk:
                    brand_name = extract_brand_from_url(brand_url)
                    # Check if this brand was already processed
                    if not any(r['brand_url'] == brand_url for r in results):
                        results.append({
                            'brand_url': brand_url,
                            'brand_name': brand_name,
                            'model_urls': [],
                            'error': str(e),
                        })

    # Final memory check
    final_memory = get_memory_usage_mb()
    if final_memory > peak_memory_mb:
        peak_memory_mb = final_memory
    
    # Calculate total scraping time
    scraping_elapsed_time = time.time() - scraping_start_time

    print(f'Completed scraping. Total models found: {total_models}')
    print(f'Memory usage - Initial: {initial_memory:.2f} MB, Final: {final_memory:.2f} MB, Peak: {peak_memory_mb:.2f} MB')
    print(f'Total time taken to scrape all {len(brand_urls)} brand pages: {scraping_elapsed_time:.2f} seconds ({scraping_elapsed_time/60:.2f} minutes)')

    return {
        'success': True,
        'models': results,
        'total_models': total_models,
        'scraping_time_seconds': round(scraping_elapsed_time, 2),
    }


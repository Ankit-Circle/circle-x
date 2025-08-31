import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Header-based Anti-Detection Configuration
# Enhanced with more sophisticated techniques

# User Agent Rotation - Enhanced for better anti-detection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

# Browser Fingerprinting - Enhanced with more profiles
BROWSER_PROFILES = {
    "chrome_windows": {
        "platform": "Win32",
        "vendor": "Google Inc.",
        "language": "en-US",
        "languages": ["en-US", "en"],
        "doNotTrack": "1",
        "sec_ch_ua": '"Chromium";v="120", "Google Chrome";v="120", "Not_A Brand";v="99"',
        "sec_ch_ua_mobile": "?0",
        "sec_ch_ua_platform": '"Windows"',
        "accept_encoding": "gzip, deflate, br",
        "accept_language": "en-US,en;q=0.9,hi;q=0.8"
    },
    "firefox_windows": {
        "platform": "Win32",
        "vendor": "",
        "language": "en-US",
        "languages": ["en-US", "en"],
        "doNotTrack": "1",
        "sec_ch_ua": None,
        "sec_ch_ua_mobile": None,
        "sec_ch_ua_platform": None,
        "accept_encoding": "gzip, deflate, br",
        "accept_language": "en-US,en;q=0.9,hi;q=0.8"
    },
    "safari_mac": {
        "platform": "MacIntel",
        "vendor": "Apple Computer, Inc.",
        "language": "en-US",
        "languages": ["en-US", "en"],
        "doNotTrack": None,
        "sec_ch_ua": None,
        "sec_ch_ua_mobile": None,
        "sec_ch_ua_platform": None,
        "accept_encoding": "gzip, deflate, br",
        "accept_language": "en-US,en;q=0.9"
    },
    "edge_windows": {
        "platform": "Win32",
        "vendor": "Google Inc.",
        "language": "en-US",
        "languages": ["en-US", "en"],
        "doNotTrack": "1",
        "sec_ch_ua": '"Microsoft Edge";v="120", "Chromium";v="120", "Not_A Brand";v="99"',
        "sec_ch_ua_mobile": "?0",
        "sec_ch_ua_platform": '"Windows"',
        "accept_encoding": "gzip, deflate, br",
        "accept_language": "en-US,en;q=0.9,hi;q=0.8"
    },
    "chrome_linux": {
        "platform": "Linux x86_64",
        "vendor": "Google Inc.",
        "language": "en-US",
        "languages": ["en-US", "en"],
        "doNotTrack": "1",
        "sec_ch_ua": '"Chromium";v="120", "Google Chrome";v="120", "Not_A Brand";v="99"',
        "sec_ch_ua_mobile": "?0",
        "sec_ch_ua_platform": '"Linux"',
        "accept_encoding": "gzip, deflate, br",
        "accept_language": "en-US,en;q=0.9"
    }
}

# Enhanced Anti-Detection Settings
ANTI_DETECTION = {
    "use_cookies": True,
    "use_session_persistence": True,
    "randomize_headers": True,
    "add_viewport_headers": True,
    "use_realistic_delays": True,
    "max_consecutive_failures": 5,
    "session_refresh_on_failure": True
}

# Scraping Configuration - Enhanced for better anti-detection
REQUEST_DELAY = 5  # Increased base delay
RANDOM_DELAY_RANGE = (3, 8)  # Increased random delay range
MAX_RETRIES = 5   # Increased retry attempts
TIMEOUT = 30      # Increased timeout
SESSION_REFRESH_INTERVAL = 5  # Refresh session more frequently
DELAY_ON_FAILURE = 10  # Longer delay on failure

# 91mobiles Configuration
MOBILES91_BASE_URL = "https://www.91mobiles.com"
MOBILES91_BRANDS = {
    "Apple": {
        "url": "https://www.91mobiles.com/apple-mobile-price-list-in-india",
        "search_terms": ["iPhone 15", "iPhone 14", "iPhone 13", "iPhone 12", "iPhone 11"],
        "category": "mobile"
    },
    "Samsung": {
        "url": "https://www.91mobiles.com/samsung-mobile-price-list-in-india",
        "search_terms": ["Galaxy S24", "Galaxy S23", "Galaxy S22", "Galaxy A55", "Galaxy A35"],
        "category": "mobile"
    },
    "Xiaomi": {
        "url": "https://www.91mobiles.com/xiaomi-mobile-price-list-in-india",
        "search_terms": ["Redmi Note", "POCO", "Mi"],
        "category": "mobile"
    },
    "OnePlus": {
        "url": "https://www.91mobiles.com/oneplus-mobile-price-list-in-india",
        "search_terms": ["OnePlus 12", "OnePlus 11", "OnePlus Nord"],
        "category": "mobile"
    },
    "Realme": {
        "url": "https://www.91mobiles.com/realme-mobile-price-list-in-india",
        "search_terms": ["Realme GT", "Realme Number", "Realme Narzo"],
        "category": "mobile"
    }
}

# Amazon Configuration (keeping for reference)
AMAZON_BASE_URL = "https://www.amazon.com"
AMAZON_SEARCH_URL = "https://www.amazon.com/s"

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://sfcogubngefogxboabwq.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNmY29ndWJuZ2Vmb2d4Ym9hYndxIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0ODg0NjUxMCwiZXhwIjoyMDY0NDIyNTEwfQ.sq5GbDfuxrA6GPbcFLd-ZKhU3OpK7cmdoqhGTBbZTFs")

# Scraping Configuration - MOBILE PHONES ONLY
BRANDS = {
    "Apple": {
        "search_terms": ["iPhone 15", "iPhone 14", "iPhone 13", "iPhone 12", "iPhone 11"],
        "category": "mobile"
    },
    "Samsung": {
        "search_terms": ["Galaxy S24", "Galaxy S23", "Galaxy S22", "Galaxy A55", "Galaxy A35"],
        "category": "mobile"
    }
}

# Batch processing
BATCH_SIZE = 50
DELAY_BETWEEN_BATCHES = 6  # Increased delay between brands

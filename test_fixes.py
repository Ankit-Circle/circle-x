#!/usr/bin/env python3
"""
Test script to verify fixes for duplicate processing and memory issues
"""

import requests
import time
import json

def test_batch_processing():
    """Test batch processing to ensure no duplicate processing"""
    base_url = "http://127.0.0.1:5000"
    
    # Test with the same 5 images that were causing issues
    test_images = [
        "https://res.cloudinary.com/dkjsiqjfr/image/upload/v1756293958/product_submissions/img_1756293954238_xniv3z.jpg",
        "https://res.cloudinary.com/dkjsiqjfr/image/upload/v1756293959/product_submissions/img_1756293954239_bj7mqb.jpg", 
        "https://res.cloudinary.com/dkjsiqjfr/image/upload/v1756293964/product_submissions/img_1756293954241_2eeamg.jpg",
        "https://res.cloudinary.com/dkjsiqjfr/image/upload/v1756293961/product_submissions/img_1756293954242_itwuh9.jpg",
        "https://res.cloudinary.com/dkjsiqjfr/image/upload/v1756293966/product_submissions/img_1756293954243_r3q9bf.jpg"
    ]
    
    # Test 1: Normal batch processing
    print("🧪 Test 1: Normal batch processing")
    payload = {"image_urls": test_images}
    
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/api/enhance/batch", json=payload, timeout=300)
        end_time = time.time()
        
        print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('success')}")
            print(f"📈 Total images: {result.get('total_images')}")
            print(f"🎯 Successful: {result.get('successful_images')}")
            
            # Check for duplicates in results
            results = result.get('results', [])
            urls_processed = set()
            duplicates_found = []
            
            for res in results:
                if res.get('image_url') in urls_processed:
                    duplicates_found.append(res.get('image_url'))
                else:
                    urls_processed.add(res.get('image_url'))
            
            if duplicates_found:
                print(f"❌ Duplicates found: {duplicates_found}")
            else:
                print("✅ No duplicate processing detected")
                
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Batch with duplicate URLs (should be deduplicated)
    print("🧪 Test 2: Batch with duplicate URLs (deduplication test)")
    duplicate_payload = {"image_urls": test_images + test_images}  # Duplicate the list
    
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/api/enhance/batch", json=duplicate_payload, timeout=300)
        end_time = time.time()
        
        print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('success')}")
            print(f"📈 Total images requested: {len(duplicate_payload['image_urls'])}")
            print(f"📈 Total images processed: {result.get('total_images')}")
            print(f"🎯 Successful: {result.get('successful_images')}")
            
            # Should process only 5 unique images, not 10
            if result.get('total_images') == 5:
                print("✅ Duplicate URLs properly deduplicated")
            else:
                print(f"❌ Expected 5 images, got {result.get('total_images')}")
                
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

def test_single_image():
    """Test single image processing"""
    print("🧪 Test 3: Single image processing")
    base_url = "http://127.0.0.1:5000"
    
    test_image = "https://res.cloudinary.com/dkjsiqjfr/image/upload/v1756293958/product_submissions/img_1756293954238_xniv3z.jpg"
    
    payload = {"image_url": test_image}
    
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/api/enhance/", json=payload, timeout=120)
        end_time = time.time()
        
        print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('success')}")
            print(f"🔗 Enhanced URL: {result.get('enhanced_image_url', 'N/A')}")
            print(f"⚙️  Factors: {result.get('enhancement_factors', 'N/A')}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Testing Image Enhancement Fixes")
    print("="*50)
    
    # Test single image first
    test_single_image()
    print("\n" + "="*50 + "\n")
    
    # Test batch processing
    test_batch_processing()
    
    print("\n🎉 Testing completed!")

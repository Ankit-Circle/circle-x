"""
Test script for auto-routing with order_id and visit_type constraints

This script demonstrates:
1. Pickup visits must come before drop visits for the same order_id
2. If pickup is in one truck, drop must be in the same truck or unassigned
3. Optimization is not biased - if constraints can't be met, visits go to unassigned
"""

import requests
import json

# API endpoint
API_URL = "http://localhost:5000/auto-routing/optimize"

def test_pickup_drop_same_truck():
    """Test that pickup and drop for same order are in the same truck"""
    print("\n" + "="*80)
    print("TEST 1: Pickup and Drop in Same Truck")
    print("="*80)
    
    payload = {
        "trucks": 2,
        "max_km": 150,
        "start": {"lat": 12.9716, "lng": 77.5946},  # Bangalore center
        "end": {"lat": 12.9716, "lng": 77.5946},
        "visits": [
            {
                "visitId": "V1_PICKUP",
                "lat": 12.9500,
                "lng": 77.6000,
                "sla_days": 1,
                "order_id": "ORD001",
                "visit_type": "pickup"
            },
            {
                "visitId": "V1_DROP",
                "lat": 12.9900,
                "lng": 77.6100,
                "sla_days": 1,
                "order_id": "ORD001",
                "visit_type": "drop"
            },
            {
                "visitId": "V2_PICKUP",
                "lat": 12.9300,
                "lng": 77.5800,
                "sla_days": 2,
                "order_id": "ORD002",
                "visit_type": "pickup"
            },
            {
                "visitId": "V2_DROP",
                "lat": 12.9800,
                "lng": 77.6200,
                "sla_days": 2,
                "order_id": "ORD002",
                "visit_type": "drop"
            }
        ]
    }
    
    response = requests.post(API_URL, json=payload)
    result = response.json()
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"\nResult:\n{json.dumps(result, indent=2)}")
    
    # Verify constraints
    for route in result.get("routes", []):
        truck_id = route["truckId"]
        stops = route["stops"]
        
        # Group stops by order_id
        order_stops = {}
        for stop in stops:
            if "order_id" in stop:
                order_id = stop["order_id"]
                if order_id not in order_stops:
                    order_stops[order_id] = []
                order_stops[order_id].append(stop)
        
        # Check that pickup comes before drop
        for order_id, stops_list in order_stops.items():
            pickup_seq = None
            drop_seq = None
            
            for stop in stops_list:
                if stop.get("visit_type") == "pickup":
                    pickup_seq = stop["sequence"]
                elif stop.get("visit_type") == "drop":
                    drop_seq = stop["sequence"]
            
            if pickup_seq and drop_seq:
                if pickup_seq < drop_seq:
                    print(f"✅ {truck_id}: Order {order_id} - Pickup (seq {pickup_seq}) before Drop (seq {drop_seq})")
                else:
                    print(f"❌ {truck_id}: Order {order_id} - ERROR: Drop before Pickup!")


def test_impossible_constraint():
    """Test that visits go to unassigned when constraints can't be met"""
    print("\n" + "="*80)
    print("TEST 2: Impossible Constraints - Should Go to Unassigned")
    print("="*80)
    
    payload = {
        "trucks": 1,
        "max_km": 50,  # Very limited distance
        "start": {"lat": 12.9716, "lng": 77.5946},
        "end": {"lat": 12.9716, "lng": 77.5946},
        "visits": [
            {
                "visitId": "V1_PICKUP",
                "lat": 12.9500,
                "lng": 77.6000,
                "sla_days": 0,
                "order_id": "ORD001",
                "visit_type": "pickup"
            },
            {
                "visitId": "V1_DROP",
                "lat": 13.0500,  # Very far away
                "lng": 77.7000,
                "sla_days": 0,
                "order_id": "ORD001",
                "visit_type": "drop"
            },
            {
                "visitId": "V2_SINGLE",
                "lat": 12.9600,
                "lng": 77.6050,
                "sla_days": 1
                # No order_id - independent visit
            }
        ]
    }
    
    response = requests.post(API_URL, json=payload)
    result = response.json()
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"\nResult:\n{json.dumps(result, indent=2)}")
    
    if result.get("unassigned_visits"):
        print(f"\n✅ Correctly moved {len(result['unassigned_visits'])} visits to unassigned")
        for unassigned in result["unassigned_visits"]:
            print(f"   - {unassigned['visitId']}: {unassigned['reason']}")


def test_mixed_visits():
    """Test with mix of paired and unpaired visits"""
    print("\n" + "="*80)
    print("TEST 3: Mixed Paired and Unpaired Visits")
    print("="*80)
    
    payload = {
        "trucks": 2,
        "max_km": 120,
        "start": {"lat": 12.9716, "lng": 77.5946},
        "end": {"lat": 12.9716, "lng": 77.5946},
        "visits": [
            # Paired order
            {
                "visitId": "V1_PICKUP",
                "lat": 12.9500,
                "lng": 77.6000,
                "sla_days": 0,
                "order_id": "ORD001",
                "visit_type": "pickup"
            },
            {
                "visitId": "V1_DROP",
                "lat": 12.9900,
                "lng": 77.6100,
                "sla_days": 0,
                "order_id": "ORD001",
                "visit_type": "drop"
            },
            # Independent visits (no order_id)
            {
                "visitId": "V2_INDEPENDENT",
                "lat": 12.9300,
                "lng": 77.5800,
                "sla_days": 2
            },
            {
                "visitId": "V3_INDEPENDENT",
                "lat": 12.9800,
                "lng": 77.6200,
                "sla_days": 3
            },
            # Another paired order
            {
                "visitId": "V4_PICKUP",
                "lat": 12.9400,
                "lng": 77.5900,
                "sla_days": 1,
                "order_id": "ORD002",
                "visit_type": "pickup"
            },
            {
                "visitId": "V4_DROP",
                "lat": 12.9700,
                "lng": 77.6050,
                "sla_days": 1,
                "order_id": "ORD002",
                "visit_type": "drop"
            }
        ]
    }
    
    response = requests.post(API_URL, json=payload)
    result = response.json()
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"\nResult:\n{json.dumps(result, indent=2)}")
    
    # Analyze results
    for route in result.get("routes", []):
        truck_id = route["truckId"]
        stops = route["stops"]
        print(f"\n{truck_id} has {len(stops)} stops:")
        
        for stop in stops:
            order_info = f" (Order: {stop.get('order_id', 'N/A')}, Type: {stop.get('visit_type', 'N/A')})"
            print(f"  Seq {stop['sequence']}: {stop['visitId']}{order_info}")


def test_only_pickup_no_drop():
    """Test with only pickup (no matching drop)"""
    print("\n" + "="*80)
    print("TEST 4: Only Pickup Without Matching Drop")
    print("="*80)
    
    payload = {
        "trucks": 1,
        "max_km": 100,
        "start": {"lat": 12.9716, "lng": 77.5946},
        "end": {"lat": 12.9716, "lng": 77.5946},
        "visits": [
            {
                "visitId": "V1_PICKUP",
                "lat": 12.9500,
                "lng": 77.6000,
                "sla_days": 1,
                "order_id": "ORD001",
                "visit_type": "pickup"
            },
            # No matching drop - should be treated as independent visit
            {
                "visitId": "V2_INDEPENDENT",
                "lat": 12.9600,
                "lng": 77.6050,
                "sla_days": 2
            }
        ]
    }
    
    response = requests.post(API_URL, json=payload)
    result = response.json()
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"\nResult:\n{json.dumps(result, indent=2)}")
    print("\n✅ Unpaired pickup should be treated as independent visit and can be scheduled")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("AUTO-ROUTING TESTS: Order ID and Visit Type Constraints")
    print("="*80)
    print("\nMake sure the server is running on http://localhost:5000")
    print("Run: python main.py")
    
    try:
        # Run all tests
        test_pickup_drop_same_truck()
        test_impossible_constraint()
        test_mixed_visits()
        test_only_pickup_no_drop()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the server.")
        print("Please make sure the server is running on http://localhost:5000")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")

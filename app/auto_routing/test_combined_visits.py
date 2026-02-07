"""
Test script for combined visits at same location with order constraints

This script demonstrates:
1. Multiple visits at the same lat/lng are combined into one stop
2. Pickup-drop constraints are still respected
3. All individual visits are shown in the output
"""

import requests
import json

# API endpoint
API_URL = "http://localhost:5000/auto-routing/optimize"

def test_combined_visits_same_location():
    """Test that visits at same location are combined"""
    print("\n" + "="*80)
    print("TEST 1: Multiple Visits at Same Location")
    print("="*80)
    
    payload = {
        "trucks": 2,
        "max_km": 150,
        "start": {"lat": 12.9716, "lng": 77.5946},
        "end": {"lat": 12.9716, "lng": 77.5946},
        "visits": [
            # Three visits at the same location
            {
                "visitId": "V1",
                "lat": 12.9500,
                "lng": 77.6000,
                "sla_days": 1,
                "order_id": "ORD001",
                "visit_type": "pickup"
            },
            {
                "visitId": "V2",
                "lat": 12.9500,  # Same location as V1
                "lng": 77.6000,
                "sla_days": 2,
                "order_id": "ORD002",
                "visit_type": "pickup"
            },
            {
                "visitId": "V3",
                "lat": 12.9500,  # Same location as V1 and V2
                "lng": 77.6000,
                "sla_days": 3
                # No order_id - independent visit
            },
            # Drops at different location
            {
                "visitId": "V4_DROP",
                "lat": 12.9900,
                "lng": 77.6100,
                "sla_days": 1,
                "order_id": "ORD001",
                "visit_type": "drop"
            },
            {
                "visitId": "V5_DROP",
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
    
    # Verify that all 3 visits at (12.95, 77.60) are shown in output
    for route in result.get("routes", []):
        truck_id = route["truckId"]
        stops = route["stops"]
        print(f"\n{truck_id} has {len(stops)} stops:")
        
        # Group by sequence to see combined visits
        sequence_groups = {}
        for stop in stops:
            seq = stop['sequence']
            if seq not in sequence_groups:
                sequence_groups[seq] = []
            sequence_groups[seq].append(stop)
        
        for seq, stops_at_seq in sorted(sequence_groups.items()):
            if len(stops_at_seq) > 1:
                print(f"  ✅ Sequence {seq}: {len(stops_at_seq)} visits at same location:")
                for stop in stops_at_seq:
                    order_info = f" (Order: {stop.get('order_id', 'N/A')}, Type: {stop.get('visit_type', 'N/A')})"
                    print(f"     - {stop['visitId']}{order_info}")
            else:
                stop = stops_at_seq[0]
                order_info = f" (Order: {stop.get('order_id', 'N/A')}, Type: {stop.get('visit_type', 'N/A')})"
                print(f"  Sequence {seq}: {stop['visitId']}{order_info}")


def test_pickup_drop_same_location():
    """Test pickup and drop at the same location"""
    print("\n" + "="*80)
    print("TEST 2: Pickup and Drop at Same Location")
    print("="*80)
    
    payload = {
        "trucks": 1,
        "max_km": 100,
        "start": {"lat": 12.9716, "lng": 77.5946},
        "end": {"lat": 12.9716, "lng": 77.5946},
        "visits": [
            {
                "visitId": "PICKUP_ORD001",
                "lat": 12.9500,
                "lng": 77.6000,
                "sla_days": 0,
                "order_id": "ORD001",
                "visit_type": "pickup"
            },
            {
                "visitId": "DROP_ORD001",
                "lat": 12.9500,  # Same location as pickup!
                "lng": 77.6000,
                "sla_days": 0,
                "order_id": "ORD001",
                "visit_type": "drop"
            },
            {
                "visitId": "INDEPENDENT_V1",
                "lat": 12.9600,
                "lng": 77.6050,
                "sla_days": 1
            }
        ]
    }
    
    response = requests.post(API_URL, json=payload)
    result = response.json()
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"\nResult:\n{json.dumps(result, indent=2)}")
    
    print("\n✅ Pickup and drop at same location should both be in the route")
    print("   They should have the same sequence number (same stop)")


def test_multiple_orders_same_location():
    """Test multiple orders with pickups and drops at same locations"""
    print("\n" + "="*80)
    print("TEST 3: Multiple Orders - Pickups and Drops at Same Locations")
    print("="*80)
    
    payload = {
        "trucks": 2,
        "max_km": 150,
        "start": {"lat": 12.9716, "lng": 77.5946},
        "end": {"lat": 12.9716, "lng": 77.5946},
        "visits": [
            # Warehouse A - Multiple pickups
            {
                "visitId": "PICKUP_ORD001",
                "lat": 12.9500,
                "lng": 77.6000,
                "sla_days": 0,
                "order_id": "ORD001",
                "visit_type": "pickup"
            },
            {
                "visitId": "PICKUP_ORD002",
                "lat": 12.9500,  # Same warehouse
                "lng": 77.6000,
                "sla_days": 0,
                "order_id": "ORD002",
                "visit_type": "pickup"
            },
            {
                "visitId": "PICKUP_ORD003",
                "lat": 12.9500,  # Same warehouse
                "lng": 77.6000,
                "sla_days": 1,
                "order_id": "ORD003",
                "visit_type": "pickup"
            },
            # Customer Location B - Multiple drops
            {
                "visitId": "DROP_ORD001",
                "lat": 12.9900,
                "lng": 77.6100,
                "sla_days": 0,
                "order_id": "ORD001",
                "visit_type": "drop"
            },
            {
                "visitId": "DROP_ORD002",
                "lat": 12.9900,  # Same customer location
                "lng": 77.6100,
                "sla_days": 0,
                "order_id": "ORD002",
                "visit_type": "drop"
            },
            # Different drop location
            {
                "visitId": "DROP_ORD003",
                "lat": 12.9800,
                "lng": 77.6200,
                "sla_days": 1,
                "order_id": "ORD003",
                "visit_type": "drop"
            }
        ]
    }
    
    response = requests.post(API_URL, json=payload)
    result = response.json()
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"\nResult:\n{json.dumps(result, indent=2)}")
    
    print("\n✅ Expected behavior:")
    print("   - 3 pickups at (12.95, 77.60) should be combined into 1 stop")
    print("   - 2 drops at (12.99, 77.61) should be combined into 1 stop")
    print("   - Each order's pickup must come before its drop")
    print("   - Each order's pickup and drop must be on the same truck")


def test_efficiency_gain():
    """Test showing efficiency gain from combining visits"""
    print("\n" + "="*80)
    print("TEST 4: Efficiency Gain from Combining Visits")
    print("="*80)
    
    payload = {
        "trucks": 1,
        "max_km": 200,
        "start": {"lat": 12.9716, "lng": 77.5946},
        "end": {"lat": 12.9716, "lng": 77.5946},
        "visits": [
            # 5 visits at location A
            {"visitId": "V1", "lat": 12.9500, "lng": 77.6000, "sla_days": 1},
            {"visitId": "V2", "lat": 12.9500, "lng": 77.6000, "sla_days": 1},
            {"visitId": "V3", "lat": 12.9500, "lng": 77.6000, "sla_days": 1},
            {"visitId": "V4", "lat": 12.9500, "lng": 77.6000, "sla_days": 1},
            {"visitId": "V5", "lat": 12.9500, "lng": 77.6000, "sla_days": 1},
            # 3 visits at location B
            {"visitId": "V6", "lat": 12.9900, "lng": 77.6100, "sla_days": 2},
            {"visitId": "V7", "lat": 12.9900, "lng": 77.6100, "sla_days": 2},
            {"visitId": "V8", "lat": 12.9900, "lng": 77.6100, "sla_days": 2},
            # 2 visits at location C
            {"visitId": "V9", "lat": 12.9300, "lng": 77.5800, "sla_days": 3},
            {"visitId": "V10", "lat": 12.9300, "lng": 77.5800, "sla_days": 3}
        ]
    }
    
    response = requests.post(API_URL, json=payload)
    result = response.json()
    
    print(f"\nStatus Code: {response.status_code}")
    
    total_visits = 10
    unique_locations = 3
    
    print(f"\n📊 Efficiency Analysis:")
    print(f"   Total visits: {total_visits}")
    print(f"   Unique locations: {unique_locations}")
    print(f"   Reduction: {total_visits - unique_locations} fewer stops needed")
    print(f"   Efficiency gain: {((total_visits - unique_locations) / total_visits * 100):.1f}%")
    
    if result.get("routes"):
        route = result["routes"][0]
        print(f"\n   Actual route stops: {route['waypoint_count']}")
        print(f"   Total visits handled: {len(route['stops'])}")
        print(f"   Estimated distance: {route['estimated_km']} km")
        
        # Show combined stops
        sequence_groups = {}
        for stop in route['stops']:
            seq = stop['sequence']
            if seq not in sequence_groups:
                sequence_groups[seq] = []
            sequence_groups[seq].append(stop['visitId'])
        
        print(f"\n   Combined stops:")
        for seq, visit_ids in sorted(sequence_groups.items()):
            print(f"     Stop {seq}: {len(visit_ids)} visits - {', '.join(visit_ids)}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("AUTO-ROUTING TESTS: Combined Visits at Same Location")
    print("="*80)
    print("\nMake sure the server is running on http://localhost:5000")
    
    try:
        test_combined_visits_same_location()
        test_pickup_drop_same_location()
        test_multiple_orders_same_location()
        test_efficiency_gain()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the server.")
        print("Please make sure the server is running on http://localhost:5000")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

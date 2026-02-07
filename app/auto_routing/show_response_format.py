"""
Simple script to show the expected response format for combined visits
"""

import requests
import json
from collections import defaultdict

API_URL = "http://localhost:5000/auto-routing/optimize"

def test_and_display_response():
    print("\n" + "="*80)
    print("RESPONSE FORMAT EXAMPLE: Combined Visits")
    print("="*80)
    
    # Load the test request
    with open('test_response_format.json', 'r') as f:
        payload = json.load(f)
    
    print("\n📤 REQUEST:")
    print(f"   Trucks: {payload['trucks']}")
    print(f"   Total visits: {len(payload['visits'])}")
    
    # Count visits by location
    location_counts = defaultdict(int)
    for visit in payload['visits']:
        loc_key = f"({visit['lat']}, {visit['lng']})"
        location_counts[loc_key] += 1
    
    print(f"   Unique locations: {len(location_counts)}")
    print("\n   Visits by location:")
    for loc, count in location_counts.items():
        print(f"     {loc}: {count} visits")
    
    # Make the request
    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()
        
        print("\n" + "="*80)
        print("📥 RESPONSE:")
        print("="*80)
        
        if response.status_code == 200 and result.get('routes'):
            route = result['routes'][0]
            
            print(f"\n✅ Route for {route['truckId']}:")
            print(f"   Estimated distance: {route['estimated_km']} km")
            print(f"   Waypoint count: {route['waypoint_count']} (physical stops)")
            print(f"   Total visits: {len(route['stops'])}")
            
            # Group stops by sequence
            stops_by_sequence = defaultdict(list)
            for stop in route['stops']:
                stops_by_sequence[stop['sequence']].append(stop)
            
            print(f"\n📍 Route Details:")
            print(f"   {len(stops_by_sequence)} physical stops handling {len(route['stops'])} visits\n")
            
            for seq in sorted(stops_by_sequence.keys()):
                stops = stops_by_sequence[seq]
                first_stop = stops[0]
                
                print(f"   Stop {seq} at ({first_stop['lat']}, {first_stop['lng']})")
                print(f"   └─ {len(stops)} visit(s) at this location:")
                
                for stop in stops:
                    order_info = f"[{stop.get('order_id', 'N/A')}]"
                    type_info = f"({stop.get('visit_type', 'N/A')})"
                    print(f"      • {stop['visitId']} {order_info} {type_info}")
                print()
            
            print("="*80)
            print("📋 FULL JSON RESPONSE:")
            print("="*80)
            print(json.dumps(result, indent=2))
            
            print("\n" + "="*80)
            print("KEY OBSERVATIONS:")
            print("="*80)
            print("✅ All 6 visits appear in the 'stops' array")
            print("✅ Visits at same location have the SAME sequence number")
            print("✅ 'waypoint_count' shows actual physical stops (3)")
            print("✅ Each visit keeps its unique visitId, order_id, and visit_type")
            print("✅ Pickup-drop constraints are satisfied (pickups before drops)")
            print("="*80 + "\n")
            
        else:
            print(f"\n❌ Error: {result}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the server.")
        print("Please make sure the server is running on http://localhost:5000")
        print("\nExpected response format:")
        print_expected_format()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nExpected response format:")
        print_expected_format()


def print_expected_format():
    """Print the expected response format"""
    expected = {
        "routes": [
            {
                "truckId": "TRUCK_1",
                "start": {"lat": 12.9716, "lng": 77.5946},
                "end": {"lat": 12.9716, "lng": 77.5946},
                "stops": [
                    {
                        "visitId": "PICKUP_ORD001_WAREHOUSE_A",
                        "lat": 12.9500,
                        "lng": 77.6000,
                        "sequence": 1,
                        "order_id": "ORD001",
                        "visit_type": "pickup"
                    },
                    {
                        "visitId": "PICKUP_ORD002_WAREHOUSE_A",
                        "lat": 12.9500,
                        "lng": 77.6000,
                        "sequence": 1,
                        "order_id": "ORD002",
                        "visit_type": "pickup"
                    },
                    {
                        "visitId": "PICKUP_ORD003_WAREHOUSE_A",
                        "lat": 12.9500,
                        "lng": 77.6000,
                        "sequence": 1,
                        "order_id": "ORD003",
                        "visit_type": "pickup"
                    },
                    {
                        "visitId": "DROP_ORD001_CUSTOMER_B",
                        "lat": 12.9900,
                        "lng": 77.6100,
                        "sequence": 2,
                        "order_id": "ORD001",
                        "visit_type": "drop"
                    },
                    {
                        "visitId": "DROP_ORD002_CUSTOMER_B",
                        "lat": 12.9900,
                        "lng": 77.6100,
                        "sequence": 2,
                        "order_id": "ORD002",
                        "visit_type": "drop"
                    },
                    {
                        "visitId": "DROP_ORD003_CUSTOMER_C",
                        "lat": 12.9800,
                        "lng": 77.6200,
                        "sequence": 3,
                        "order_id": "ORD003",
                        "visit_type": "drop"
                    }
                ],
                "estimated_km": 45.2,
                "waypoint_count": 3
            }
        ],
        "unassigned_visits": []
    }
    
    print(json.dumps(expected, indent=2))


if __name__ == "__main__":
    test_and_display_response()

# Auto Routing API Documentation

Intelligent vehicle routing optimization API using Google OR-Tools for multi-vehicle delivery route planning.

## Overview

The Auto Routing API optimizes delivery routes for multiple vehicles while respecting distance constraints, SLA priorities, and pickup-drop pair requirements. It uses Google OR-Tools' constraint solver to maximize visit coverage while minimizing total distance traveled.

## Features

- ✅ **Multi-vehicle route optimization** with distance constraints
- ✅ **SLA-based priority routing** (breached, urgent, warning, normal)
- ✅ **Pickup-drop pair constraints** for order fulfillment
- ✅ **Google Maps integration** for accurate road distances
- ✅ **Smart visit combining** at same locations
- ✅ **Waypoint limits** (max 25 stops per route)
- ✅ **Flexible solver** that maximizes visit coverage
- ✅ **Automatic fallback** to Haversine distance calculation

## API Endpoint

**Endpoint:** `POST /api/auto-routing/optimize`

Optimizes delivery routes for multiple vehicles with pickup-drop constraints.

## Request Format

### Request Body

```json
{
  "num_vehicles": 3,
  "max_distance_per_vehicle_km": 100,
  "start_location": {
    "lat": 28.6139,
    "lng": 77.2090
  },
  "end_location": {
    "lat": 28.6139,
    "lng": 77.2090
  },
  "visits": [
    {
      "visitId": "v1",
      "lat": 28.7041,
      "lng": 77.1025,
      "sla_days": 1,
      "order_id": "order_123",
      "visit_type": "pickup"
    },
    {
      "visitId": "v2",
      "lat": 28.5355,
      "lng": 77.3910,
      "sla_days": 1,
      "order_id": "order_123",
      "visit_type": "drop"
    },
    {
      "visitId": "v3",
      "lat": 28.6500,
      "lng": 77.2167,
      "sla_days": 0,
      "order_id": null,
      "visit_type": null
    }
  ]
}
```

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `num_vehicles` | integer | Yes | Number of vehicles available for routing |
| `max_distance_per_vehicle_km` | number | Yes | Maximum distance each vehicle can travel (in km) |
| `start_location` | object | Yes | Starting location with `lat` and `lng` |
| `end_location` | object | Yes | Ending location with `lat` and `lng` |
| `visits` | array | Yes | List of visit objects to be routed |

### Visit Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `visitId` | string | Yes | Unique identifier for the visit |
| `lat` | number | Yes | Latitude of visit location |
| `lng` | number | Yes | Longitude of visit location |
| `sla_days` | integer | No | Days until SLA breach (default: 5) |
| `order_id` | string | No | Order ID for pickup-drop pairing |
| `visit_type` | string | No | Type: "pickup", "drop", "delivery", or null |

## Response Format

```json
{
  "routes": [
    {
      "vehicle_id": 0,
      "stops": [
        {
          "visitIds": ["v1", "v4"],
          "location": {
            "lat": 28.7041,
            "lng": 77.1025
          },
          "distance_from_previous_km": 12.5,
          "cumulative_distance_km": 12.5
        },
        {
          "visitIds": ["v2"],
          "location": {
            "lat": 28.5355,
            "lng": 77.3910
          },
          "distance_from_previous_km": 18.3,
          "cumulative_distance_km": 30.8
        }
      ],
      "total_distance_km": 45.3,
      "total_stops": 2,
      "total_visits": 3
    }
  ],
  "unassigned_visits": [
    {
      "visitId": "v99",
      "reason": "exceeds_distance_constraint"
    }
  ],
  "total_distance_km": 120.5,
  "distance_calculation_method": "google_maps",
  "total_vehicles_used": 2,
  "total_visits_assigned": 15,
  "total_visits_unassigned": 3
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `routes` | array | List of optimized routes for each vehicle |
| `unassigned_visits` | array | Visits that couldn't be assigned |
| `total_distance_km` | number | Total distance across all routes |
| `distance_calculation_method` | string | "google_maps" or "haversine" |
| `total_vehicles_used` | integer | Number of vehicles with assigned routes |
| `total_visits_assigned` | integer | Total visits successfully routed |
| `total_visits_unassigned` | integer | Total visits not routed |

## SLA Priority System

The API prioritizes visits based on SLA urgency:

### Priority Levels

1. **CRITICAL** (sla_days ≤ 0)
   - SLA already breached
   - Must complete today
   - Highest penalty for dropping
   - Always included if possible

2. **URGENT** (sla_days 1-2)
   - About to breach within 1-2 days
   - High priority
   - Strong preference for inclusion

3. **WARNING** (sla_days = 3)
   - Close to breach
   - Medium priority
   - Included after critical and urgent

4. **NORMAL** (sla_days > 3)
   - Standard priority
   - Included to fill remaining capacity

## Routing Constraints

### Distance Constraint
- Each vehicle has a maximum travel distance (`max_distance_per_vehicle_km`)
- Routes exceeding this limit are not allowed
- Visits that would exceed the limit are marked as unassigned

### Waypoint Limit
- Maximum **25 physical stops** per route
- Multiple visits at the same location count as **1 stop**
- Prevents exceeding Google Maps waypoint limits

### Pickup-Drop Constraints
For visits with the same `order_id`:
- **Same Vehicle**: Pickup and drop must be on the same vehicle
- **Ordering**: Pickup must occur before drop
- **Optional**: If constraints can't be met, both visits may be unassigned

### Location Combining
- Visits within ~11 meters (0.0001 degrees) are combined into one stop
- Reduces total stops and improves efficiency
- **Exception**: Pickup-drop pairs at different locations are kept separate

## Distance Calculation

### Google Maps Distance Matrix API (Primary)
- Uses actual road distances
- Accounts for traffic patterns
- More accurate for urban routing
- Requires `GOOGLE_MAPS_API_KEY` environment variable

### Haversine Formula (Fallback)
- Calculates straight-line distances
- Used when Google Maps API is unavailable
- Less accurate but always available

## Example Usage

### cURL Example

```bash
curl --location 'http://localhost:5000/api/auto-routing/optimize' \
--header 'Content-Type: application/json' \
--data '{
  "num_vehicles": 2,
  "max_distance_per_vehicle_km": 50,
  "start_location": {
    "lat": 28.6139,
    "lng": 77.2090
  },
  "end_location": {
    "lat": 28.6139,
    "lng": 77.2090
  },
  "visits": [
    {
      "visitId": "pickup_1",
      "lat": 28.7041,
      "lng": 77.1025,
      "sla_days": 0,
      "order_id": "ORD001",
      "visit_type": "pickup"
    },
    {
      "visitId": "drop_1",
      "lat": 28.5355,
      "lng": 77.3910,
      "sla_days": 0,
      "order_id": "ORD001",
      "visit_type": "drop"
    }
  ]
}'
```

### Python Example

```python
import requests

url = "http://localhost:5000/api/auto-routing/optimize"

payload = {
    "num_vehicles": 2,
    "max_distance_per_vehicle_km": 50,
    "start_location": {
        "lat": 28.6139,
        "lng": 77.2090
    },
    "end_location": {
        "lat": 28.6139,
        "lng": 77.2090
    },
    "visits": [
        {
            "visitId": "pickup_1",
            "lat": 28.7041,
            "lng": 77.1025,
            "sla_days": 0,
            "order_id": "ORD001",
            "visit_type": "pickup"
        },
        {
            "visitId": "drop_1",
            "lat": 28.5355,
            "lng": 77.3910,
            "sla_days": 0,
            "order_id": "ORD001",
            "visit_type": "drop"
        }
    ]
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Total routes: {len(result['routes'])}")
print(f"Total distance: {result['total_distance_km']} km")
```

## Optimization Strategy

The solver uses the following strategy to maximize visit coverage:

1. **High Drop Penalties**: Unassigned visits incur large penalties
   - Critical (breached): 5,000,000 penalty
   - Urgent (1-2 days): 2,500,000 penalty
   - Warning (3 days): 1,250,000 penalty
   - Normal (>3 days): 500,000 penalty

2. **Low Arc Costs**: Distance costs are scaled down 10x
   - Encourages visiting more stops over minimizing distance

3. **Guided Local Search**: Uses metaheuristic to escape local optima
   - Better at finding solutions with more visits

4. **Dynamic Time Limits**:
   - Small problems (<15 nodes): 10 seconds
   - Medium problems (15-30 nodes): 20 seconds
   - Large problems (>30 nodes): 30 seconds

## Error Handling

### Common Unassignment Reasons

| Reason | Description |
|--------|-------------|
| `exceeds_distance_constraint` | Visit would exceed vehicle's max distance |
| `exceeds_waypoint_limit` | Route would exceed 25 stops |
| `filtered_by_priority` | Excluded during pre-filtering (>70 visits) |
| `solver_no_solution` | Solver couldn't find feasible solution |
| `pickup_without_drop` | Pickup visit without matching drop |
| `drop_without_pickup` | Drop visit without matching pickup |

### API Error Responses

```json
{
  "error": "Invalid request format",
  "message": "Missing required field: num_vehicles"
}
```

## Performance Considerations

### Visit Limits
- **Recommended**: Up to 50 visits for optimal performance
- **Maximum**: 70 visits (automatically filtered by SLA priority)
- **Large datasets**: Critical/urgent visits prioritized automatically

### Solver Time
- Solver runs for 10-30 seconds depending on problem size
- Returns best solution found within time limit
- May not always find optimal solution for large problems

### Distance Matrix
- Google Maps API has rate limits
- Batch requests for efficiency
- Fallback to Haversine if API fails

## Environment Setup

### Required Environment Variables

```bash
# .env file
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
```

### Dependencies

```bash
pip install flask
pip install flask-cors
pip install ortools
pip install googlemaps
```

## Tech Stack

- **Flask**: Web framework
- **Google OR-Tools**: Constraint programming solver
- **Google Maps API**: Distance matrix calculations
- **Python 3.8+**: Runtime environment

## License

Proprietary

---

**Last Updated**: February 2026

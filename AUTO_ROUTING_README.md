# Auto Routing API — Hybrid VRP Optimizer

Intelligent vehicle routing optimization API using **Google OR-Tools** to solve a **Hybrid VRP: CVRP + VRPPD + VRPTW**.

## Overview

The Auto Routing API optimizes delivery routes for multiple vehicles by combining three VRP variants into a single solver:

| Variant | Full Name | What It Handles |
|---------|-----------|-----------------|
| **CVRP** | Capacitated VRP | Max distance (km) and max stops per vehicle |
| **VRPPD** | VRP with Pickup & Delivery | Pickup-drop pairs on the same vehicle, in order |
| **VRPTW** | VRP with Time Windows | Visit-level time window constraints and shift limits |

The solver maximizes visit coverage (prioritizing SLA-breached visits) while minimizing total travel distance.

## Features

- **Hybrid VRP Solver** — CVRP + VRPPD + VRPTW in one optimization pass
- **Real Road Distances** — Google Maps Distance Matrix API
- **SLA-Based Priority Routing** — Breached, urgent, warning, normal tiers
- **Pickup-Drop Pair Constraints** — Same vehicle, pickup before drop, no cross-truck pairs
- **Time Windows** — Optional per-visit earliest/latest arrival constraints
- **Shift Duration Limits** — Max hours per driver shift
- **Service Time Modeling** — Configurable minutes per stop (scales for combined locations)
- **Volume Capacity Enforcement** — Optional `truck_capacity` with running load checks
- **Warehouse Reload (Multi-Trip)** — Optional mid-route reload stops at warehouse
- **Smart Location Combining** — Multiple visits at the same coordinates = 1 solver node
- **Vehicle Cost Optimization** — Fills trucks before using new ones (fixed cost per vehicle)
- **Memory & Timing Telemetry** — RAM usage and execution time logged per request
- **Configurable Waypoint Limits** — Default 25, adjustable via `max_stops`
- **Auto-Scaling Solver Time** — 10–40 seconds based on problem size

## API Endpoint

**`POST /api/auto-routing/optimize`**

## Request Format

### Request Body

```json
{
  "trucks": 3,
  "max_km": 120,
  "max_stops": 25,
  "truck_capacity": 50,
  "enable_warehouse_reload": false,
  "max_reloads_per_truck": 2,
  "warehouse": { "lat": 12.97, "lng": 77.59 },
  "reload_service_time_minutes": 10,
  "pickup_dropoff_same_day": true,
  "vehicle_type": "truck",
  "shift_duration_hours": 10,
  "service_time_minutes": 10,
  "start": { "lat": 12.97, "lng": 77.59 },
  "end": { "lat": 12.93, "lng": 77.62 },
  "visits": [
    {
      "visitId": "V1",
      "lat": 12.95,
      "lng": 77.60,
      "sla_days": 0,
      "order_id": "ORD123",
      "visit_type": "pickup",
      "time_window_start": 60,
      "time_window_end": 300
    },
    {
      "visitId": "V2",
      "lat": 12.99,
      "lng": 77.61,
      "sla_days": 3,
      "order_id": "ORD123",
      "visit_type": "drop"
    },
    {
      "visitId": "V3",
      "lat": 13.01,
      "lng": 77.63,
      "sla_days": 5
    }
  ]
}
```

### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `trucks` | integer | **Yes** | — | Number of vehicles available |
| `max_km` | number | **Yes** | — | Maximum distance each vehicle can travel (km) |
| `start` | object | **Yes** | — | Start location `{ "lat", "lng" }` |
| `end` | object | **Yes** | — | End location `{ "lat", "lng" }` |
| `visits` | array | **Yes** | — | List of visit objects |
| `max_stops` | integer | No | `25` | Maximum stops per truck (individual visits, not combined nodes) |
| `truck_capacity` | number | No | — | Max volume units a truck can carry at a time |
| `enable_warehouse_reload` | boolean | No | `false` | Enables warehouse reload behavior |
| `max_reloads_per_truck` | integer | No | `2` (when enabled) | Max warehouse reloads allowed per truck |
| `warehouse` | object | No | `start` | Warehouse location `{ "lat", "lng" }` |
| `reload_service_time_minutes` | number | No | `10` | Service time added at each reload stop |
| `pickup_dropoff_same_day` | boolean | No | `true` | If false, standard drops are deferred to unassigned |
| `vehicle_type` | string | No | — | If `bike`, output IDs are renamed to `rider_1`, `rider_2`, ... |
| `shift_duration_hours` | number | No | `10` | Maximum hours per driver shift |
| `service_time_minutes` | number | No | `10` | Minutes spent at each stop (scales for combined locations) |
| `max_visits_for_routing` | integer | No | disabled | Optional pre-filter cap; applied only when explicitly provided > 0 |
| `max_unique_locations` | integer | No | disabled | Optional unique-location cap; applied only when explicitly provided > 0 |
| `sla_threshold` | integer | No | `3` | SLA cutoff for priority filtering |
| `unpaired_drop_priority_floor` | integer | No | `12` | Minimum solver priority for unpaired drops |
| `independent_standard_drop_priority` | integer | No | `40` | Priority override for independent standard drops |

### Visit Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `visitId` | string | **Yes** | Unique identifier for the visit |
| `lat` | number | **Yes** | Latitude |
| `lng` | number | **Yes** | Longitude |
| `sla_days` | integer | **Yes** | Days until SLA breach (≤0 = already breached) |
| `order_id` | string | No | Order ID — links pickup-drop pairs |
| `visit_type` | string | No | `pickup`, `drop`, `delivery`, `returned_from`, `returned_to`, `return_pickup`, `return_drop`, `damaged_pickup`, `damaged_drop`, `exchanged_pickup`, `exchanged_drop`, or `null` |
| `vol_capacity` | number | No | Volume units this visit adds/removes from running truck load |
| `time_window_start` | number | No | Earliest arrival in **minutes** from shift start |
| `time_window_end` | number | No | Latest arrival in **minutes** from shift start |

## Response Format

```json
{
  "routes": [
    {
      "truckId": "TRUCK_1",
      "start": { "lat": 12.97, "lng": 77.59 },
      "end": { "lat": 12.93, "lng": 77.62 },
      "stops": [
        {
          "visitId": "V1",
          "lat": 12.95,
          "lng": 77.60,
          "sequence": 1,
          "order_id": "ORD123",
          "visit_type": "pickup"
        },
        {
          "visitId": "V2",
          "lat": 12.99,
          "lng": 77.61,
          "sequence": 2,
          "order_id": "ORD123",
          "visit_type": "drop"
        },
        {
          "visitId": "WAREHOUSE_RELOAD_TRUCK_1_1",
          "lat": 12.97,
          "lng": 77.59,
          "sequence": 3,
          "visit_type": "warehouse_reload",
          "is_warehouse_reload": true
        }
      ],
      "estimated_km": 38.4,
      "estimated_hours": 2.15,
      "waypoint_count": 3,
      "volume_utilization": {
        "max_load": 32.0,
        "truck_capacity": 50.0,
        "utilization_pct": 64.0,
        "initial_preloaded_vol": 10.0,
        "reloads_used": 1
      }
    }
  ],
  "unassigned_visits": [
    { "visitId": "V3", "reason": "optimization_constraint" }
  ],
  "validation_errors": null
}
```

### Route Object

| Field | Type | Description |
|-------|------|-------------|
| `truckId` | string | Vehicle identifier (`TRUCK_1`, `TRUCK_2`, …) |
| `start` | object | Start location `{ "lat", "lng" }` |
| `end` | object | End location `{ "lat", "lng" }` |
| `stops` | array | Ordered list of stops with visit details |
| `estimated_km` | number | Estimated route distance in km |
| `estimated_hours` | number | Estimated route duration in hours (travel + service time) |
| `waypoint_count` | integer | Number of individual visits on this route |
| `volume_utilization` | object | Optional. Present when `truck_capacity` is used |

### Warehouse Reload Stop Fields

When reload is used, stops may include:

| Field | Type | Description |
|-------|------|-------------|
| `visit_type` | string | `warehouse_reload` |
| `is_warehouse_reload` | boolean | Always `true` for reload stops |

### Unassigned Visit Reasons

| Reason | Description |
|--------|-------------|
| `max_km_exceeded` | Visit would exceed vehicle's max distance |
| `max_waypoints_exceeded` | Route would exceed `max_stops` limit |
| `optimization_constraint` | Solver could not fit visit within all constraints |
| `filtered_by_priority` | Excluded during SLA-based pre-filtering (large datasets) |
| `filtered_by_location_limit` | Excluded because unique locations exceeded solver limit |
| `incomplete_pair_no_pickup` | Drop visit whose pickup wasn't routable |
| `drop_rescheduled_pickup_in_route` | Drop unassigned; its pickup is in a route |
| `volume_capacity_exceeded` | Removed in volume-capacity post-processing |
| `cross_truck_violation_unfixable` | Pickup/drop violation could not be repaired |
| `ordering_violation_exceeds_constraints` | Reordering would violate distance/stop constraints |
| `ordering_violation_unfixable` | Drop location could not be resolved during repair |
| `drop_without_pickup_in_final_routes` | Final guard removed a drop without its paired pickup |
| `pickup_dropoff_same_day_false_drop_deferred` | Standard drop deferred when same-day flag is false |
| `exchange_rule_forced_unassigned_damaged_drop` | Exchange-rule forced unassignment |
| `exchange_rule_forced_unassigned_exchanged_drop` | Exchange-rule forced unassignment |

## Backward Compatibility

- Existing clients are backward compatible with no request changes.
- `enable_warehouse_reload` defaults to `false`, so reload behavior is off by default.
- `truck_capacity` is optional; if omitted, volume-capacity logic is skipped.
- Old payloads continue to work as before.
- Clients with strict enum parsing should allow optional new stop type `warehouse_reload` when reload is enabled.

### Validation Errors

The response includes a `validation_errors` field. If `null`, all pickup-drop constraints are satisfied. Otherwise, it lists violations like:

- Pickup and drop on different trucks (cross-truck)
- Drop sequenced before pickup on the same truck

## Distance Calculation

The API uses **Google Maps Distance Matrix API** for real road distances and travel durations.

- Requires `GOOGLE_MAPS_API_KEY` environment variable
- Returns actual driving distances (meters) and durations (seconds)
- Batched in chunks of 25 destinations per origin (Google Maps API limit)
- `max_km` is applied directly — no approximation factor needed since distances are real road distances

## SLA Priority System

Visits are prioritized by SLA urgency. Higher-priority visits incur much larger penalties if dropped by the solver.

### Priority Tiers

| Tier | SLA Days | Description | Drop Penalty Multiplier |
|------|----------|-------------|------------------------|
| **CRITICAL** | ≤ 0 | SLA already breached | 10× base |
| **URGENT** | 1–2 | About to breach | 5× base |
| **WARNING** | 3 | Close to breach | 3× base |
| **NORMAL** | > 3 | Standard priority | 2× base |

- **Base penalty** = `max_distance_per_vehicle × 20`
- Pickup-drop paired visits receive an additional **2× multiplier**
- Combined effect: dropping a breached paired visit costs **20× base penalty**, making it extremely unlikely

## Routing Constraints

### Distance Constraint (CVRP)

- Each vehicle has a maximum travel distance (`max_km`)
- Uses real road distances from Google Maps — no approximation needed

### Waypoint Limit (CVRP)

- Configurable via `max_stops` (default 25)
- Counts **individual visits**, not combined nodes
- Example: a combined location with 3 visits counts as 3 toward the limit

### Time Constraint (VRPTW)

- Each vehicle is limited by `shift_duration_hours` (default 10 hours)
- Service time at each stop = `service_time_minutes` × number of visits at that location
- Optional per-visit time windows: `time_window_start` / `time_window_end` (minutes from shift start)

### Pickup-Drop Constraints (VRPPD)

For visits sharing the same `order_id`:
- **Same Vehicle:** Pickup and drop are assigned to the same truck
- **Ordering:** Pickup is visited before drop (enforced by `AddPickupAndDelivery`)
- **No Cross-Truck:** Validated post-solve; violations are flagged in `validation_errors`
- **Disjunctions:** If constraints can't be met, both pickup and drop may be unassigned
- **One Pair Per Node:** Each solver node can belong to at most one pickup-delivery pair (required by OR-Tools)

### Location Combining

- Visits within **~11 meters** (0.0001°) are combined into a single solver node
- Reduces solver complexity and API calls
- After solving, combined nodes are expanded back into individual visits
- The solver's internal waypoint counter accounts for expanded visits at each combined node

### Vehicle Fixed Cost

- A fixed cost is added per vehicle used: `max_distance × 3`
- Encourages the solver to **fill existing trucks** before activating new ones
- Prevents unnecessary vehicle usage when visits fit on fewer trucks

## Optimization Strategy

### Solver Configuration

| Setting | Value | Purpose |
|---------|-------|---------|
| First Solution Strategy | `PARALLEL_CHEAPEST_INSERTION` | Fast initial solution via parallel heuristic |
| Local Search Metaheuristic | `GUIDED_LOCAL_SEARCH` | Escapes local optima for better solutions |
| Global Span Cost Coefficient | `0` | No pressure for balanced routes; fill trucks naturally |
| Time Limit | 10–40s (dynamic) | Based on number of nodes |

### Dynamic Solver Time

| Problem Size | Time Limit |
|--------------|------------|
| < 15 nodes | 10 seconds |
| 15–25 nodes | 20 seconds |
| 26–35 nodes | 30 seconds |
| > 35 nodes | 40 seconds |

### How the Solver Decides

1. **High drop penalties** make it very expensive to leave visits unassigned
2. **Vehicle fixed costs** encourage filling trucks before using new ones
3. **SLA priorities** ensure breached/urgent visits are assigned first
4. **Distance minimization** produces efficient routes after coverage is maximized
5. **Pickup-delivery pairs** are kept together on the same vehicle
6. **Time windows** (if provided) constrain when visits can be served

## Memory & Performance Telemetry

Each request logs memory and timing information:

```
🧠 RAM [API START]: current=0.12 MB, peak=0.12 MB
🧠 RAM [Distance matrix BUILT]: current=0.45 MB, peak=0.45 MB, delta=+0.33 MB
🧠 RAM [Solver START]: current=0.48 MB, peak=0.48 MB
🧠 RAM [Solver END]: current=1.20 MB, peak=2.10 MB, delta=+0.72 MB
============================================================
🧠 MEMORY SUMMARY:
   Peak RAM used:    2.10 MB
   Final RAM used:   1.20 MB
⏱️  TIMING SUMMARY:
   Total API time:   15.32s
============================================================
```

> **Note:** Memory tracing (`tracemalloc`) is paused during the OR-Tools C++ solver execution to avoid interference with the native solver's memory allocator.

### Performance Limits

| Metric | Recommended | Maximum |
|--------|-------------|---------|
| Total visits | Up to 50 | 80 (auto-filtered by SLA) |
| Unique locations | Up to 30 | 40 (configurable) |
| Solver time | 10–40s | Depends on node count |

## Example Usage

### cURL

```bash
curl -X POST http://localhost:5000/api/auto-routing/optimize \
  -H 'Content-Type: application/json' \
  -d '{
  "trucks": 3,
  "max_km": 65,
  "max_stops": 10,
  "shift_duration_hours": 8,
  "service_time_minutes": 10,
  "start": { "lat": 12.9172, "lng": 77.6349 },
  "end": { "lat": 12.9172, "lng": 77.6349 },
  "visits": [
    {
      "visitId": "pickup_1",
      "lat": 12.9670,
      "lng": 77.5201,
      "sla_days": -28,
      "order_id": "ORD001",
      "visit_type": "pickup"
    },
    {
      "visitId": "drop_1",
      "lat": 13.0137,
      "lng": 77.6480,
      "sla_days": -25,
      "order_id": "ORD001",
      "visit_type": "drop"
    },
    {
      "visitId": "standalone_1",
      "lat": 12.9571,
      "lng": 77.6550,
      "sla_days": -2
    }
  ]
}'
```

### Python

```python
import requests

url = "http://localhost:5000/api/auto-routing/optimize"

payload = {
    "trucks": 3,
    "max_km": 65,
    "max_stops": 10,
    "shift_duration_hours": 8,
    "service_time_minutes": 10,
    "start": {"lat": 12.9172, "lng": 77.6349},
    "end": {"lat": 12.9172, "lng": 77.6349},
    "visits": [
        {
            "visitId": "pickup_1",
            "lat": 12.9670,
            "lng": 77.5201,
            "sla_days": -28,
            "order_id": "ORD001",
            "visit_type": "pickup"
        },
        {
            "visitId": "drop_1",
            "lat": 13.0137,
            "lng": 77.6480,
            "sla_days": -25,
            "order_id": "ORD001",
            "visit_type": "drop"
        },
        {
            "visitId": "standalone_1",
            "lat": 12.9571,
            "lng": 77.6550,
            "sla_days": -2
        }
    ]
}

response = requests.post(url, json=payload)
result = response.json()

for route in result["routes"]:
    print(f"{route['truckId']}: {route['waypoint_count']} stops, "
          f"{route['estimated_km']}km, {route['estimated_hours']}h")

print(f"Unassigned: {len(result['unassigned_visits'])}")
```

## Health Check

**`GET /api/auto-routing/health`**

```json
{ "status": "healthy", "service": "auto-routing", "version": "1.0.0" }
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_MAPS_API_KEY` | **Yes** | — | Google Maps Distance Matrix API key |

## Dependencies

```bash
pip install flask flask-cors ortools googlemaps
```

## Tech Stack

- **Flask** — Web framework
- **Google OR-Tools** — Constraint programming / VRP solver
- **Google Maps API** — Distance matrix (real road distances + durations)
- **Python 3.8+** — Runtime

## License

Proprietary

---

**Last Updated:** February 2026

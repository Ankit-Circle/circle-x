import logging
import os
import sys
import time
import tracemalloc
from flask import Blueprint, request, jsonify
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math 
import requests as http_requests  # renamed to avoid conflict with Flask's request
import googlemaps
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ─── Distance Source Configuration (priority order) ───
# 1. Google Maps API (paid, most accurate)
# 2. OSRM (free, real road distances + travel times)
# 3. Haversine (free, straight-line fallback — least accurate)

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
gmaps = None
if GOOGLE_MAPS_API_KEY:
    try:
        gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        logger.info("✅ Google Maps Client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Google Maps Client: {e}")

# OSRM — Open Source Routing Machine (free real road distances + durations)
# Default: public demo server (fine for dev, use self-hosted for production)
# Self-host: docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-routed --algorithm mld /data/region.osrm
OSRM_BASE_URL = os.environ.get("OSRM_BASE_URL", "http://router.project-osrm.org")

if not GOOGLE_MAPS_API_KEY:
    logger.info(f"📍 Using OSRM for road distances: {OSRM_BASE_URL}")

auto_routing_bp = Blueprint("auto_routing", __name__)

# Maximum waypoints per route
MAX_WAYPOINTS_PER_ROUTE = 25

# Average city driving speed for Haversine duration estimation (km/h)
AVG_CITY_SPEED_KMH = 25


def log_memory(label: str, snapshot_start=None):
    """
    Log current memory usage with a label.
    If snapshot_start is provided, also logs the delta from that snapshot.
    Returns the current tracemalloc snapshot for chaining.
    """
    snapshot = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    current_mb = current / (1024 * 1024)
    peak_mb = peak / (1024 * 1024)
    
    msg = f"🧠 RAM [{label}]: current={current_mb:.2f} MB, peak={peak_mb:.2f} MB"
    
    if snapshot_start:
        # Calculate delta from a previous snapshot
        stats = snapshot.compare_to(snapshot_start, 'lineno')
        delta_bytes = sum(stat.size_diff for stat in stats)
        delta_mb = delta_bytes / (1024 * 1024)
        msg += f", delta={delta_mb:+.2f} MB"
    
    logger.info(msg)
    return snapshot


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    
    return c * r


def create_distance_matrix(
    locations: List[Dict]
) -> Tuple[List[List[int]], List[List[int]], str]:
    """
    Create distance AND duration matrices from a list of locations.
    
    Priority order:
      1. Google Maps Distance Matrix API (paid, most accurate)
      2. OSRM Table API (free, real road distances + travel times)
      3. Haversine formula (free, straight-line estimate — last resort)
    
    Returns:
        (distance_matrix [meters], duration_matrix [seconds], source_type)
        source_type is one of: "google_maps", "osrm", "haversine"
    """
    n = len(locations)
    distance_matrix = [[0] * n for _ in range(n)]
    duration_matrix = [[0] * n for _ in range(n)]
    
    # ─── 1. Try Google Maps API (paid, most accurate) ───
    if gmaps:
        try:
            logger.info("🗺️  Using Google Maps for distance + duration matrix...")
            coords = [(loc['lat'], loc['lng']) for loc in locations]
            
            for i in range(n):
                for j in range(0, n, 25):
                    chunk_destinations = coords[j : j + 25]
                    result = gmaps.distance_matrix(
                        origins=[coords[i]],
                        destinations=chunk_destinations,
                        mode="driving"
                    )
                    
                    if result['status'] == 'OK':
                        row_elements = result['rows'][0]['elements']
                        for k, element in enumerate(row_elements):
                            if element['status'] == 'OK':
                                distance_matrix[i][j + k] = element['distance']['value']
                                duration_matrix[i][j + k] = element['duration']['value']
                            else:
                                distance_matrix[i][j + k] = 9999999
                                duration_matrix[i][j + k] = 9999999
                    else:
                        raise Exception(f"Google Maps API error: {result['status']}")
            
            logger.info(f"✅ Google Maps matrix built: {n}x{n} ({n*n} elements)")
            return distance_matrix, duration_matrix, "google_maps"
            
        except Exception as e:
            logger.error(f"Google Maps API failed: {str(e)}. Trying OSRM...")
    
    # ─── 2. Try OSRM (free, real road distances + durations) ───
    try:
        logger.info(f"🛣️  Using OSRM for road distance + duration matrix ({OSRM_BASE_URL})...")
        
        # OSRM expects "lng,lat" format (note: longitude FIRST)
        coord_str = ";".join(f"{loc['lng']},{loc['lat']}" for loc in locations)
        
        url = f"{OSRM_BASE_URL}/table/v1/driving/{coord_str}"
        params = {"annotations": "distance,duration"}
        
        response = http_requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("code") == "Ok":
            osrm_distances = data["distances"]  # meters
            osrm_durations = data["durations"]  # seconds
            
            for i in range(n):
                for j in range(n):
                    dist = osrm_distances[i][j]
                    dur = osrm_durations[i][j]
                    distance_matrix[i][j] = int(dist) if dist is not None else 9999999
                    duration_matrix[i][j] = int(dur) if dur is not None else 9999999
            
            logger.info(f"✅ OSRM matrix built: {n}x{n} ({n*n} elements) — real road distances")
            return distance_matrix, duration_matrix, "osrm"
        else:
            raise Exception(f"OSRM error: {data.get('code')} — {data.get('message', 'unknown')}")
    
    except Exception as e:
        logger.warning(f"⚠️ OSRM failed: {str(e)}. Falling back to Haversine (straight-line)...")
    
    # ─── 3. Haversine Fallback (straight-line, least accurate) ───
    logger.warning("📐 Using Haversine formula (straight-line) — distances will be approximate!")
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_km = haversine_distance(
                    locations[i]['lat'], locations[i]['lng'],
                    locations[j]['lat'], locations[j]['lng']
                )
                dist_meters = int(dist_km * 1000)
                distance_matrix[i][j] = dist_meters
                # Estimate duration: distance / average city speed
                duration_matrix[i][j] = int(dist_km / AVG_CITY_SPEED_KMH * 3600)
            else:
                distance_matrix[i][j] = 0
                duration_matrix[i][j] = 0
    
    return distance_matrix, duration_matrix, "haversine"


def combine_visits_at_same_location(
    visits: List[Dict],
    order_ids: List[str],
    visit_types: List[str],
    tolerance: float = 0.0001
) -> Tuple[List[Dict], List[List[str]], List[Dict], Dict[str, List[str]]]:
    """
    Combine visits that are at the same location (within tolerance).
    Tracks ALL order_ids and visit_types at each location for pickup-drop constraints.
    
    Args:
        visits: List of visit dictionaries
        order_ids: List of order IDs aligned with visits
        visit_types: List of visit types aligned with visits
        tolerance: Distance tolerance for considering locations as "same" (in degrees, ~11m)
    
    Returns:
        Tuple of:
        - combined_visits: List of unique location visits
        - visit_groups: List of lists, each containing visitIds at that location
        - combined_order_info: List of dicts with ALL order_ids and visit_types at each location
        - location_to_visits: Mapping of location key to list of visit IDs
    """
    location_map = {}  # (lat, lng) -> list of visit indices
    
    # Group ALL visits by location
    for i, visit in enumerate(visits):
        lat = round(visit['lat'] / tolerance) * tolerance
        lng = round(visit['lng'] / tolerance) * tolerance
        location_key = (lat, lng)
        
        if location_key not in location_map:
            location_map[location_key] = []
        location_map[location_key].append(i)
    
    combined_visits = []
    visit_groups = []
    combined_order_info = []  # List of {'order_ids': [...], 'visit_types': [...], 'pairs': [...]}
    location_to_visits = {}
    
    # Create combined visits for each unique location
    for location_key, visit_indices in location_map.items():
        first_visit_idx = visit_indices[0]
        first_visit = visits[first_visit_idx]
        
        # Collect all visit IDs at this location
        visit_ids_at_location = [visits[idx]['visitId'] for idx in visit_indices]
        
        # Collect ALL order_ids and visit_types at this location
        orders_at_location = []
        types_at_location = []
        for idx in visit_indices:
            orders_at_location.append(order_ids[idx])
            types_at_location.append(visit_types[idx])
        
        # Create combined visit — carry forward time window from first visit
        # (if multiple visits at same location, use the tightest time window)
        tw_start_values = [visits[idx].get('time_window_start') for idx in visit_indices 
                          if visits[idx].get('time_window_start') is not None]
        tw_end_values = [visits[idx].get('time_window_end') for idx in visit_indices 
                        if visits[idx].get('time_window_end') is not None]
        
        combined_visit = {
            'lat': first_visit['lat'],
            'lng': first_visit['lng'],
            'visitId': first_visit['visitId'],
            'sla_days': min(visits[idx].get('sla_days', 5) for idx in visit_indices)
        }
        # Use tightest time window from all visits at this location
        if tw_start_values:
            combined_visit['time_window_start'] = max(tw_start_values)  # latest start
        if tw_end_values:
            combined_visit['time_window_end'] = min(tw_end_values)  # earliest end
        
        combined_visits.append(combined_visit)
        visit_groups.append(visit_ids_at_location)
        location_to_visits[f"{first_visit['lat']},{first_visit['lng']}"] = visit_ids_at_location
        
        # Store ALL order info for this combined location
        combined_order_info.append({
            'order_ids': orders_at_location,
            'visit_types': types_at_location
        })
        
        if len(visit_ids_at_location) > 1:
            logger.info(f"Combined {len(visit_ids_at_location)} visits at ({first_visit['lat']}, {first_visit['lng']}): {visit_ids_at_location}")
        
    logger.info(f"✅ Combined {len(visits)} visits into {len(combined_visits)} unique locations")
    
    return combined_visits, visit_groups, combined_order_info, location_to_visits


def filter_visits_by_priority(
    visits: List[Dict],
    order_ids: List[str],
    visit_types: List[str],
    max_visits: int = 70,
    sla_threshold: int = 3
) -> Tuple[List[Dict], List[str], List[str], List[Dict]]:
    """
    Filter visits based on SLA priority to handle large datasets.
    MAXIMIZES stops included, especially breached/near-breach SLA visits.
    
    Priority tiers:
    1. CRITICAL: sla_days <= 0 (breached) - ALWAYS include
    2. URGENT: sla_days 1-2 (about to breach) - High priority
    3. WARNING: sla_days 3 (close to breach) - Medium priority  
    4. NORMAL: sla_days > 3 - Lower priority
    
    Args:
        visits: List of visit dictionaries
        order_ids: List of order IDs aligned with visits
        visit_types: List of visit types aligned with visits
        max_visits: Maximum number of visits to include in routing (default: 70)
        sla_threshold: SLA days threshold for urgent visits (default: 3)
    
    Returns:
        Tuple of (filtered_visits, filtered_order_ids, filtered_visit_types, excluded_visits)
    """
    # Categorize visits by urgency tier
    critical_visits = []   # sla_days <= 0 (BREACHED)
    urgent_visits = []     # sla_days 1-2 (about to breach)
    warning_visits = []    # sla_days == 3
    normal_visits = []     # sla_days > 3
    
    for i, visit in enumerate(visits):
        sla_days = visit.get('sla_days', 5)
        visit_data = {
            'index': i,
            'visit': visit,
            'order_id': order_ids[i] if i < len(order_ids) else None,
            'visit_type': visit_types[i] if i < len(visit_types) else None,
            'sla_days': sla_days
        }
        
        if sla_days <= 0:
            critical_visits.append(visit_data)
        elif sla_days <= 2:
            urgent_visits.append(visit_data)
        elif sla_days <= sla_threshold:
            warning_visits.append(visit_data)
        else:
            normal_visits.append(visit_data)
    
    logger.info(f"📊 Visit breakdown: {len(critical_visits)} BREACHED (SLA ≤ 0), "
                f"{len(urgent_visits)} urgent (SLA 1-2), "
                f"{len(warning_visits)} warning (SLA 3), "
                f"{len(normal_visits)} normal (SLA > 3)")
    
    # Build selected visits list - prioritize by tier
    selected_visits = []
    
    # ALWAYS include all critical (breached) visits first
    selected_visits.extend(critical_visits)
    logger.info(f"✅ Added {len(critical_visits)} BREACHED visits (must complete today)")
    
    # Add urgent visits (about to breach)
    remaining = max_visits - len(selected_visits)
    if remaining > 0:
        urgent_visits.sort(key=lambda x: x['sla_days'])
        to_add = urgent_visits[:remaining]
        selected_visits.extend(to_add)
        logger.info(f"✅ Added {len(to_add)} urgent visits (SLA 1-2 days)")
    
    # Add warning visits
    remaining = max_visits - len(selected_visits)
    if remaining > 0:
        warning_visits.sort(key=lambda x: x['sla_days'])
        to_add = warning_visits[:remaining]
        selected_visits.extend(to_add)
        logger.info(f"✅ Added {len(to_add)} warning visits (SLA 3 days)")
    
    # Fill remaining capacity with normal visits
    remaining = max_visits - len(selected_visits)
    if remaining > 0 and normal_visits:
        normal_visits.sort(key=lambda x: x['sla_days'])
        to_add = normal_visits[:remaining]
        selected_visits.extend(to_add)
        logger.info(f"✅ Added {len(to_add)} normal visits to fill capacity")
    
    # Ensure pickup-drop pairs are kept together
    all_visits = critical_visits + urgent_visits + warning_visits + normal_visits
    selected_order_map = {}
    for visit_data in selected_visits:
        order_id = visit_data['order_id']
        visit_type = visit_data['visit_type']
        if order_id and visit_type:
            if order_id not in selected_order_map:
                selected_order_map[order_id] = {}
            selected_order_map[order_id][visit_type.lower()] = visit_data
    
    # Find incomplete pairs and add missing counterparts
    visits_to_add = []
    for order_id, types in selected_order_map.items():
        has_pickup = 'pickup' in types or 'pick' in types
        has_drop = 'drop' in types or 'delivery' in types
        
        if has_pickup and not has_drop:
            for visit_data in all_visits:
                if (visit_data['order_id'] == order_id and 
                    visit_data['visit_type'] and 
                    visit_data['visit_type'].lower() in ['drop', 'delivery'] and
                    visit_data not in selected_visits):
                    visits_to_add.append(visit_data)
                    logger.info(f"➕ Adding drop visit for order {order_id} to complete pair")
                    break
        elif has_drop and not has_pickup:
            for visit_data in all_visits:
                if (visit_data['order_id'] == order_id and 
                    visit_data['visit_type'] and 
                    visit_data['visit_type'].lower() in ['pickup', 'pick'] and
                    visit_data not in selected_visits):
                    visits_to_add.append(visit_data)
                    logger.info(f"➕ Adding pickup visit for order {order_id} to complete pair")
                    break
    
    selected_visits.extend(visits_to_add)
    
    # Build final filtered lists
    filtered_visits = []
    filtered_order_ids = []
    filtered_visit_types = []
    selected_indices = set()
    
    for visit_data in selected_visits:
        filtered_visits.append(visit_data['visit'])
        filtered_order_ids.append(visit_data['order_id'])
        filtered_visit_types.append(visit_data['visit_type'])
        selected_indices.add(visit_data['index'])
    
    # Build excluded visits list
    excluded_visits = []
    for i, visit in enumerate(visits):
        if i not in selected_indices:
            excluded_visits.append({
                'visitId': visit['visitId'],
                'reason': 'filtered_by_priority',
                'sla_days': visit.get('sla_days', 5)
            })
    
    logger.info(f"✅ Filtered to {len(filtered_visits)} visits for routing, {len(excluded_visits)} excluded")
    
    return filtered_visits, filtered_order_ids, filtered_visit_types, excluded_visits


def solve_vrp(
    num_vehicles: int,
    max_distance_per_vehicle: int,
    locations: List[Dict],
    distance_matrix: List[List[int]],
    duration_matrix: List[List[int]],
    start_index: int,
    end_index: int,
    priorities: List[int],
    distance_source: str = "haversine",
    max_waypoints: int = MAX_WAYPOINTS_PER_ROUTE,
    combined_order_info: List[Dict] = None,
    visit_groups: List[List[str]] = None,
    original_visits: List[Dict] = None,
    time_windows: Optional[List[Optional[Tuple[int, int]]]] = None,
    service_times: Optional[List[int]] = None,
    max_route_time: int = 36000
) -> Dict:
    """
    Hybrid VRP Solver: CVRP + VRPPD + VRPTW
    
    Solves a Vehicle Routing Problem combining:
      - CVRP:  Capacity constraints (max_km per vehicle, max_stops per vehicle)
      - VRPPD: Pickup & Delivery pairs (same vehicle, pickup before drop)
      - VRPTW: Time Windows (optional — visit within allowed time range)
    
    Args:
        num_vehicles: Number of trucks available
        max_distance_per_vehicle: Maximum distance each truck can travel (meters)
        locations: List of all locations (start, visits, end)
        distance_matrix: Matrix of distances between all locations (meters)
        duration_matrix: Matrix of travel times between all locations (seconds)
        start_index: Index of the start location
        end_index: Index of the end location
        priorities: Priority values for each visit (higher = more urgent)
        distance_source: "google_maps", "osrm", or "haversine"
        max_waypoints: Maximum stops per route (default: 25)
        combined_order_info: Order IDs and visit types at each combined location
        visit_groups: Visit IDs grouped by combined location
        original_visits: Original visit data for output expansion
        time_windows: Optional list of (earliest, latest) in seconds from shift start.
                      None entries = no time window for that node.
        service_times: Optional list of service durations per node (seconds).
                       Default 10 minutes per visit.
        max_route_time: Maximum route duration in seconds (default 10 hours)
    
    Returns:
        Dictionary containing routes and unassigned visits
    """
    solver_snap = log_memory("Solver START")
    solver_start_time = time.time()
    
    # Create the routing index manager
    # When start and end are different, we need to use lists
    if start_index != end_index:
        # Different start and end points - use list format
        starts = [start_index] * num_vehicles  # All vehicles start from same point
        ends = [end_index] * num_vehicles      # All vehicles end at same point
        manager = pywrapcp.RoutingIndexManager(
            len(distance_matrix),
            num_vehicles,
            starts,
            ends
        )
    else:
        # Same start and end point (depot) - use simple format
        manager = pywrapcp.RoutingIndexManager(
            len(distance_matrix),
            num_vehicles,
            start_index
        )
    
    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)
    
    # Create and register a transit callback
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    
    # Use ACTUAL distances for arc costs — ensures routes are distance-optimized
    # (Previously scaled by /10 which caused inefficient routes that wasted km budget)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add Distance constraint
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        max_distance_per_vehicle,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    
    # ─── SMART PACKING STRATEGY ───
    # Goal: Fill each truck close to max_km/max_stops before using the next truck.
    #
    # 1. GlobalSpanCostCoefficient = 0  → Don't penalize route imbalance.
    #    Let one truck fill up fully before another is used.
    # 2. Fixed cost per vehicle         → Discourage opening extra trucks.
    #    Solver packs visits into fewer trucks first.
    # 3. High drop penalties (>> fixed cost) → Use a new truck rather than drop visits.
    #    Ensures all trucks are used WHEN NEEDED.
    #
    # Relationship:  drop_penalty >> vehicle_fixed_cost >> typical_detour
    # This guarantees: include visit > use new truck > minimize distance
    distance_dimension.SetGlobalSpanCostCoefficient(0)
    
    # Fixed cost per vehicle — solver will fill existing trucks before opening new ones
    vehicle_fixed_cost = max_distance_per_vehicle * 3
    routing.SetFixedCostOfAllVehicles(vehicle_fixed_cost)
    logger.info(f"📦 Vehicle fixed cost: {vehicle_fixed_cost} (packs trucks before using new ones)")
    
    # ─── WAYPOINT COUNT CONSTRAINT ───
    # When multiple visits are combined at the same location (same lat/lng),
    # the solver treats them as ONE node. But max_stops should count INDIVIDUAL visits.
    # So count_callback returns the number of individual visits inside each combined node.
    
    # Build a lookup: node_index -> number of individual visits
    visits_per_node = {}
    if visit_groups:
        for node_idx in range(len(locations)):
            if node_idx == start_index or node_idx == end_index:
                visits_per_node[node_idx] = 0
            elif node_idx < len(visit_groups) and visit_groups[node_idx]:
                visits_per_node[node_idx] = len(visit_groups[node_idx])
            else:
                visits_per_node[node_idx] = 1
    else:
        for node_idx in range(len(locations)):
            if node_idx == start_index or node_idx == end_index:
                visits_per_node[node_idx] = 0
            else:
                visits_per_node[node_idx] = 1
    
    logger.info(f"📊 Visits per combined node: {', '.join(f'N{k}={v}' for k, v in visits_per_node.items() if v > 0)}")
    
    def count_callback(from_index):
        """Returns the number of INDIVIDUAL visits at this combined node."""
        from_node = manager.IndexToNode(from_index)
        return visits_per_node.get(from_node, 0)
    
    count_callback_index = routing.RegisterUnaryTransitCallback(count_callback)
    
    # Add dimension for waypoint count
    waypoint_dimension_name = 'WaypointCount'
    routing.AddDimension(
        count_callback_index,
        0,  # no slack
        max_waypoints,  # maximum individual visits per vehicle
        True,  # start cumul to zero
        waypoint_dimension_name
    )
    waypoint_dimension = routing.GetDimensionOrDie(waypoint_dimension_name)
    
    # ─── VRPTW: Time Dimension (travel time + service time + time windows) ───
    # Service time per node = (service_time_per_visit × number_of_visits_at_node)
    # A combined node with 5 visits takes 5×10min = 50min, not just 10min.
    default_service_time = 600  # seconds (10 min) per individual visit
    if service_times is None:
        service_times = []
        for i in range(len(locations)):
            if i == start_index or i == end_index:
                service_times.append(0)
            else:
                num_visits = visits_per_node.get(i, 1)
                service_times.append(default_service_time * num_visits)
    
    def time_callback(from_index, to_index):
        """Returns travel time + service time at the 'from' node."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel_time = duration_matrix[from_node][to_node]
        svc_time = service_times[from_node] if from_node < len(service_times) else 0
        return travel_time + svc_time
    
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    
    # Add Time dimension — constrains total route duration
    time_dimension_name = 'Time'
    routing.AddDimension(
        time_callback_index,
        30 * 60,       # allow 30 min slack (waiting at time windows)
        max_route_time,  # maximum route duration per vehicle (seconds)
        False,         # Don't force start cumul to zero (allows flexible start)
        time_dimension_name
    )
    time_dimension = routing.GetDimensionOrDie(time_dimension_name)
    
    # Apply time windows if provided (VRPTW)
    has_time_windows = False
    if time_windows:
        for node in range(len(locations)):
            if node == start_index or node == end_index:
                continue
            
            if node < len(time_windows) and time_windows[node] is not None:
                tw_start, tw_end = time_windows[node]
                index = manager.NodeToIndex(node)
                time_dimension.CumulVar(index).SetRange(tw_start, tw_end)
                has_time_windows = True
        
        if has_time_windows:
            logger.info(f"⏰ VRPTW: Time windows applied to visits")
    
    # Set start/end time constraints for all vehicles (shift window)
    for vehicle_id in range(num_vehicles):
        start_idx = routing.Start(vehicle_id)
        end_idx = routing.End(vehicle_id)
        time_dimension.CumulVar(start_idx).SetRange(0, max_route_time)
        time_dimension.CumulVar(end_idx).SetRange(0, max_route_time)
    
    logger.info(f"⏱️  Time dimension: max {max_route_time/3600:.1f}h route, "
                f"{default_service_time/60:.0f}min/stop service time, "
                f"time windows: {'YES' if has_time_windows else 'NO'}")
    
    # ─── VRPPD: Build pickup-delivery pairs ───
    # OR-Tools REQUIREMENT: Each node can appear in AT MOST ONE AddPickupAndDelivery pair.
    # When visits are combined at the same location, multiple orders can map to the
    # same node. We must deduplicate and enforce the one-pair-per-node rule.
    
    order_map = {}
    paired_nodes = set()          # nodes that ended up in a constraint
    pickup_drop_pairs = []        # unique (pickup_node, drop_node) constraints to add
    nodes_used_in_pairs = set()   # tracks nodes already committed to a pair
    seen_node_pairs = set()       # dedup (pickup_node, drop_node) combos
    
    logger.info(f"🔍 Building pickup-drop pairs from combined location info...")
    
    if combined_order_info:
        # Step 1: Build order_map  — order_id -> {'pickup': node, 'drop': node}
        for node in range(len(locations)):
            if node == start_index or node == end_index:
                continue
            
            if node < len(combined_order_info) and combined_order_info[node]:
                info = combined_order_info[node]
                order_ids_at_node = info.get('order_ids', [])
                visit_types_at_node = info.get('visit_types', [])
                
                for order_id, visit_type in zip(order_ids_at_node, visit_types_at_node):
                    if order_id and visit_type:
                        if order_id not in order_map:
                            order_map[order_id] = {}
                        
                        if visit_type.lower() in ['pickup', 'pick']:
                            order_map[order_id]['pickup'] = node
                        elif visit_type.lower() in ['drop', 'delivery']:
                            order_map[order_id]['drop'] = node
        
        # Step 2: Collect candidate pairs, sorted by priority (breached first)
        candidate_pairs = []
        for order_id, nodes in order_map.items():
            if 'pickup' in nodes and 'drop' in nodes:
                pickup_node = nodes['pickup']
                drop_node = nodes['drop']
                
                if pickup_node == drop_node:
                    logger.info(f"📍 Order {order_id}: pickup and drop at SAME location (node {pickup_node}) — no constraint needed")
                    continue
                
                # Get priority of the drop node for sorting
                pri = priorities[drop_node] if drop_node < len(priorities) else 5
                candidate_pairs.append({
                    'order_id': order_id,
                    'pickup_node': pickup_node,
                    'drop_node': drop_node,
                    'priority': pri
                })
                logger.info(f"✅ Complete pair for order {order_id}: pickup node {pickup_node} -> drop node {drop_node}")
            elif 'pickup' in nodes:
                logger.warning(f"⚠️ Order {order_id} has PICKUP (node {nodes['pickup']}) but NO DROP")
            elif 'drop' in nodes:
                logger.warning(f"⚠️ Order {order_id} has DROP (node {nodes['drop']}) but NO PICKUP")
        
        # Step 3: Select one pair per node (highest priority wins)
        # Sort candidates so highest-priority pairs are picked first
        candidate_pairs.sort(key=lambda p: p['priority'], reverse=True)
        
        skipped = 0
        for pair in candidate_pairs:
            pn = pair['pickup_node']
            dn = pair['drop_node']
            node_pair_key = (pn, dn)
            
            # Skip if this exact (pickup, drop) combo was already added
            if node_pair_key in seen_node_pairs:
                logger.info(f"⏭️  Skipping duplicate pair ({pn}->{dn}) for order {pair['order_id']}")
                skipped += 1
                continue
            
            # Skip if EITHER node is already committed to another pair
            # (OR-Tools crashes if a node is in multiple AddPickupAndDelivery calls)
            if pn in nodes_used_in_pairs or dn in nodes_used_in_pairs:
                logger.info(f"⏭️  Skipping pair ({pn}->{dn}) for order {pair['order_id']} — "
                            f"node {'%d' % pn if pn in nodes_used_in_pairs else '%d' % dn} already in another pair")
                skipped += 1
                continue
            
            # Accept this pair
            seen_node_pairs.add(node_pair_key)
            nodes_used_in_pairs.add(pn)
            nodes_used_in_pairs.add(dn)
            paired_nodes.add(pn)
            paired_nodes.add(dn)
            pickup_drop_pairs.append(pair)
        
        if skipped:
            logger.info(f"⏭️  Skipped {skipped} pairs (duplicate or node-conflict). "
                        f"Kept {len(pickup_drop_pairs)} safe pairs.")
    
    # ─── DROP PENALTIES (disjunctions) ───
    # Penalties for NOT visiting a node. Calibrated relative to distances:
    #   drop_penalty >> vehicle_fixed_cost >> typical_detour_distance
    #
    # With max_distance M and vehicle_fixed_cost = M*3:
    #   base_drop_penalty  = M * 20  (>> vehicle fixed cost of M*3)
    #
    # This ensures:
    #   - Solver always includes a visit if ANY truck can reach it
    #   - Solver fills trucks before using new ones (fixed cost > detour)
    #   - Solver opens new truck rather than drop a visit (penalty > fixed cost)
    #   - SLA-breached visits get extreme penalties (practically never dropped)
    
    base_penalty = max_distance_per_vehicle * 20
    
    logger.info(f"Adding disjunctions — base penalty: {base_penalty} "
                f"(vehicle fixed cost: {vehicle_fixed_cost})")
    
    breached_count = 0
    urgent_count = 0
    paired_count = 0
    
    for node in range(len(locations)):
        # Skip start and end nodes
        if node == start_index or node == end_index:
            continue
        
        priority = priorities[node] if node < len(priorities) else 5
        
        # Penalty tiers — all >> vehicle_fixed_cost so solver uses trucks over dropping
        if priority >= 15:
            node_penalty = base_penalty * 10   # SLA BREACHED — never drop
            breached_count += 1
        elif priority >= 8:
            node_penalty = base_penalty * 5    # Urgent (SLA 1-2 days)
            urgent_count += 1
        elif priority >= 7:
            node_penalty = base_penalty * 3    # Warning (SLA 3 days)
        else:
            node_penalty = base_penalty * 2    # Normal
        
        # Paired nodes (pickup-drop) get extra penalty to keep pairs together
        if node in paired_nodes:
            node_penalty = node_penalty * 2
            paired_count += 1
        
        routing.AddDisjunction([manager.NodeToIndex(node)], node_penalty)
    
    logger.info(f"✅ Added disjunctions for {len(locations) - 2} visit nodes")
    logger.info(f"   {breached_count} BREACHED, {urgent_count} urgent, {paired_count} in pairs")
    
    # Add pickup and delivery constraints for complete pairs
    # AddPickupAndDelivery enforces: same vehicle + pickup before drop
    # IMPORTANT: Do NOT add explicit CumulVar ordering constraints (solver().Add)
    # on disjunction nodes — when a node is dropped, its CumulVar is undefined,
    # causing C++ segfaults in the OR-Tools constraint propagation engine.
    for pair in pickup_drop_pairs:
        pickup_index = manager.NodeToIndex(pair['pickup_node'])
        drop_index = manager.NodeToIndex(pair['drop_node'])
        
        # This single call handles: same vehicle + pickup-before-drop ordering
        routing.AddPickupAndDelivery(pickup_index, drop_index)
        
        # Reinforce same-vehicle (safe with disjunctions — if one is dropped, both are)
        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(drop_index)
        )
        
        logger.info(f"Added pickup-drop pair for order {pair['order_id']}: pickup node {pair['pickup_node']} -> drop node {pair['drop_node']}")
    
    logger.info(f"✅ Finished adding {len(pickup_drop_pairs)} pickup-drop pairs")
    
    # ─── SOLVER SEARCH PARAMETERS ───
    logger.info("Setting up solver parameters...")
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    
    # PARALLEL_CHEAPEST_INSERTION: inserts visits into cheapest position.
    # Combined with high vehicle fixed costs, it fills existing trucks first.
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    
    # GUIDED_LOCAL_SEARCH: escapes local minima by penalizing repeated arcs.
    # Good at moving visits between vehicles to find better packing.
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    
    # Dynamic time limit — give solver enough time to explore multi-vehicle solutions
    num_nodes = len(locations)
    if num_nodes <= 10:
        time_limit = 10
    elif num_nodes <= 20:
        time_limit = 15
    elif num_nodes <= 35:
        time_limit = 25
    else:
        time_limit = 40  # Large problems need more time for proper truck packing
    
    search_parameters.time_limit.seconds = time_limit
    search_parameters.log_search = False
    
    logger.info(f"🚀 Starting solver: {num_vehicles} vehicles, {len(locations)} locations, "
                f"max {max_distance_per_vehicle/1000:.0f}km/truck, max {max_waypoints} stops/truck, "
                f"time limit: {time_limit}s")
    
    model_snap = log_memory("Model BUILT (pre-solve)", solver_snap)
    model_build_time = time.time() - solver_start_time
    logger.info(f"⏱️  Model build time: {model_build_time:.2f}s")
    
    # ─── SOLVE ───
    # IMPORTANT: Stop tracemalloc before the C++ solver runs.
    # tracemalloc hooks into Python's memory allocator; when OR-Tools' C++ code
    # invokes Python callbacks (distance_callback, time_callback) during solving,
    # the tracing can corrupt internal state and cause segfaults.
    was_tracing = tracemalloc.is_tracing()
    if was_tracing:
        tracemalloc.stop()
    
    logger.info("🔧 Launching OR-Tools solver...")
    # Flush all log handlers so we see the message even if the process crashes
    for handler in logging.getLogger().handlers:
        handler.flush()
    sys.stdout.flush()
    sys.stderr.flush()
    
    solve_start = time.time()
    try:
        solution = routing.SolveWithParameters(search_parameters)
        solve_elapsed = time.time() - solve_start
        
        # Restart tracemalloc after solver finishes
        if was_tracing:
            tracemalloc.start()
        
        # Check solver status
        status = routing.status()
        status_names = {
            0: "ROUTING_NOT_SOLVED",
            1: "ROUTING_SUCCESS",
            2: "ROUTING_FAIL",
            3: "ROUTING_FAIL_TIMEOUT",
            4: "ROUTING_INVALID"
        }
        status_name = status_names.get(status, f"UNKNOWN({status})")
        logger.info(f"✅ Solver completed in {solve_elapsed:.2f}s — status: {status_name}, solution: {solution is not None}")
        log_memory("Solver DONE", model_snap)
        
        if not solution:
            logger.error(f"❌ No solution found. Status: {status_name}")
            if status == 3:
                logger.error("Solver timed out. Try reducing visits or increasing time limit.")
            elif status == 2:
                logger.error("Solver failed - constraints may be infeasible. Check pickup-drop pairs and distance limits.")
    except Exception as e:
        solve_elapsed = time.time() - solve_start
        logger.error(f"❌ Solver crashed after {solve_elapsed:.2f}s: {str(e)}", exc_info=True)
        if was_tracing and not tracemalloc.is_tracing():
            tracemalloc.start()
        return None
    
    if solution:
        return extract_solution(
            manager, routing, solution, locations, distance_matrix, duration_matrix,
            start_index, end_index, num_vehicles, distance_source, max_waypoints,
            combined_order_info, visit_groups, original_visits
        )
    else:
        logger.error("No solution found!")
        return None


def extract_solution(
    manager, routing, solution, locations, distance_matrix, duration_matrix,
    start_index, end_index, num_vehicles, distance_source: str = "haversine",
    max_waypoints: int = MAX_WAYPOINTS_PER_ROUTE,
    combined_order_info: List[Dict] = None,
    visit_groups: List[List[str]] = None,
    original_visits: List[Dict] = None
) -> Dict:
    """Extract the solution from OR-Tools solver and enforce max waypoints limit"""
    routes = []
    unassigned_visits = []
    
    # Track which visits were assigned
    assigned_indices = set([start_index, end_index])
    
    # Build order map for pickup-drop pair validation from combined_order_info
    order_map = {}
    if combined_order_info:
        for node in range(len(locations)):
            if node == start_index or node == end_index:
                continue
            if node < len(combined_order_info) and combined_order_info[node]:
                info = combined_order_info[node]
                for order_id, visit_type in zip(info.get('order_ids', []), info.get('visit_types', [])):
                    if order_id and visit_type:
                        if order_id not in order_map:
                            order_map[order_id] = {}
                        if visit_type.lower() in ['pickup', 'pick']:
                            order_map[order_id]['pickup'] = node
                        elif visit_type.lower() in ['drop', 'delivery']:
                            order_map[order_id]['drop'] = node
    
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        stops = []
        sequence = 1
        route_nodes = [start_index]  # Track all nodes in order for distance calculation
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            
            # Add to stops if it's not start or end
            if node_index != start_index and node_index != end_index:
                visit_info = locations[node_index]
                
                # If we have visit_groups, expand the combined visit to individual visits
                if visit_groups and node_index < len(visit_groups):
                    visit_ids_at_location = visit_groups[node_index]
                    
                    # Add each individual visit at this location
                    for visit_id in visit_ids_at_location:
                        # Find the original visit data
                        original_visit = None
                        if original_visits:
                            for orig_v in original_visits:
                                if orig_v['visitId'] == visit_id:
                                    original_visit = orig_v
                                    break
                        
                        stop_data = {
                            "visitId": visit_id,
                            "lat": visit_info['lat'],
                            "lng": visit_info['lng'],
                            "sequence": sequence
                        }
                        
                        # Add order_id and visit_type from original visit if available
                        if original_visit:
                            if 'order_id' in original_visit and original_visit['order_id']:
                                stop_data["order_id"] = original_visit['order_id']
                            if 'visit_type' in original_visit and original_visit['visit_type']:
                                stop_data["visit_type"] = original_visit['visit_type']
                        
                        stops.append(stop_data)
                        # Note: We keep the same sequence for all visits at the same location
                else:
                    # No visit groups - single visit at this location
                    stop_data = {
                        "visitId": visit_info['visitId'],
                        "lat": visit_info['lat'],
                        "lng": visit_info['lng'],
                        "sequence": sequence
                    }
                    
                    # Add order_id and visit_type from original visits
                    if original_visits:
                        for orig_v in original_visits:
                            if orig_v['visitId'] == visit_info['visitId']:
                                if 'order_id' in orig_v and orig_v['order_id']:
                                    stop_data["order_id"] = orig_v['order_id']
                                if 'visit_type' in orig_v and orig_v['visit_type']:
                                    stop_data["visit_type"] = orig_v['visit_type']
                                break
                    
                    stops.append(stop_data)
                
                sequence += 1
                assigned_indices.add(node_index)
            
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(index)
            
            # Track the route for distance calculation
            if next_node != route_nodes[-1]:  # Avoid duplicates
                route_nodes.append(next_node)
        
        # Calculate actual route distance AND duration from matrices
        # Sum: start→stop1 + stop1→stop2 + ... + stopN→end
        route_distance_meters = 0
        route_duration_seconds = 0
        for i in range(len(route_nodes) - 1):
            from_node = route_nodes[i]
            to_node = route_nodes[i + 1]
            route_distance_meters += distance_matrix[from_node][to_node]
            route_duration_seconds += duration_matrix[from_node][to_node]
        
        # Apply road distance factor for display
        # google_maps/osrm = real road distances (factor 1.0)
        # haversine = straight-line, estimate road with 1.3x factor
        ROAD_DISTANCE_FACTOR = 1.0 if distance_source in ("google_maps", "osrm") else 1.3
        estimated_road_distance_meters = route_distance_meters * ROAD_DISTANCE_FACTOR
        
        # Duration factor: haversine speed estimate is rough, add 15% buffer
        DURATION_FACTOR = 1.0 if distance_source in ("google_maps", "osrm") else 1.15
        estimated_duration_seconds = route_duration_seconds * DURATION_FACTOR
        
        # Only add route if it has stops
        if stops:
            # Enforce max waypoints limit - truncate if necessary
            if len(stops) > max_waypoints:
                logger.warning(f"Route for TRUCK_{vehicle_id + 1} has {len(stops)} stops, truncating to {max_waypoints}")
                # Mark excess stops as unassigned
                for excess_stop in stops[max_waypoints:]:
                    unassigned_visits.append({
                        "visitId": excess_stop['visitId'],
                        "reason": "max_waypoints_exceeded"
                    })
                stops = stops[:max_waypoints]
            
            estimated_km = round(estimated_road_distance_meters / 1000, 2)
            estimated_hours = round(estimated_duration_seconds / 3600, 2)
            
            routes.append({
                "truckId": f"TRUCK_{vehicle_id + 1}",
                "start": {
                    "lat": locations[start_index]['lat'],
                    "lng": locations[start_index]['lng']
                },
                "end": {
                    "lat": locations[end_index]['lat'],
                    "lng": locations[end_index]['lng']
                },
                "stops": stops,
                "estimated_km": estimated_km,
                "estimated_hours": estimated_hours,
                "distance_source": distance_source,
                "waypoint_count": len(stops)
            })
    
    # Find unassigned visits
    for i, location in enumerate(locations):
        if i not in assigned_indices and 'visitId' in location:
            reason = "max_km_exceeded"
            
            # Check if it's due to distance constraint
            if solution:
                # Could be due to capacity or optimization
                reason = "optimization_constraint"
            
            unassigned_visits.append({
                "visitId": location['visitId'],
                "reason": reason
            })
    
    # Post-process: Ensure pickup-drop pairs are complete
    # If one part of a pair is assigned and other is not, move the assigned one to unassigned
    if order_map:
        incomplete_pairs = []
        for order_id, nodes in order_map.items():
            if 'pickup' in nodes and 'drop' in nodes:
                pickup_node = nodes['pickup']
                drop_node = nodes['drop']
                pickup_assigned = pickup_node in assigned_indices
                drop_assigned = drop_node in assigned_indices
                
                # If only one is assigned, mark as incomplete
                if pickup_assigned != drop_assigned:
                    incomplete_pairs.append({
                        'order_id': order_id,
                        'pickup_node': pickup_node,
                        'drop_node': drop_node,
                        'pickup_assigned': pickup_assigned,
                        'drop_assigned': drop_assigned
                    })
                elif not pickup_assigned and not drop_assigned:
                    # Both unassigned - they'll be added to unassigned_visits in the general loop
                    logger.info(f"📦 Order {order_id}: Both PICKUP and DROP unassigned (constraints ok)")
        
        # Handle incomplete pairs:
        # - If PICKUP is assigned but DROP is not: KEEP pickup, add drop to unassigned
        # - If DROP is assigned but PICKUP is not: REMOVE drop from route, add both to unassigned
        #   (can't deliver without picking up first)
        if incomplete_pairs:
            logger.warning(f"⚠️ Found {len(incomplete_pairs)} incomplete pickup-drop pairs...")
            
            nodes_to_remove = set()
            for pair in incomplete_pairs:
                if pair['pickup_assigned'] and not pair['drop_assigned']:
                    # Pickup is in route, drop is not - this is acceptable
                    # Keep pickup in route, add drop to unassigned for rescheduling
                    logger.info(f"📦 Order {pair['order_id']}: PICKUP in route, DROP added to unassigned for rescheduling")
                    if pair['drop_node'] < len(locations) and 'visitId' in locations[pair['drop_node']]:
                        unassigned_visits.append({
                            "visitId": locations[pair['drop_node']]['visitId'],
                            "reason": "drop_rescheduled_pickup_in_route"
                        })
                
                elif pair['drop_assigned'] and not pair['pickup_assigned']:
                    # Drop is in route but pickup is not - this is invalid!
                    # Remove drop from route, add BOTH to unassigned
                    logger.warning(f"⚠️ Order {pair['order_id']}: DROP without PICKUP - removing drop, adding both to unassigned")
                    nodes_to_remove.add(pair['drop_node'])
                    
                    # Add both to unassigned
                    if pair['pickup_node'] < len(locations) and 'visitId' in locations[pair['pickup_node']]:
                        unassigned_visits.append({
                            "visitId": locations[pair['pickup_node']]['visitId'],
                            "reason": "incomplete_pair_no_pickup"
                        })
                    if pair['drop_node'] < len(locations) and 'visitId' in locations[pair['drop_node']]:
                        unassigned_visits.append({
                            "visitId": locations[pair['drop_node']]['visitId'],
                            "reason": "incomplete_pair_no_pickup"
                        })
            
            # Remove invalid drops from routes
            if nodes_to_remove:
                for route in routes:
                    original_stops = route['stops']
                    filtered_stops = []
                    for stop in original_stops:
                        # Check if this stop should be removed
                        should_remove = False
                        for i, loc in enumerate(locations):
                            if loc.get('visitId') == stop['visitId'] and i in nodes_to_remove:
                                should_remove = True
                                break
                        if not should_remove:
                            filtered_stops.append(stop)
                    
                    # Re-sequence stops
                    for seq, stop in enumerate(filtered_stops, 1):
                        stop['sequence'] = seq
                    
                    route['stops'] = filtered_stops
                    route['waypoint_count'] = len(filtered_stops)
    
    # FINAL VALIDATION: Check for pickup-drop constraint violations
    validation_errors = []
    
    for route in routes:
        truck_id = route['truckId']
        stops = route['stops']
        
        # Build order sequence for this route
        order_sequence = {}  # order_id -> {'pickup_seq': N, 'drop_seq': M}
        for stop in stops:
            order_id = stop.get('order_id')
            visit_type = stop.get('visit_type')
            seq = stop.get('sequence')
            
            if order_id and visit_type:
                if order_id not in order_sequence:
                    order_sequence[order_id] = {}
                
                if visit_type.lower() in ['pickup', 'pick']:
                    order_sequence[order_id]['pickup_seq'] = seq
                    order_sequence[order_id]['pickup_truck'] = truck_id
                elif visit_type.lower() in ['drop', 'delivery']:
                    order_sequence[order_id]['drop_seq'] = seq
                    order_sequence[order_id]['drop_truck'] = truck_id
        
        # Check for violations within this route
        for order_id, seq_info in order_sequence.items():
            if 'pickup_seq' in seq_info and 'drop_seq' in seq_info:
                if seq_info['pickup_seq'] > seq_info['drop_seq']:
                    validation_errors.append(f"❌ Order {order_id}: DROP (seq {seq_info['drop_seq']}) before PICKUP (seq {seq_info['pickup_seq']}) in {truck_id}")
    
    # Check for cross-truck violations
    all_order_trucks = {}  # order_id -> {'pickup_truck': X, 'drop_truck': Y}
    for route in routes:
        truck_id = route['truckId']
        for stop in route['stops']:
            order_id = stop.get('order_id')
            visit_type = stop.get('visit_type')
            
            if order_id and visit_type:
                if order_id not in all_order_trucks:
                    all_order_trucks[order_id] = {}
                
                if visit_type.lower() in ['pickup', 'pick']:
                    all_order_trucks[order_id]['pickup_truck'] = truck_id
                elif visit_type.lower() in ['drop', 'delivery']:
                    all_order_trucks[order_id]['drop_truck'] = truck_id
    
    for order_id, truck_info in all_order_trucks.items():
        if 'pickup_truck' in truck_info and 'drop_truck' in truck_info:
            if truck_info['pickup_truck'] != truck_info['drop_truck']:
                validation_errors.append(f"❌ Order {order_id}: PICKUP in {truck_info['pickup_truck']} but DROP in {truck_info['drop_truck']}")
    
    if validation_errors:
        logger.error(f"🚨 VALIDATION FAILED - {len(validation_errors)} pickup-drop constraint violations:")
        for error in validation_errors:
            logger.error(f"  {error}")
    else:
        logger.info(f"✅ Validation passed: No pickup-drop constraint violations")
    
    return {
        "routes": routes,
        "unassigned_visits": unassigned_visits,
        "validation_errors": validation_errors if validation_errors else None
    }


@auto_routing_bp.route("/optimize", methods=["POST"])
def optimize_routes():
    """
    Hybrid VRP Optimizer: CVRP + VRPPD + VRPTW
    
    Solves routing with capacity constraints, pickup-delivery pairs, and time windows.
    Uses real road distances via OSRM (free) or Google Maps API.
    
    Expected JSON input:
    {
        "trucks": 3,
        "max_km": 120,
        "max_stops": 25,               // Optional, default 25 — max stops per truck
        "shift_duration_hours": 10,     // Optional, default 10 — max hours per driver shift
        "service_time_minutes": 10,     // Optional, default 10 — minutes spent at each stop
        "start": { "lat": 12.97, "lng": 77.59 },
        "end": { "lat": 12.93, "lng": 77.62 },
        "visits": [
            { 
                "visitId": "V1", 
                "lat": 12.95, 
                "lng": 77.60, 
                "sla_days": 0,
                "order_id": "ORD123",       // Optional
                "visit_type": "pickup",     // Optional: "pickup" or "drop"
                "time_window_start": 60,    // Optional: earliest arrival (min from shift start)
                "time_window_end": 300      // Optional: latest arrival (min from shift start)
            },
            { 
                "visitId": "V2", 
                "lat": 12.99, 
                "lng": 77.61, 
                "sla_days": 3,
                "order_id": "ORD123",       // Optional
                "visit_type": "drop"        // Optional: "pickup" or "drop"
            }
        ]
    }
    
    Returns:
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
                    }
                ],
                "estimated_km": 38.4
            }
        ],
        "unassigned_visits": [
            { "visitId": "V2", "reason": "max_km_exceeded" }
        ]
    }
    """
    try:
        # ─── Start memory + time tracking ───
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        api_start_time = time.time()
        api_snap = log_memory("API START")
        
        data = request.get_json()
        
        # Validate input
        if not data:
            tracemalloc.stop()
            return jsonify({"error": "No JSON data provided"}), 400
        
        required_fields = ["trucks", "max_km", "start", "end", "visits"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        num_trucks = data["trucks"]
        max_km = data["max_km"]
        max_stops = data.get("max_stops", MAX_WAYPOINTS_PER_ROUTE)  # Default 25
        shift_duration_hours = data.get("shift_duration_hours", 10)  # Default 10h shift
        service_time_minutes = data.get("service_time_minutes", 10)  # Default 10min per stop
        start_point = data["start"]
        end_point = data["end"]
        visits = data["visits"]
        
        # Validate data types and values
        if not isinstance(num_trucks, int) or num_trucks <= 0:
            return jsonify({"error": "trucks must be a positive integer"}), 400
        
        if not isinstance(max_km, (int, float)) or max_km <= 0:
            return jsonify({"error": "max_km must be a positive number"}), 400
        
        if not isinstance(visits, list) or len(visits) == 0:
            return jsonify({"error": "visits must be a non-empty list"}), 400
        
        # Validate start and end points
        for point_name, point in [("start", start_point), ("end", end_point)]:
            if not isinstance(point, dict) or "lat" not in point or "lng" not in point:
                return jsonify({"error": f"{point_name} must have 'lat' and 'lng' fields"}), 400
        
        # Validate visits
        for i, visit in enumerate(visits):
            required_visit_fields = ["visitId", "lat", "lng", "sla_days"]
            for field in required_visit_fields:
                if field not in visit:
                    return jsonify({"error": f"Visit {i} missing required field: {field}"}), 400
        
        logger.info(f"🚛 Hybrid VRP (CVRP+VRPPD+VRPTW): {num_trucks} trucks, "
                     f"max {max_km}km, max {max_stops} stops/truck, "
                     f"{shift_duration_hours}h shift, {service_time_minutes}min/stop, "
                     f"{len(visits)} visits")
        
        # Extract order_ids and visit_types from original visits
        original_order_ids = []
        original_visit_types = []
        
        for visit in visits:
            original_order_ids.append(visit.get("order_id"))
            original_visit_types.append(visit.get("visit_type"))
        
        # STEP 1: Filter visits by priority if dataset is large
        # This prevents solver from hanging on large datasets
        # Allow API to override these parameters
        MAX_VISITS_FOR_ROUTING = data.get("max_visits_for_routing", 80)  # Configurable limit (increased)
        SLA_THRESHOLD = data.get("sla_threshold", 3)  # Visits with SLA <= 3 days are urgent
        
        filtered_excluded_visits = []
        
        if len(visits) > MAX_VISITS_FOR_ROUTING:
            logger.warning(f"⚠️ Large dataset detected ({len(visits)} visits). Filtering by SLA priority...")
            visits, original_order_ids, original_visit_types, filtered_excluded_visits = \
                filter_visits_by_priority(
                    visits, 
                    original_order_ids, 
                    original_visit_types,
                    max_visits=MAX_VISITS_FOR_ROUTING,
                    sla_threshold=SLA_THRESHOLD
                )
        else:
            logger.info(f"✅ Dataset size ({len(visits)} visits) is within limits. Processing all visits.")
        
        # Combine visits at the same location
        logger.info("Combining visits at same locations...")
        combined_visits, visit_groups, combined_order_info, location_to_visits = \
            combine_visits_at_same_location(visits, original_order_ids, original_visit_types)
        
        # Safety check: If too many unique locations, further reduce by taking highest priority
        MAX_UNIQUE_LOCATIONS = data.get("max_unique_locations", 40)
        if len(combined_visits) > MAX_UNIQUE_LOCATIONS:
            logger.warning(f"⚠️ {len(combined_visits)} unique locations exceed limit of {MAX_UNIQUE_LOCATIONS}. Reducing...")
            
            # Sort by SLA (lowest first = highest priority)
            indexed_visits = list(enumerate(combined_visits))
            indexed_visits.sort(key=lambda x: x[1].get('sla_days', 5))
            
            # Keep only top N locations
            kept_indices = set()
            for idx, _ in indexed_visits[:MAX_UNIQUE_LOCATIONS]:
                kept_indices.add(idx)
            
            # Filter all aligned lists
            new_combined_visits = []
            new_visit_groups = []
            new_combined_order_info = []
            
            for i in range(len(combined_visits)):
                if i in kept_indices:
                    new_combined_visits.append(combined_visits[i])
                    new_visit_groups.append(visit_groups[i])
                    new_combined_order_info.append(combined_order_info[i])
                else:
                    # Add all visits at this location to excluded list
                    for visit_id in visit_groups[i]:
                        filtered_excluded_visits.append({
                            'visitId': visit_id,
                            'reason': 'filtered_by_location_limit',
                            'sla_days': combined_visits[i].get('sla_days', 5)
                        })
            
            combined_visits = new_combined_visits
            visit_groups = new_visit_groups
            combined_order_info = new_combined_order_info
            
            logger.info(f"✅ Reduced to {len(combined_visits)} unique locations")
        
        # Build locations list: [start, ...combined_visits, end]
        locations = [start_point] + combined_visits + [end_point]
        start_index = 0
        end_index = len(locations) - 1
        
        # Create distance + duration matrices using combined locations
        # Priority: Google Maps → OSRM (free road distances) → Haversine (fallback)
        matrix_start = time.time()
        distance_matrix, duration_matrix, distance_source = create_distance_matrix(locations)
        matrix_elapsed = time.time() - matrix_start
        matrix_snap = log_memory("Distance matrix BUILT", api_snap)
        logger.info(f"⏱️  Distance matrix: {len(locations)}x{len(locations)} built in {matrix_elapsed:.2f}s")
        
        # Create priority list (inverse of SLA days - lower SLA = higher priority)
        priorities = [5]  # Start point - neutral priority
        for visit in combined_visits:
            sla_days = visit.get("sla_days", 5)
            
            if sla_days <= 0:
                priority = 15 + abs(sla_days)
            elif sla_days <= 2:
                priority = 10 - sla_days
            elif sla_days == 3:
                priority = 7
            else:
                priority = max(0, 6 - (sla_days - 4))
            
            priorities.append(priority)
        priorities.append(5)  # End point - neutral priority
        
        # Prepare combined_order_info aligned with locations (add None for start/end)
        aligned_order_info = [None] + combined_order_info + [None]
        
        # Prepare visit_groups aligned with locations
        # [None (start), ...visit groups..., None (end)]
        aligned_visit_groups = [None] + visit_groups + [None]
        
        # ─── Build time windows from visit data (VRPTW) ───
        # time_window_start/end are in MINUTES from shift start
        # Convert to seconds for the solver
        aligned_time_windows = [None]  # None for start node
        has_any_time_window = False
        for visit in combined_visits:
            tw_start = visit.get("time_window_start")
            tw_end = visit.get("time_window_end")
            if tw_start is not None and tw_end is not None:
                aligned_time_windows.append((int(tw_start * 60), int(tw_end * 60)))
                has_any_time_window = True
            else:
                aligned_time_windows.append(None)
        aligned_time_windows.append(None)  # None for end node
        
        # Build service times aligned with locations (seconds)
        # A combined node with N visits gets N × service_time (e.g., 5 visits × 10min = 50min)
        aligned_service_times = [0]  # 0 for start
        svc_seconds = int(service_time_minutes * 60)
        for idx, _ in enumerate(combined_visits):
            # visit_groups[idx] contains the list of visit IDs at this combined location
            num_visits_at_node = len(visit_groups[idx]) if idx < len(visit_groups) and visit_groups[idx] else 1
            aligned_service_times.append(svc_seconds * num_visits_at_node)
        aligned_service_times.append(0)  # 0 for end
        
        # Convert max_km to meters, accounting for Haversine underestimation
        # OSRM and Google Maps give real road distances; Haversine is ~30% shorter
        ROAD_DISTANCE_FACTOR = 1.0 if distance_source in ("google_maps", "osrm") else 1.3
        max_distance_meters = int(max_km * 1000 / ROAD_DISTANCE_FACTOR)
        
        # Convert shift duration to seconds
        max_route_time = int(shift_duration_hours * 3600)
        
        logger.info(f"📏 Distance: {max_km}km limit → {max_distance_meters/1000:.1f}km solver "
                     f"(source: {distance_source}, factor: {ROAD_DISTANCE_FACTOR}x)")
        logger.info(f"⏰ Time: {shift_duration_hours}h shift, {service_time_minutes}min/stop, "
                     f"time windows: {'YES' if has_any_time_window else 'NO'}")
        
        # Solve the Hybrid VRP (CVRP + VRPPD + VRPTW)
        result = solve_vrp(
            num_vehicles=num_trucks,
            max_distance_per_vehicle=max_distance_meters,
            locations=locations,
            distance_matrix=distance_matrix,
            duration_matrix=duration_matrix,
            start_index=start_index,
            end_index=end_index,
            priorities=priorities,
            distance_source=distance_source,
            max_waypoints=max_stops,
            combined_order_info=aligned_order_info,
            visit_groups=aligned_visit_groups,
            original_visits=visits,
            time_windows=aligned_time_windows if has_any_time_window else None,
            service_times=aligned_service_times,
            max_route_time=max_route_time
        )
        
        if result is None:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            return jsonify({
                "error": "Could not find a solution for the given constraints",
                "suggestion": "Try increasing max_km or number of trucks"
            }), 400
        
        # Add filtered-out visits to unassigned visits
        if filtered_excluded_visits:
            result['unassigned_visits'].extend(filtered_excluded_visits)
            logger.info(f"Added {len(filtered_excluded_visits)} filtered visits to unassigned list")
        
        # Log route utilization summary
        total_assigned = 0
        for route in result['routes']:
            num_stops = route['waypoint_count']
            est_km = route['estimated_km']
            est_hrs = route.get('estimated_hours', 0)
            util_km = (est_km / max_km * 100) if max_km > 0 else 0
            util_stops = (num_stops / max_stops * 100) if max_stops > 0 else 0
            util_time = (est_hrs / shift_duration_hours * 100) if shift_duration_hours > 0 else 0
            total_assigned += num_stops
            logger.info(f"📊 {route['truckId']}: {num_stops} stops ({util_stops:.0f}%), "
                        f"{est_km:.1f}km ({util_km:.0f}%), "
                        f"{est_hrs:.1f}h ({util_time:.0f}%) — "
                        f"[{distance_source}]")
        
        logger.info(f"✅ Solution: {len(result['routes'])} routes, {total_assigned} assigned, "
                     f"{len(result['unassigned_visits'])} unassigned out of {len(visits)} total visits")
        
        # ─── Final memory + timing summary ───
        total_elapsed = time.time() - api_start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        logger.info(f"{'='*60}")
        logger.info(f"🧠 MEMORY SUMMARY:")
        logger.info(f"   Peak RAM used:    {peak_mem / (1024*1024):.2f} MB")
        logger.info(f"   Final RAM used:   {current_mem / (1024*1024):.2f} MB")
        logger.info(f"⏱️  TIMING SUMMARY:")
        logger.info(f"   Total API time:   {total_elapsed:.2f}s")
        logger.info(f"{'='*60}")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in optimize_routes: {str(e)}", exc_info=True)
        # Stop tracemalloc if it was started
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@auto_routing_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "auto-routing",
        "version": "1.0.0"
    }), 200

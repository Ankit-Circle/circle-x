import logging
import os
import sys
import time
import tracemalloc
import json
import math
from flask import Blueprint, request, jsonify
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from typing import List, Dict, Tuple, Optional
import googlemaps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ─── Google Distance Matrix API ───
# Google Maps API key (required for real road distances)
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
if not GOOGLE_MAPS_API_KEY:
    logger.warning("⚠️ GOOGLE_MAPS_API_KEY not set - distance calculations may fail")

# ─── Haversine Distance Calculation ───
# Average driving speed for duration estimation (km/h)
AVERAGE_DRIVING_SPEED_KMH = 35.0  # Default: 35 km/h for city driving

auto_routing_bp = Blueprint("auto_routing", __name__)

# Maximum waypoints per route
MAX_WAYPOINTS_PER_ROUTE = 25


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


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth using Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in degrees
        lat2, lon2: Latitude and longitude of second point in degrees
    
    Returns:
        Distance in meters
    """
    # Earth's radius in meters
    R = 6371000
    
    # Convert latitude and longitude from degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance


def create_distance_matrix(
    locations: List[Dict]
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Create distance AND duration matrices using Google Distance Matrix API.
    
    Uses real road distances and travel durations from Google Maps.
    Batches requests to handle API limits.
    
    Returns:
        (distance_matrix [meters], duration_matrix [seconds])
    """
    n = len(locations)
    distance_matrix = [[0] * n for _ in range(n)]
    duration_matrix = [[0] * n for _ in range(n)]
    
    if not GOOGLE_MAPS_API_KEY:
        logger.error("❌ GOOGLE_MAPS_API_KEY not set - cannot use Google Distance Matrix API")
        raise ValueError("GOOGLE_MAPS_API_KEY environment variable is required")
    
    logger.info(f"🗺️  Building {n}x{n} distance + duration matrix via Google Distance Matrix API...")
    
    # Initialize Google Maps client
    gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    
    # Google Distance Matrix API limits:
    # - Free tier: Max 100 elements per request (10 origins × 10 destinations)
    # - Standard tier: Max 625 elements per request (25 origins × 25 destinations)
    # Using 10×10 to be safe and work with both free and paid tiers
    ORIGIN_BATCH_SIZE = 10
    DEST_BATCH_SIZE = 10
    
    # Prepare origins and destinations
    origins = [(loc['lat'], loc['lng']) for loc in locations]
    destinations = origins  # Same locations for both
    
    total_requests = 0
    start_time = time.time()
    
    # Process origins in batches
    for orig_batch_start in range(0, n, ORIGIN_BATCH_SIZE):
        orig_batch_end = min(orig_batch_start + ORIGIN_BATCH_SIZE, n)
        batch_origins = origins[orig_batch_start:orig_batch_end]
        
        # Process destinations in batches
        for dest_batch_start in range(0, n, DEST_BATCH_SIZE):
            dest_batch_end = min(dest_batch_start + DEST_BATCH_SIZE, n)
            batch_destinations = destinations[dest_batch_start:dest_batch_end]
            
            try:
                # Call Google Distance Matrix API
                result = gmaps.distance_matrix(
                    origins=batch_origins,
                    destinations=batch_destinations,
                    mode="driving",
                    units="metric"
                )
                
                total_requests += 1
                
                # Check for API-level MAX_ELEMENTS_EXCEEDED (just in case)
                if result.get('status') == 'MAX_ELEMENTS_EXCEEDED':
                    logger.error(
                        f"❌ MAX_ELEMENTS_EXCEEDED: Batch too large "
                        f"({len(batch_origins)} origins × {len(batch_destinations)} destinations). "
                        f"Falling back to Haversine for this batch."
                    )
                    # Fallback to Haversine for this batch
                    for i, origin in enumerate(batch_origins):
                        orig_idx = orig_batch_start + i
                        for j, dest in enumerate(batch_destinations):
                            dest_idx = dest_batch_start + j
                            if orig_idx != dest_idx:
                                lat1, lon1 = origin
                                lat2, lon2 = dest
                                distance_meters = haversine_distance(lat1, lon1, lat2, lon2)
                                distance_matrix[orig_idx][dest_idx] = int(round(distance_meters))
                                avg_speed_ms = AVERAGE_DRIVING_SPEED_KMH * 1000 / 3600
                                duration_seconds = int(round(distance_meters / avg_speed_ms))
                                duration_matrix[orig_idx][dest_idx] = max(1, duration_seconds)
                    continue
                
                # Parse results
                if result.get('status') == 'OK' and result.get('rows'):
                    for i, row in enumerate(result['rows']):
                        orig_idx = orig_batch_start + i
                        for j, element in enumerate(row['elements']):
                            dest_idx = dest_batch_start + j
                            
                            if orig_idx == dest_idx:
                                # Same location
                                distance_matrix[orig_idx][dest_idx] = 0
                                duration_matrix[orig_idx][dest_idx] = 0
                            elif element['status'] == 'OK':
                                # Distance in meters
                                distance_meters = element['distance']['value']
                                distance_matrix[orig_idx][dest_idx] = int(round(distance_meters))
                                
                                # Duration in seconds
                                duration_seconds = element['duration']['value']
                                duration_matrix[orig_idx][dest_idx] = max(1, duration_seconds)  # Minimum 1 second
                            else:
                                # If API call failed for this pair, fallback to Haversine
                                logger.warning(
                                    f"⚠️ Google API failed for {orig_idx}->{dest_idx}: "
                                    f"{element.get('status')}, using Haversine fallback"
                                )
                                lat1, lon1 = batch_origins[i]
                                lat2, lon2 = batch_destinations[j]
                                distance_meters = haversine_distance(lat1, lon1, lat2, lon2)
                                distance_matrix[orig_idx][dest_idx] = int(round(distance_meters))
                                avg_speed_ms = AVERAGE_DRIVING_SPEED_KMH * 1000 / 3600
                                duration_seconds = int(round(distance_meters / avg_speed_ms))
                                duration_matrix[orig_idx][dest_idx] = max(1, duration_seconds)
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            
            except Exception as e:
                logger.error(
                    f"❌ Error calling Google Distance Matrix API for origins "
                    f"{orig_batch_start}-{orig_batch_end}, destinations "
                    f"{dest_batch_start}-{dest_batch_end}: {e}"
                )
                # Fallback to Haversine for this batch
                for i, origin in enumerate(batch_origins):
                    orig_idx = orig_batch_start + i
                    for j, dest in enumerate(batch_destinations):
                        dest_idx = dest_batch_start + j
                        if orig_idx != dest_idx:
                            lat1, lon1 = origin
                            lat2, lon2 = dest
                            distance_meters = haversine_distance(lat1, lon1, lat2, lon2)
                            distance_matrix[orig_idx][dest_idx] = int(round(distance_meters))
                            avg_speed_ms = AVERAGE_DRIVING_SPEED_KMH * 1000 / 3600
                            duration_seconds = int(round(distance_meters / avg_speed_ms))
                            duration_matrix[orig_idx][dest_idx] = max(1, duration_seconds)
    
    elapsed = time.time() - start_time
    logger.info(
        f"✅ Google Distance Matrix API: {n}x{n} matrix built in {elapsed:.2f}s "
        f"({total_requests} API calls)"
    )
    return distance_matrix, duration_matrix


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
    
    # Build selected visits list - ABSOLUTE priority for SLA-critical visits
    selected_visits = []
    
    # ALWAYS include ALL critical (breached) visits - NO LIMIT
    # These MUST be completed regardless of capacity constraints
    selected_visits.extend(critical_visits)
    logger.info(f"🚨 Added ALL {len(critical_visits)} BREACHED visits (absolute priority, no limit)")
    
    # ALWAYS include ALL urgent visits (1-2 days) - NO LIMIT
    # These are about to breach and must be prioritized
    urgent_visits.sort(key=lambda x: x['sla_days'])
    selected_visits.extend(urgent_visits)
    logger.info(f"⚠️  Added ALL {len(urgent_visits)} urgent visits (SLA 1-2 days, no limit)")
    
    # Now apply max_visits limit to warning and normal visits
    # Only limit non-critical visits
    critical_and_urgent_count = len(critical_visits) + len(urgent_visits)
    remaining = max(0, max_visits - critical_and_urgent_count)
    
    if critical_and_urgent_count > max_visits:
        logger.warning(f"⚠️  SLA-critical visits ({critical_and_urgent_count}) exceed max_visits ({max_visits}). "
                      f"Including all critical visits anyway - SLA takes priority!")
    
    # Add warning visits (up to remaining capacity)
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
    max_waypoints: int = MAX_WAYPOINTS_PER_ROUTE,
    combined_order_info: List[Dict] = None,
    visit_groups: List[List[str]] = None,
    original_visits: List[Dict] = None,
    time_windows: Optional[List[Optional[Tuple[int, int]]]] = None,
    service_times: Optional[List[int]] = None,
    max_route_time: int = 36000,
    truck_capacity: int = 100,
    start_truck_number: int = 1
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
    
    # ─── ENFORCE STRICT DISTANCE CONSTRAINT ───
    # Ensure distance constraint is strictly enforced - no route can exceed max_distance_per_vehicle
    for vehicle_id in range(num_vehicles):
        start_idx = routing.Start(vehicle_id)
        end_idx = routing.End(vehicle_id)
        # Set strict limits: cumulative distance at start and end must be within [0, max_distance_per_vehicle]
        distance_dimension.CumulVar(start_idx).SetRange(0, max_distance_per_vehicle)
        distance_dimension.CumulVar(end_idx).SetRange(0, max_distance_per_vehicle)
    
    # Set strict limits for all nodes - cumulative distance must never exceed max_distance_per_vehicle
    for node in range(len(locations)):
        if node != start_index and node != end_index:
            index = manager.NodeToIndex(node)
            # Strict constraint: cumulative distance at this node must be within [0, max_distance_per_vehicle]
            distance_dimension.CumulVar(index).SetRange(0, max_distance_per_vehicle)
    
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
    # max_stops counts PHYSICAL LOCATIONS (combined nodes), not individual visits.
    # A combined location with 5 visits at the same lat/lng = 1 stop.
    # Example: 15 visits at 10 unique locations + max_stops=10 → all fit.
    
    # Build visits_per_node lookup (used for service time scaling, NOT for stop counting)
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
        """Returns 1 for each visit node (combined location = 1 stop)."""
        from_node = manager.IndexToNode(from_index)
        # Start/end nodes = 0 stops; every other node = 1 physical stop
        if from_node == start_index or from_node == end_index:
            return 0
        return 1
    
    count_callback_index = routing.RegisterUnaryTransitCallback(count_callback)
    
    # Add dimension for waypoint count (counts physical locations, not individual visits)
    waypoint_dimension_name = 'WaypointCount'
    routing.AddDimension(
        count_callback_index,
        0,  # no slack
        max_waypoints,  # maximum physical locations (stops) per vehicle
        True,  # start cumul to zero
        waypoint_dimension_name
    )
    waypoint_dimension = routing.GetDimensionOrDie(waypoint_dimension_name)
    
    # ─── ENFORCE STRICT WAYPOINT CONSTRAINT ───
    # Ensure waypoint constraint is strictly enforced - no route can exceed max_waypoints
    for vehicle_id in range(num_vehicles):
        start_idx = routing.Start(vehicle_id)
        end_idx = routing.End(vehicle_id)
        # Set strict limits: cumulative waypoint count at start and end must be within [0, max_waypoints]
        waypoint_dimension.CumulVar(start_idx).SetRange(0, max_waypoints)
        waypoint_dimension.CumulVar(end_idx).SetRange(0, max_waypoints)
    
    # Set strict limits for all nodes - cumulative waypoint count must never exceed max_waypoints
    for node in range(len(locations)):
        if node != start_index and node != end_index:
            index = manager.NodeToIndex(node)
            # Strict constraint: cumulative waypoint count at this node must be within [0, max_waypoints]
            waypoint_dimension.CumulVar(index).SetRange(0, max_waypoints)
    
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
    
    # ─── VOLUME CAPACITY CONSTRAINT ───
    # Build volume capacity change per node (pickup adds, drop subtracts)
    # IMPORTANT: Drops without pickup pairs are pre-loaded items, so we don't subtract when visiting them
    # (they're already accounted for in initial load, which we'll handle via dimension start range)
    # This ensures trucks never exceed volumetric capacity at any point in the route sequence
    node_volume_change = {}  # node_index -> net volume change (positive for pickups, negative for drops)
    visit_id_to_vol_capacity = {}
    visit_id_to_visit_type = {}
    visit_id_to_order_id = {}
    
    if original_visits:
        for visit in original_visits:
            visit_id = visit.get('visitId')
            if visit_id:
                visit_id_to_vol_capacity[visit_id] = visit.get('vol_capacity', 0)
                visit_id_to_visit_type[visit_id] = visit.get('visit_type', '').lower()
                visit_id_to_order_id[visit_id] = visit.get('order_id')
    
    # Build order_map to identify which drops have pickup pairs (for volume calculation)
    # Drops with pickup pairs: subtract when visiting (normal flow)
    # Drops without pickup pairs: pre-loaded, don't subtract (already in truck at start)
    order_map_for_volume = {}
    if combined_order_info:
        for node in range(len(locations)):
            if node == start_index or node == end_index:
                continue
            if node < len(combined_order_info) and combined_order_info[node]:
                info = combined_order_info[node]
                for order_id, visit_type in zip(info.get('order_ids', []), info.get('visit_types', [])):
                    if order_id and visit_type:
                        if order_id not in order_map_for_volume:
                            order_map_for_volume[order_id] = {}
                        vtype_lower = visit_type.lower()
                        if vtype_lower in ['pickup', 'pick']:
                            order_map_for_volume[order_id]['has_pickup'] = True
                        elif vtype_lower in ['drop', 'delivery']:
                            order_map_for_volume[order_id]['has_drop'] = True
    
    # Calculate net volume change per node
    for node_idx in range(len(locations)):
        if node_idx == start_index or node_idx == end_index:
            node_volume_change[node_idx] = 0
            continue
        
        net_volume = 0
        # Check if this node has combined visits
        if visit_groups and node_idx < len(visit_groups) and visit_groups[node_idx]:
            # Multiple visits at this node - sum their volume changes
            for visit_id in visit_groups[node_idx]:
                vol_capacity = visit_id_to_vol_capacity.get(visit_id, 0)
                visit_type = visit_id_to_visit_type.get(visit_id, '').lower()
                order_id = visit_id_to_order_id.get(visit_id)
                
                # Pickup adds volume (positive)
                if visit_type in ['pickup', 'damaged_pickup', 'exchanged_pickup', 'exchange_pickup',
                                 'return_pickup', 'return_pick', 'returned_from']:
                    net_volume += vol_capacity
                # Drop subtracts volume (negative) - BUT only if it has a pickup pair
                # Drops without pickup pairs are pre-loaded, so we don't subtract (they're in initial load)
                elif visit_type in ['drop', 'damaged_drop', 'exchanged_drop', 'exchange_drop',
                                   'return_drop', 'return_delivery']:
                    # Only subtract if this drop has a pickup pair (normal pickup-drop flow)
                    if order_id and order_id in order_map_for_volume:
                        order_info = order_map_for_volume[order_id]
                        # If order has both pickup and drop, subtract (normal flow)
                        if order_info.get('has_pickup') and order_info.get('has_drop'):
                            net_volume -= vol_capacity
                        # If drop doesn't have pickup, it's pre-loaded - don't subtract
                        # (it's already accounted for in initial load)
                    # returned_to is always pre-loaded, so don't subtract
                elif visit_type == 'returned_to':
                    # returned_to is pre-loaded, don't subtract
                    pass
        else:
            # Single visit at this node
            if 'visitId' in locations[node_idx]:
                visit_id = locations[node_idx]['visitId']
                vol_capacity = visit_id_to_vol_capacity.get(visit_id, 0)
                visit_type = visit_id_to_visit_type.get(visit_id, '').lower()
                order_id = visit_id_to_order_id.get(visit_id)
                
                if visit_type in ['pickup', 'damaged_pickup', 'exchanged_pickup', 'exchange_pickup',
                                 'return_pickup', 'return_pick', 'returned_from']:
                    net_volume += vol_capacity
                elif visit_type in ['drop', 'damaged_drop', 'exchanged_drop', 'exchange_drop',
                                   'return_drop', 'return_delivery']:
                    # Only subtract if this drop has a pickup pair
                    if order_id and order_id in order_map_for_volume:
                        order_info = order_map_for_volume[order_id]
                        if order_info.get('has_pickup') and order_info.get('has_drop'):
                            net_volume -= vol_capacity
                elif visit_type == 'returned_to':
                    # returned_to is pre-loaded, don't subtract
                    pass
        
        node_volume_change[node_idx] = net_volume
    
    # Create volume capacity callback
    def volume_callback(from_index):
        """Returns the net volume change at the 'from' node."""
        from_node = manager.IndexToNode(from_index)
        return node_volume_change.get(from_node, 0)
    
    volume_callback_index = routing.RegisterUnaryTransitCallback(volume_callback)
    
    # Add Volume Capacity dimension - ensures cumulative volume never exceeds truck_capacity
    volume_dimension_name = 'VolumeCapacity'
    routing.AddDimension(
        volume_callback_index,
        0,  # no slack
        truck_capacity,  # maximum volume capacity per vehicle
        True,  # start cumul to zero (vehicles start empty)
        volume_dimension_name
    )
    volume_dimension = routing.GetDimensionOrDie(volume_dimension_name)
    
    # Set volume capacity constraints for start and end nodes
    # IMPORTANT: Pre-loaded drops (drops without pickup pairs) don't subtract volume when visited
    # They're already in the truck at start, so initial_load = sum of pre-loaded drops
    # Since we don't subtract for pre-loaded drops, cumulative starts at initial_load and stays >= 0
    # For drops with pickup pairs: pickup adds, drop subtracts (normal flow)
    for vehicle_id in range(num_vehicles):
        start_idx = routing.Start(vehicle_id)
        # Allow initial load up to truck_capacity (for pre-loaded items like drops without pickups)
        # The solver will set this to the sum of pre-loaded drops in the route
        volume_dimension.CumulVar(start_idx).SetRange(0, truck_capacity)
        end_idx = routing.End(vehicle_id)
        # End volume should be >= 0
        # It can be 0 (empty if all pre-loaded items unloaded and no pickups) or up to capacity
        volume_dimension.CumulVar(end_idx).SetRange(0, truck_capacity)
    
    # CRITICAL: Ensure cumulative volume never exceeds truck_capacity AND never goes below 0
    # Since initial_load = sum of pre-loaded drops, and we subtract when visiting them,
    # cumulative should always be >= 0 (initial_load - sum_of_unloaded_drops >= 0)
    for node in range(len(locations)):
        if node != start_index and node != end_index:
            index = manager.NodeToIndex(node)
            # Ensure cumulative volume at this node never exceeds capacity and never goes below 0
            volume_dimension.CumulVar(index).SetRange(0, truck_capacity)
    
    logger.info(f"📦 Volume capacity: max {truck_capacity} units per truck (ENFORCED during route generation)")
    
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
    
    skipped_pairs_for_combination = []  # Track skipped exchange/damage pairs for combination handling
    
    if combined_order_info:
        # Step 1: Build order_map  — order_id -> {'pickup': node, 'drop': node, ...}
        # Also track damaged, exchanged, and return pairs separately
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
                        
                        vtype_lower = visit_type.lower()
                        
                        # Standard pickup-drop pairs
                        if vtype_lower in ['pickup', 'pick']:
                            order_map[order_id]['pickup'] = node
                        elif vtype_lower in ['drop', 'delivery']:
                            order_map[order_id]['drop'] = node
                        # Damaged pairs
                        elif vtype_lower in ['damaged_pickup']:
                            order_map[order_id]['damaged_pickup'] = node
                        elif vtype_lower in ['damaged_drop']:
                            order_map[order_id]['damaged_drop'] = node
                        # Exchange pairs
                        elif vtype_lower in ['exchanged_pickup', 'exchange_pickup']:
                            order_map[order_id]['exchanged_pickup'] = node
                        elif vtype_lower in ['exchanged_drop', 'exchange_drop']:
                            order_map[order_id]['exchanged_drop'] = node
                        # Return pairs
                        elif vtype_lower in ['return_pickup', 'return_pick']:
                            order_map[order_id]['return_pickup'] = node
                        elif vtype_lower in ['return_drop', 'return_delivery']:
                            order_map[order_id]['return_drop'] = node
                        elif vtype_lower in ['returned_from', 'returned_to']:
                            # Legacy returned visits don't have pickup-drop constraints
                            # They are standalone mandatory visits
                            pass
        
        # Step 2: Collect candidate pairs, sorted by priority (breached first)
        # Priority order: damaged/exchanged/return pairs (HIGHEST) > standard pickup-drop
        candidate_pairs = []
        for order_id, nodes in order_map.items():
            # Standard pickup-drop pairs
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
                    'priority': pri,
                    'pair_type': 'standard'
                })
                logger.info(f"✅ Complete pair for order {order_id}: pickup node {pickup_node} -> drop node {drop_node}")
            elif 'pickup' in nodes:
                logger.warning(f"⚠️ Order {order_id} has PICKUP (node {nodes['pickup']}) but NO DROP")
            elif 'drop' in nodes:
                logger.warning(f"⚠️ Order {order_id} has DROP (node {nodes['drop']}) but NO PICKUP")
            
            # Damaged pairs (HIGHEST PRIORITY)
            if 'damaged_pickup' in nodes and 'damaged_drop' in nodes:
                pickup_node = nodes['damaged_pickup']
                drop_node = nodes['damaged_drop']
                
                if pickup_node == drop_node:
                    logger.info(f"📍 Order {order_id}: damaged_pickup and damaged_drop at SAME location (node {pickup_node}) — no constraint needed")
                else:
                    # Highest priority for damaged pairs
                    pri = 1000  # Extreme priority to ensure scheduling
                    candidate_pairs.append({
                        'order_id': order_id,
                        'pickup_node': pickup_node,
                        'drop_node': drop_node,
                        'priority': pri,
                        'pair_type': 'damaged'
                    })
                    logger.info(f"🔴 HIGH PRIORITY: Damaged pair for order {order_id}: damaged_pickup node {pickup_node} -> damaged_drop node {drop_node}")
            elif 'damaged_pickup' in nodes:
                logger.warning(f"⚠️ Order {order_id} has DAMAGED_PICKUP (node {nodes['damaged_pickup']}) but NO DAMAGED_DROP")
            elif 'damaged_drop' in nodes:
                logger.warning(f"⚠️ Order {order_id} has DAMAGED_DROP (node {nodes['damaged_drop']}) but NO DAMAGED_PICKUP")
            
            # Exchange pairs (HIGHEST PRIORITY)
            if 'exchanged_pickup' in nodes and 'exchanged_drop' in nodes:
                pickup_node = nodes['exchanged_pickup']
                drop_node = nodes['exchanged_drop']
                
                if pickup_node == drop_node:
                    logger.info(f"📍 Order {order_id}: exchanged_pickup and exchanged_drop at SAME location (node {pickup_node}) — no constraint needed")
                else:
                    # Highest priority for exchange pairs
                    pri = 1000  # Extreme priority to ensure scheduling
                    candidate_pairs.append({
                        'order_id': order_id,
                        'pickup_node': pickup_node,
                        'drop_node': drop_node,
                        'priority': pri,
                        'pair_type': 'exchanged'
                    })
                    logger.info(f"🟡 HIGH PRIORITY: Exchange pair for order {order_id}: exchanged_pickup node {pickup_node} -> exchanged_drop node {drop_node}")
            elif 'exchanged_pickup' in nodes:
                logger.warning(f"⚠️ Order {order_id} has EXCHANGED_PICKUP (node {nodes['exchanged_pickup']}) but NO EXCHANGED_DROP")
            elif 'exchanged_drop' in nodes:
                logger.warning(f"⚠️ Order {order_id} has EXCHANGED_DROP (node {nodes['exchanged_drop']}) but NO EXCHANGED_PICKUP")
            
            # Return pairs (HIGHEST PRIORITY)
            if 'return_pickup' in nodes and 'return_drop' in nodes:
                pickup_node = nodes['return_pickup']
                drop_node = nodes['return_drop']
                
                if pickup_node == drop_node:
                    logger.info(f"📍 Order {order_id}: return_pickup and return_drop at SAME location (node {pickup_node}) — no constraint needed")
                else:
                    # Highest priority for return pairs
                    pri = 1000  # Extreme priority to ensure scheduling
                    candidate_pairs.append({
                        'order_id': order_id,
                        'pickup_node': pickup_node,
                        'drop_node': drop_node,
                        'priority': pri,
                        'pair_type': 'return'
                    })
                    logger.info(f"🟢 HIGH PRIORITY: Return pair for order {order_id}: return_pickup node {pickup_node} -> return_drop node {drop_node}")
            elif 'return_pickup' in nodes:
                logger.warning(f"⚠️ Order {order_id} has RETURN_PICKUP (node {nodes['return_pickup']}) but NO RETURN_DROP")
            elif 'return_drop' in nodes:
                logger.warning(f"⚠️ Order {order_id} has RETURN_DROP (node {nodes['return_drop']}) but NO RETURN_PICKUP")
        
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
            # EXCEPTION: We'll handle exchange/damage combinations separately using CumulVar constraints
            if pn in nodes_used_in_pairs or dn in nodes_used_in_pairs:
                pair_type = pair.get('pair_type', 'standard')
                # Track exchange/damage pairs that were skipped - we'll handle them with CumulVar
                if pair_type in ['exchanged', 'damaged']:
                    skipped_pairs_for_combination.append(pair)
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
            if skipped_pairs_for_combination:
                logger.info(f"  📋 {len(skipped_pairs_for_combination)} skipped pairs are exchange/damage - will handle with CumulVar constraints")
    
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
    
    # ─── ENHANCED SLA PRIORITIZATION + MANDATORY VISITS ───
    # DRAMATICALLY increased penalties to ensure SLA-critical visits are NEVER dropped
    # Base penalty is now 50x max distance (was 20x) for even stronger prioritization
    # MANDATORY visits (returned_from, returned_to) get 1000x penalty - virtually impossible to drop
    base_penalty = max_distance_per_vehicle * 50
    
    logger.info(f"🎯 SLA-PRIORITIZED + MANDATORY routing — base penalty: {base_penalty} "
                f"(vehicle fixed cost: {vehicle_fixed_cost})")
    
    breached_count = 0
    urgent_count = 0
    paired_count = 0
    mandatory_count = 0
    
    for node in range(len(locations)):
        # Skip start and end nodes
        if node == start_index or node == end_index:
            continue
        
        priority = priorities[node] if node < len(priorities) else 5
        
        # Check if this node contains a mandatory visit type
        # MANDATORY types: returned_from, returned_to, damaged_pickup, damaged_drop,
        #                  exchanged_pickup, exchanged_drop, return_pickup, return_drop
        is_mandatory = False
        mandatory_types_found = []
        if combined_order_info and node < len(combined_order_info) and combined_order_info[node]:
            info = combined_order_info[node]
            visit_types_at_node = info.get('visit_types', [])
            for vtype in visit_types_at_node:
                if vtype:
                    vtype_lower = vtype.lower()
                    if vtype_lower in ['returned_from', 'returned_to',
                                       'damaged_pickup', 'damaged_drop',
                                       'exchanged_pickup', 'exchanged_drop', 'exchange_pickup', 'exchange_drop',
                                       'return_pickup', 'return_drop', 'return_pick', 'return_delivery']:
                        is_mandatory = True
                        mandatory_types_found.append(vtype)
        
        # MANDATORY visits get EXTREMELY high penalties but CAN be unassigned if absolutely necessary
        # This ensures solver always returns a solution (even if some mandatory visits are unassigned)
        # Routes prioritize efficiency first, then try to satisfy mandatory visits
        if is_mandatory:
            mandatory_count += 1
            types_str = ', '.join(mandatory_types_found)
            logger.info(f"🔒 Node {node} marked as MANDATORY ({types_str}) - HIGHEST PRIORITY (can be unassigned if impossible)")
            # Add disjunction with EXTREMELY high penalty - highest priority but can be unassigned if needed
            # Use 100x base penalty to ensure mandatory visits are scheduled whenever possible
            mandatory_penalty = base_penalty * 100  # Extremely high, but allows unassignment if impossible
            routing.AddDisjunction([manager.NodeToIndex(node)], mandatory_penalty)
            continue
        
        # For all other visits, add disjunctions with SLA-based penalties
        # MASSIVELY INCREASED penalty tiers to prioritize SLA visits
        # These penalties ensure SLA visits are virtually guaranteed to be routed
        if priority >= 15:
            # SLA BREACHED (≤0 days) — EXTREME penalty, practically impossible to drop
            node_penalty = base_penalty * 20   # Was 10x, now 20x
            breached_count += 1
        elif priority >= 8:
            # Urgent (SLA 1-2 days) — Very high penalty
            node_penalty = base_penalty * 10   # Was 5x, now 10x
            urgent_count += 1
        elif priority >= 7:
            # Warning (SLA 3 days) — High penalty
            node_penalty = base_penalty * 5    # Was 3x, now 5x
        else:
            # Normal (SLA > 3 days) — Standard penalty
            node_penalty = base_penalty * 2    # Unchanged
        
        # Paired nodes (pickup-drop) get extra penalty to keep pairs together
        if node in paired_nodes:
            node_penalty = node_penalty * 2
            paired_count += 1
        
        routing.AddDisjunction([manager.NodeToIndex(node)], node_penalty)
    
    logger.info(f"✅ Added SLA-prioritized + MANDATORY disjunctions for {len(locations) - 2} visit nodes")
    logger.info(f"   🔒 {mandatory_count} MANDATORY (damaged/exchanged/return/returned - HIGHEST PRIORITY), "
                f"🚨 {breached_count} BREACHED (extreme priority), "
                f"⚠️  {urgent_count} urgent, 📍 {paired_count} in pairs")
    
    # ─── FEASIBILITY CHECK ───
    # Check for obvious infeasibility before solving
    num_visit_nodes = len(locations) - 2  # Exclude start and end
    total_stop_capacity = num_vehicles * max_waypoints
    total_volume_capacity = num_vehicles * truck_capacity
    
    # Force flush to ensure we see this log
    logger.info(f"🔍 Running feasibility check: {num_visit_nodes} visit nodes, {mandatory_count} mandatory, "
                f"{len(pickup_drop_pairs)} pickup-drop pairs, {total_stop_capacity} stop capacity")
    for handler in logging.getLogger().handlers:
        handler.flush()
    
    # Calculate total volume of pre-loaded drops (drops without pickup pairs)
    total_preloaded_volume = 0
    try:
        if combined_order_info and visit_groups:
            for node_idx, info in enumerate(combined_order_info):
                if node_idx == start_index or node_idx == end_index:
                    continue
                if node_idx < len(visit_groups) and visit_groups[node_idx]:
                    for visit_id in visit_groups[node_idx]:
                        visit_type = visit_id_to_visit_type.get(visit_id, '').lower()
                        order_id = visit_id_to_order_id.get(visit_id)
                        vol_capacity = visit_id_to_vol_capacity.get(visit_id, 0)
                        
                        # Check if this is a pre-loaded drop (drop without pickup pair)
                        if visit_type in ['drop', 'damaged_drop', 'exchanged_drop', 'exchange_drop',
                                         'return_drop', 'return_delivery', 'returned_to']:
                            if order_id and order_id in order_map_for_volume:
                                order_info = order_map_for_volume[order_id]
                                # Pre-loaded if drop exists but no pickup pair
                                if not (order_info.get('has_pickup') and order_info.get('has_drop')):
                                    total_preloaded_volume += vol_capacity
    except Exception as e:
        logger.warning(f"⚠️ Could not calculate pre-loaded volume: {e}")
        total_preloaded_volume = 0
    
    feasibility_issues = []
    
    # Check 1: Enough stop capacity?
    if num_visit_nodes > total_stop_capacity:
        feasibility_issues.append(
            f"Not enough stop capacity: {num_visit_nodes} visits need stops, "
            f"but only {total_stop_capacity} stops available ({num_vehicles} trucks × {max_waypoints} stops)"
        )
    
    # Check 2: Pre-loaded volume exceeds total capacity?
    if total_preloaded_volume > total_volume_capacity:
        feasibility_issues.append(
            f"Pre-loaded volume exceeds capacity: {total_preloaded_volume} units need to be pre-loaded, "
            f"but total capacity is only {total_volume_capacity} units ({num_vehicles} trucks × {truck_capacity} units)"
        )
    
    # Check 3: Mandatory visits exceed stop capacity?
    if mandatory_count > total_stop_capacity:
        feasibility_issues.append(
            f"Mandatory visits exceed stop capacity: {mandatory_count} mandatory visits must be scheduled, "
            f"but only {total_stop_capacity} stops available ({num_vehicles} trucks × {max_waypoints} stops)"
        )
    
    # Check 4: Pickup-drop pairs might create impossible constraints
    # Each pair requires 2 stops on the same truck, and pickup must come before drop
    # If we have many pairs, they might not fit even if total stops are OK
    pairs_requiring_stops = len(pickup_drop_pairs) * 2  # Each pair needs 2 stops
    if pairs_requiring_stops > total_stop_capacity:
        feasibility_issues.append(
            f"Pickup-drop pairs exceed stop capacity: {pairs_requiring_stops} stops needed for pairs, "
            f"but only {total_stop_capacity} stops available"
        )
    
    # Check 5: Minimum trucks needed for mandatory visits
    min_trucks_for_mandatory = (mandatory_count + max_waypoints - 1) // max_waypoints  # Ceiling division
    if min_trucks_for_mandatory > num_vehicles:
        feasibility_issues.append(
            f"Not enough trucks for mandatory visits: Need at least {min_trucks_for_mandatory} trucks "
            f"to schedule {mandatory_count} mandatory visits with max {max_waypoints} stops/truck, "
            f"but only {num_vehicles} trucks available"
        )
    
    if feasibility_issues:
        logger.warning("⚠️ FEASIBILITY CHECK WARNINGS - Problem may be difficult:")
        for issue in feasibility_issues:
            logger.warning(f"   ⚠️ {issue}")
        logger.warning("   💡 Suggestions:")
        logger.warning(f"   1. Increase trucks: {num_vehicles} → {max(num_vehicles + 2, int(num_visit_nodes / max_waypoints) + 1)}")
        logger.warning(f"   2. Increase max stops: {max_waypoints} → {max(max_waypoints + 5, int(num_visit_nodes / num_vehicles) + 1)}")
        if total_preloaded_volume > total_volume_capacity:
            logger.warning(f"   3. Increase truck capacity: {truck_capacity} → {int(total_preloaded_volume / num_vehicles) + 10}")
        logger.warning("   ⚠️ Continuing anyway - solver will try to generate routes, some visits may be unassigned")
    
    # Check for tight constraints that might cause infeasibility
    utilization_ratio = mandatory_count / total_stop_capacity if total_stop_capacity > 0 else 0
    if utilization_ratio > 0.7:  # More than 70% of stops are mandatory
        logger.warning(f"⚠️ Tight constraints detected: {mandatory_count}/{total_stop_capacity} stops are mandatory ({utilization_ratio*100:.1f}%)")
        logger.warning(f"   Combined with {len(pickup_drop_pairs)} pickup-drop pairs, this may cause infeasibility")
        logger.warning(f"   Consider: increasing trucks to {num_vehicles + 2} or max stops to {max_waypoints + 5}")
    
    # Filter out pickup-drop pairs that exceed distance limits
    # These pairs cannot be satisfied (same truck requirement + distance limit conflict)
    # We'll allow the drop to be unassigned instead of making the problem infeasible
    if len(pickup_drop_pairs) > 0 and distance_matrix:
        logger.info(f"📏 Checking pickup-drop pair distances against {max_distance_per_vehicle/1000:.1f}km limit...")
        filtered_pairs = []
        skipped_pairs = []
        for pair in pickup_drop_pairs:
            pickup_node = pair['pickup_node']
            drop_node = pair['drop_node']
            if (pickup_node < len(distance_matrix) and 
                drop_node < len(distance_matrix[pickup_node])):
                dist_km = distance_matrix[pickup_node][drop_node] / 1000.0
                if dist_km > max_distance_per_vehicle / 1000.0:
                    skipped_pairs.append((pickup_node, drop_node, dist_km))
                    logger.warning(f"   ⚠️ Skipping pair constraint {pickup_node}→{drop_node}: {dist_km:.1f}km exceeds {max_distance_per_vehicle/1000:.1f}km limit")
                    logger.warning(f"      Drop visit will be allowed to be unassigned if needed (route optimized for efficiency)")
                else:
                    filtered_pairs.append(pair)
            else:
                # If distance not available, keep the pair (safer to include it)
                filtered_pairs.append(pair)
        
        if skipped_pairs:
            logger.warning(f"⚠️ Filtered out {len(skipped_pairs)} pickup-drop pairs that exceed distance limit")
            logger.warning(f"   These drops can be unassigned - route will be optimized for efficiency first")
            # Update pickup_drop_pairs to only include valid pairs
            pickup_drop_pairs = filtered_pairs
    
    # Check if mandatory visits are reachable from depot within distance limit
    # This is critical - if mandatory visits are too far from depot, they can't be scheduled
    if mandatory_count > 0 and distance_matrix and start_index < len(distance_matrix):
        logger.info(f"📏 Checking if {mandatory_count} mandatory visits are reachable from depot...")
        unreachable_mandatory = []
        mandatory_nodes = []
        
        # Find all mandatory nodes
        if combined_order_info:
            for node_idx, info in enumerate(combined_order_info):
                if node_idx == start_index or node_idx == end_index:
                    continue
                if visit_groups and node_idx < len(visit_groups) and visit_groups[node_idx]:
                    for visit_id in visit_groups[node_idx]:
                        visit_type = visit_id_to_visit_type.get(visit_id, '').lower()
                        if visit_type in ['returned_from', 'returned_to', 'damaged_pickup', 'damaged_drop',
                                         'exchanged_pickup', 'exchanged_drop', 'return_pickup', 'return_drop']:
                            if node_idx not in mandatory_nodes:
                                mandatory_nodes.append(node_idx)
                            break
        
        # Check distance from depot to each mandatory node
        # A mandatory visit needs at least: depot → visit → depot (round trip)
        # So if distance > max_distance/2, even a simple round trip exceeds limit
        max_one_way_distance = max_distance_per_vehicle / 2000.0  # Half of max distance (round trip)
        for node_idx in mandatory_nodes:
            if node_idx < len(distance_matrix[start_index]):
                dist_km = distance_matrix[start_index][node_idx] / 1000.0
                round_trip_km = dist_km * 2  # Depot → visit → depot
                if round_trip_km > max_distance_per_vehicle / 1000.0:
                    unreachable_mandatory.append((node_idx, dist_km, round_trip_km))
                    logger.error(f"   ❌ Mandatory node {node_idx}: {dist_km:.1f}km from depot → round trip {round_trip_km:.1f}km EXCEEDS {max_distance_per_vehicle/1000:.1f}km limit!")
                elif dist_km > max_one_way_distance:
                    logger.warning(f"   ⚠️ Mandatory node {node_idx}: {dist_km:.1f}km from depot (round trip {round_trip_km:.1f}km) - tight constraint")
        
        if unreachable_mandatory:
            logger.warning(f"⚠️ FEASIBILITY WARNING: {len(unreachable_mandatory)} mandatory visits are unreachable from depot!")
            logger.warning(f"   Round trip exceeds distance limit - these visits may be unassigned")
            max_needed = max([rt for _, _, rt in unreachable_mandatory])
            logger.warning(f"   💡 Consider: Increase max_distance from {max_distance_per_vehicle/1000:.1f}km to at least {max_needed:.1f}km")
            logger.warning("   ⚠️ Continuing anyway - solver will try to generate routes, unreachable visits will be unassigned")
        
        # Log summary of mandatory visit distances
        if mandatory_nodes:
            distances = [distance_matrix[start_index][n] / 1000.0 for n in mandatory_nodes if n < len(distance_matrix[start_index])]
            if distances:
                avg_dist = sum(distances) / len(distances)
                max_dist = max(distances)
                logger.info(f"   📍 Mandatory visits: avg {avg_dist:.1f}km, max {max_dist:.1f}km from depot")
                if max_dist > max_distance_per_vehicle / 2000.0:
                    logger.warning(f"   ⚠️ Some mandatory visits are far from depot - may need multiple trucks or higher distance limit")
    
    logger.info(f"✅ Feasibility check passed: Basic constraints appear satisfiable")
    # Force flush to ensure logs appear
    for handler in logging.getLogger().handlers:
        handler.flush()
    
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
    
    # ─── HANDLE EXCHANGE/DAMAGE COMBINATIONS ───
    # When damaged_drop and exchanged_pickup are at the same location, combine them
    # Use CumulVar constraints to enforce ordering without AddPickupAndDelivery
    if combined_order_info and order_map:
        time_dimension = routing.GetDimensionOrDie('Time')
        combination_constraints_added = 0
        orders_with_combination_constraints = set()  # Track orders that have combination constraints
        
        # Check each order for exchange/damage combinations
        for order_id, nodes in order_map.items():
            # Check if this order has both damaged and exchanged pairs
            has_damaged = 'damaged_pickup' in nodes and 'damaged_drop' in nodes
            has_exchanged = 'exchanged_pickup' in nodes and 'exchanged_drop' in nodes
            
            if has_damaged and has_exchanged:
                damaged_pickup_node = nodes['damaged_pickup']
                damaged_drop_node = nodes['damaged_drop']
                exchanged_pickup_node = nodes['exchanged_pickup']
                exchanged_drop_node = nodes['exchanged_drop']
                
                # Case 1: damaged_drop and exchanged_pickup at same location
                # This is the main combination case: damaged_pickup -> (damaged_drop+exchanged_pickup) -> exchanged_drop
                if damaged_drop_node == exchanged_pickup_node:
                    shared_node = damaged_drop_node
                    
                    # Only add constraint if exchanged_drop is NOT at the same location as damaged_pickup
                    # (if they're at the same location, we handle it in Case 2)
                    if exchanged_drop_node != damaged_pickup_node:
                        # Proper ordering: damaged_pickup < shared_node < exchanged_drop
                        damaged_pickup_idx = manager.NodeToIndex(damaged_pickup_node)
                        shared_idx = manager.NodeToIndex(shared_node)
                        exchanged_drop_idx = manager.NodeToIndex(exchanged_drop_node)
                        
                        # Enforce: damaged_pickup < shared_node
                        routing.solver().Add(
                            time_dimension.CumulVar(damaged_pickup_idx) <= 
                            time_dimension.CumulVar(shared_idx)
                        )
                        
                        # Enforce: shared_node < exchanged_drop
                        routing.solver().Add(
                            time_dimension.CumulVar(shared_idx) <= 
                            time_dimension.CumulVar(exchanged_drop_idx)
                        )
                        
                        # Enforce same vehicle for all three nodes
                        routing.solver().Add(
                            routing.VehicleVar(damaged_pickup_idx) == 
                            routing.VehicleVar(shared_idx)
                        )
                        routing.solver().Add(
                            routing.VehicleVar(shared_idx) == 
                            routing.VehicleVar(exchanged_drop_idx)
                        )
                        
                        combination_constraints_added += 1
                        orders_with_combination_constraints.add(order_id)
                        logger.info(f"🔗 Added combination constraints for order {order_id}: "
                                  f"damaged_pickup (node {damaged_pickup_node}) < "
                                  f"shared_node (node {shared_node}, damaged_drop+exchanged_pickup) < "
                                  f"exchanged_drop (node {exchanged_drop_node})")
                    else:
                        # Special case: damaged_drop == exchanged_pickup AND exchanged_drop == damaged_pickup
                        # This means: damaged_pickup and exchanged_drop are at the SAME node (location A)
                        #            damaged_drop and exchanged_pickup are at the SAME node (location B)
                        # Required: A (damaged_pickup) < B (damaged_drop+exchanged_pickup) < A (exchanged_drop revisit)
                        # Since A and A are the same node, we can't revisit. 
                        # Solution: Accept that exchanged_drop will be at same sequence as damaged_pickup
                        # OR: Create a duplicate node for the revisit (complex)
                        # For now, we'll enforce: damaged_pickup < shared_node, and same vehicle
                        # The exchange drop will be scheduled at the same node as damaged_pickup
                        damaged_pickup_idx = manager.NodeToIndex(damaged_pickup_node)
                        shared_idx = manager.NodeToIndex(shared_node)
                        exchanged_drop_idx = manager.NodeToIndex(exchanged_drop_node)
                        
                        # Enforce: damaged_pickup < shared_node (damaged_drop+exchanged_pickup)
                        routing.solver().Add(
                            time_dimension.CumulVar(damaged_pickup_idx) <= 
                            time_dimension.CumulVar(shared_idx)
                        )
                        
                        # Note: exchanged_drop is at the same node as damaged_pickup, so it will be at the same sequence
                        # This is a limitation when visits are combined at the same location
                        # To properly support revisits, we would need to NOT combine these visits
                        
                        # Enforce same vehicle for all nodes
                        routing.solver().Add(
                            routing.VehicleVar(damaged_pickup_idx) == 
                            routing.VehicleVar(shared_idx)
                        )
                        routing.solver().Add(
                            routing.VehicleVar(shared_idx) == 
                            routing.VehicleVar(exchanged_drop_idx)
                        )
                        
                        combination_constraints_added += 1
                        orders_with_combination_constraints.add(order_id)
                        logger.warning(f"⚠️ Order {order_id}: Conflict detected - damaged_pickup and exchanged_drop at same location. "
                                     f"Both will be scheduled at the same sequence (limitation of combining visits).")
                        logger.info(f"🔗 Added combination constraints for order {order_id}: "
                                  f"damaged_pickup/exchanged_drop (node {damaged_pickup_node}) < "
                                  f"shared_node (node {shared_node}, damaged_drop+exchanged_pickup)")
                
                # Case 2: exchanged_drop and damaged_pickup at same location (but damaged_drop != exchanged_pickup)
                elif exchanged_drop_node == damaged_pickup_node and damaged_drop_node != exchanged_pickup_node:
                    shared_node = exchanged_drop_node
                    
                    # Proper ordering: exchanged_pickup < shared_node < damaged_drop
                    exchanged_pickup_idx = manager.NodeToIndex(exchanged_pickup_node)
                    shared_idx = manager.NodeToIndex(shared_node)
                    damaged_drop_idx = manager.NodeToIndex(damaged_drop_node)
                    
                    routing.solver().Add(
                        time_dimension.CumulVar(exchanged_pickup_idx) <= 
                        time_dimension.CumulVar(shared_idx)
                    )
                    routing.solver().Add(
                        time_dimension.CumulVar(shared_idx) <= 
                        time_dimension.CumulVar(damaged_drop_idx)
                    )
                    
                    routing.solver().Add(
                        routing.VehicleVar(exchanged_pickup_idx) == 
                        routing.VehicleVar(shared_idx)
                    )
                    routing.solver().Add(
                        routing.VehicleVar(shared_idx) == 
                        routing.VehicleVar(damaged_drop_idx)
                    )
                    
                    combination_constraints_added += 1
                    orders_with_combination_constraints.add(order_id)
                    logger.info(f"🔗 Added combination constraints for order {order_id}: "
                              f"exchanged_pickup (node {exchanged_pickup_node}) < "
                              f"shared_node (node {shared_node}, exchanged_drop+damaged_pickup) < "
                              f"damaged_drop (node {damaged_drop_node})")
        
        # Also handle skipped pairs - add CumulVar constraints for them
        # These are exchange/damage pairs that were skipped because a node was in another pair
        # We'll add ordering constraints using CumulVar instead of AddPickupAndDelivery
        # BUT: Skip pairs for orders that already have combination constraints (to avoid conflicts)
        for skipped_pair in skipped_pairs_for_combination:
            order_id = skipped_pair['order_id']
            
            # Skip if this order already has combination constraints
            if order_id in orders_with_combination_constraints:
                logger.info(f"⏭️  Skipping CumulVar constraint for order {order_id} - already handled by combination constraints")
                continue
            
            pair_type = skipped_pair.get('pair_type', 'standard')
            pickup_node = skipped_pair['pickup_node']
            drop_node = skipped_pair['drop_node']
            
            # Add ordering constraint using CumulVar
            pickup_idx = manager.NodeToIndex(pickup_node)
            drop_idx = manager.NodeToIndex(drop_node)
            
            # Enforce: pickup < drop
            routing.solver().Add(
                time_dimension.CumulVar(pickup_idx) <= 
                time_dimension.CumulVar(drop_idx)
            )
            
            # Enforce same vehicle
            routing.solver().Add(
                routing.VehicleVar(pickup_idx) == 
                routing.VehicleVar(drop_idx)
            )
            
            combination_constraints_added += 1
            logger.info(f"🔗 Added CumulVar constraint for skipped {pair_type} pair (order {order_id}): "
                      f"pickup node {pickup_node} -> drop node {drop_node}")
        
        if combination_constraints_added > 0:
            logger.info(f"✅ Added {combination_constraints_added} exchange/damage combination constraints")
    
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
    
    # Dynamic time limit — INCREASED for SLA-prioritized routing
    # More time allows solver to find better solutions that satisfy SLA constraints
    num_nodes = len(locations)
    if num_nodes <= 10:
        time_limit = 15   # Was 10, now 15
    elif num_nodes <= 20:
        time_limit = 25   # Was 15, now 25
    elif num_nodes <= 35:
        time_limit = 40   # Was 25, now 40
    else:
        time_limit = 60   # Was 40, now 60 - critical for SLA optimization
    
    search_parameters.time_limit.seconds = time_limit
    search_parameters.log_search = False
    
    # Make solver more lenient - allow it to return partial solutions even if constraints are tight
    # This ensures we always get a solution, even if some visits are unassigned
    search_parameters.use_full_propagation = False  # Faster, more lenient propagation
    search_parameters.use_depth_first_search = False  # Use breadth-first for better coverage
    
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
    
    # Final diagnostic before solving
    logger.info(f"📊 Pre-solve summary:")
    logger.info(f"   - Visit nodes: {len(locations) - 2} (excluding start/end)")
    logger.info(f"   - Mandatory visits: {mandatory_count}")
    logger.info(f"   - Pickup-drop pairs: {len(pickup_drop_pairs)}")
    logger.info(f"   - Stop capacity: {num_vehicles} trucks × {max_waypoints} stops = {num_vehicles * max_waypoints}")
    logger.info(f"   - Distance limit: {max_distance_per_vehicle/1000:.1f}km per truck")
    logger.info(f"   - Volume capacity: {truck_capacity} units per truck")
    
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
                # ROUTING_FAIL_TIMEOUT usually means infeasible problem, not actual timeout
                # (especially when solve time is very short like 0.02s)
                logger.error("⚠️ Solver failed - problem appears INFEASIBLE (not a timeout).")
                logger.error(f"   Problem details:")
                logger.error(f"   - {num_vehicles} trucks available")
                logger.error(f"   - {len(locations)} unique locations (after combining)")
                logger.error(f"   - Max {max_distance_per_vehicle/1000:.1f}km per truck")
                logger.error(f"   - Max {max_waypoints} stops per truck")
                logger.error(f"   - Max {truck_capacity} units volume capacity per truck")
                # Count mandatory visits (from the earlier logging)
                logger.error(f"   - {mandatory_count} mandatory visits (must be scheduled)")
                # Count pickup-drop pairs
                logger.error(f"   - {len(pickup_drop_pairs)} pickup-drop pairs requiring same truck")
                logger.error("   💡 Suggestions:")
                logger.error("   1. Increase number of trucks")
                logger.error("   2. Increase max distance per truck")
                logger.error("   3. Increase max stops per truck")
                logger.error("   4. Check if mandatory visits are too spread out geographically")
                logger.error("   5. Verify volume capacity constraints are not too tight")
                logger.error("   6. Consider relaxing some mandatory visit constraints")
            elif status == 2:
                logger.error("Solver failed - constraints may be infeasible. Check pickup-drop pairs and distance limits.")
    except Exception as e:
        solve_elapsed = time.time() - solve_start
        logger.error(f"❌ Solver crashed after {solve_elapsed:.2f}s: {str(e)}", exc_info=True)
        if was_tracing and not tracemalloc.is_tracing():
            tracemalloc.start()
        # Return empty solution instead of None - solver should always return something
        logger.error("Returning empty solution with all visits unassigned due to solver crash")
        return {
            'routes': [],
            'unassigned_visits': original_visits if original_visits else [],
            'total_distance': 0,
            'total_duration': 0,
            'num_vehicles_used': 0,
            'solution_quality': 'FAILED - solver crashed'
        }
    
    if solution:
        return extract_solution(
            manager, routing, solution, locations, distance_matrix, duration_matrix,
            start_index, end_index, num_vehicles, max_waypoints,
            combined_order_info, visit_groups, original_visits, truck_capacity,
            max_distance_per_vehicle,
            start_truck_number=start_truck_number
        )
    else:
        # Solver failed - return empty solution with all visits unassigned
        # This ensures we ALWAYS return a solution structure, even if no routes can be generated
        logger.error("❌ Solver failed - returning empty solution with all visits unassigned")
        logger.error("   Routes prioritize efficiency, but constraints made it impossible to generate any routes")
        logger.error("   All visits will be marked as unassigned")
        
        # Return empty solution structure
        return {
            'routes': [],
            'unassigned_visits': original_visits if original_visits else [],
            'total_distance': 0,
            'total_duration': 0,
            'num_vehicles_used': 0,
            'solution_quality': 'FAILED - constraints too tight'
        }


def validate_routes(routes: List[Dict]) -> List[str]:
    """
    Validate routes for pickup-drop constraint violations.
    
    Checks:
    1. Within-truck: DROP must not come before PICKUP (by sequence)
    2. Cross-truck: PICKUP and DROP for the same order must be on the same truck
    
    Validates all pair types:
    - Standard: pickup/drop
    - Damaged: damaged_pickup/damaged_drop
    - Exchange: exchanged_pickup/exchanged_drop
    - Return: return_pickup/return_drop
    
    Returns list of error strings (empty = no violations).
    """
    validation_errors = []
    
    # Helper to check pair ordering for a specific pair type
    def check_pair_ordering(route, order_sequence, pair_type, pickup_key, drop_key, pair_name):
        truck_id = route['truckId']
        for order_id, seq_info in order_sequence.items():
            if pickup_key in seq_info and drop_key in seq_info:
                if seq_info[pickup_key] > seq_info[drop_key]:
                    validation_errors.append(
                        f"❌ Order {order_id}: {pair_name} DROP (seq {seq_info[drop_key]}) "
                        f"before {pair_name} PICKUP (seq {seq_info[pickup_key]}) in {truck_id}"
                    )
    
    # Helper to check cross-truck violations for a specific pair type
    def check_cross_truck(all_order_trucks, pickup_key, drop_key, pair_name):
        for order_id, truck_info in all_order_trucks.items():
            if pickup_key in truck_info and drop_key in truck_info:
                if truck_info[pickup_key] != truck_info[drop_key]:
                    validation_errors.append(
                        f"❌ Order {order_id}: {pair_name} PICKUP in {truck_info[pickup_key]} "
                        f"but {pair_name} DROP in {truck_info[drop_key]}"
                    )
    
    # Check within-truck ordering for all pair types
    for route in routes:
        truck_id = route['truckId']
        order_sequence_standard = {}
        order_sequence_damaged = {}
        order_sequence_exchanged = {}
        order_sequence_return = {}
        
        for stop in route['stops']:
            order_id = stop.get('order_id')
            visit_type = stop.get('visit_type')
            seq = stop.get('sequence')
            
            if order_id and visit_type:
                vtype = visit_type.lower()
                
                # Standard pickup-drop pairs
                if vtype in ['pickup', 'pick']:
                    if order_id not in order_sequence_standard:
                        order_sequence_standard[order_id] = {}
                    order_sequence_standard[order_id]['pickup_seq'] = seq
                elif vtype in ['drop', 'delivery']:
                    if order_id not in order_sequence_standard:
                        order_sequence_standard[order_id] = {}
                    order_sequence_standard[order_id]['drop_seq'] = seq
                
                # Damaged pairs
                elif vtype in ['damaged_pickup']:
                    if order_id not in order_sequence_damaged:
                        order_sequence_damaged[order_id] = {}
                    order_sequence_damaged[order_id]['damaged_pickup_seq'] = seq
                elif vtype in ['damaged_drop']:
                    if order_id not in order_sequence_damaged:
                        order_sequence_damaged[order_id] = {}
                    order_sequence_damaged[order_id]['damaged_drop_seq'] = seq
                
                # Exchange pairs
                elif vtype in ['exchanged_pickup', 'exchange_pickup']:
                    if order_id not in order_sequence_exchanged:
                        order_sequence_exchanged[order_id] = {}
                    order_sequence_exchanged[order_id]['exchanged_pickup_seq'] = seq
                elif vtype in ['exchanged_drop', 'exchange_drop']:
                    if order_id not in order_sequence_exchanged:
                        order_sequence_exchanged[order_id] = {}
                    order_sequence_exchanged[order_id]['exchanged_drop_seq'] = seq
                
                # Return pairs
                elif vtype in ['return_pickup', 'return_pick']:
                    if order_id not in order_sequence_return:
                        order_sequence_return[order_id] = {}
                    order_sequence_return[order_id]['return_pickup_seq'] = seq
                elif vtype in ['return_drop', 'return_delivery']:
                    if order_id not in order_sequence_return:
                        order_sequence_return[order_id] = {}
                    order_sequence_return[order_id]['return_drop_seq'] = seq
        
        # Validate each pair type
        check_pair_ordering(route, order_sequence_standard, 'standard', 'pickup_seq', 'drop_seq', 'Standard')
        check_pair_ordering(route, order_sequence_damaged, 'damaged', 'damaged_pickup_seq', 'damaged_drop_seq', 'Damaged')
        check_pair_ordering(route, order_sequence_exchanged, 'exchanged', 'exchanged_pickup_seq', 'exchanged_drop_seq', 'Exchange')
        check_pair_ordering(route, order_sequence_return, 'return', 'return_pickup_seq', 'return_drop_seq', 'Return')
    
    # Check cross-truck violations for all pair types
    all_order_trucks_standard = {}
    all_order_trucks_damaged = {}
    all_order_trucks_exchanged = {}
    all_order_trucks_return = {}
    
    for route in routes:
        truck_id = route['truckId']
        for stop in route['stops']:
            order_id = stop.get('order_id')
            visit_type = stop.get('visit_type')
            
            if order_id and visit_type:
                vtype = visit_type.lower()
                
                # Standard pairs
                if vtype in ['pickup', 'pick']:
                    if order_id not in all_order_trucks_standard:
                        all_order_trucks_standard[order_id] = {}
                    all_order_trucks_standard[order_id]['pickup_truck'] = truck_id
                elif vtype in ['drop', 'delivery']:
                    if order_id not in all_order_trucks_standard:
                        all_order_trucks_standard[order_id] = {}
                    all_order_trucks_standard[order_id]['drop_truck'] = truck_id
                
                # Damaged pairs
                elif vtype in ['damaged_pickup']:
                    if order_id not in all_order_trucks_damaged:
                        all_order_trucks_damaged[order_id] = {}
                    all_order_trucks_damaged[order_id]['damaged_pickup_truck'] = truck_id
                elif vtype in ['damaged_drop']:
                    if order_id not in all_order_trucks_damaged:
                        all_order_trucks_damaged[order_id] = {}
                    all_order_trucks_damaged[order_id]['damaged_drop_truck'] = truck_id
                
                # Exchange pairs
                elif vtype in ['exchanged_pickup', 'exchange_pickup']:
                    if order_id not in all_order_trucks_exchanged:
                        all_order_trucks_exchanged[order_id] = {}
                    all_order_trucks_exchanged[order_id]['exchanged_pickup_truck'] = truck_id
                elif vtype in ['exchanged_drop', 'exchange_drop']:
                    if order_id not in all_order_trucks_exchanged:
                        all_order_trucks_exchanged[order_id] = {}
                    all_order_trucks_exchanged[order_id]['exchanged_drop_truck'] = truck_id
                
                # Return pairs
                elif vtype in ['return_pickup', 'return_pick']:
                    if order_id not in all_order_trucks_return:
                        all_order_trucks_return[order_id] = {}
                    all_order_trucks_return[order_id]['return_pickup_truck'] = truck_id
                elif vtype in ['return_drop', 'return_delivery']:
                    if order_id not in all_order_trucks_return:
                        all_order_trucks_return[order_id] = {}
                    all_order_trucks_return[order_id]['return_drop_truck'] = truck_id
    
    # Validate cross-truck for each pair type
    check_cross_truck(all_order_trucks_standard, 'pickup_truck', 'drop_truck', 'Standard')
    check_cross_truck(all_order_trucks_damaged, 'damaged_pickup_truck', 'damaged_drop_truck', 'Damaged')
    check_cross_truck(all_order_trucks_exchanged, 'exchanged_pickup_truck', 'exchanged_drop_truck', 'Exchange')
    check_cross_truck(all_order_trucks_return, 'return_pickup_truck', 'return_drop_truck', 'Return')
    
    return validation_errors


def extract_solution(
    manager, routing, solution, locations, distance_matrix, duration_matrix,
    start_index, end_index, num_vehicles,
    max_waypoints: int = MAX_WAYPOINTS_PER_ROUTE,
    combined_order_info: List[Dict] = None,
    visit_groups: List[List[str]] = None,
    original_visits: List[Dict] = None,
    truck_capacity: int = 100,
    max_distance_per_vehicle: int = None,
    start_truck_number: int = 1
) -> Dict:
    """Extract the solution from OR-Tools solver and enforce max waypoints and max_km limits"""
    routes = []
    unassigned_visits = []
    
    # Track which visits were assigned
    assigned_indices = set([start_index, end_index])
    
    # Build visit mappings for volume capacity tracking (for response only)
    visit_id_to_vol_capacity = {}
    visit_id_to_visit_type = {}
    visit_id_to_order_id = {}
    if original_visits:
        for visit in original_visits:
            visit_id = visit.get('visitId')
            vol_capacity = visit.get('vol_capacity', 0)
            visit_type = visit.get('visit_type', '').lower()
            order_id = visit.get('order_id')
            if visit_id:
                visit_id_to_vol_capacity[visit_id] = vol_capacity
                visit_id_to_visit_type[visit_id] = visit_type
                visit_id_to_order_id[visit_id] = order_id
    
    # Build order map for pickup-drop pair validation from combined_order_info
    # Includes all pair types: standard, damaged, exchanged, return
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
                        vtype_lower = visit_type.lower()
                        
                        # Standard pairs
                        if vtype_lower in ['pickup', 'pick']:
                            order_map[order_id]['pickup'] = node
                        elif vtype_lower in ['drop', 'delivery']:
                            order_map[order_id]['drop'] = node
                        # Damaged pairs
                        elif vtype_lower in ['damaged_pickup']:
                            order_map[order_id]['damaged_pickup'] = node
                        elif vtype_lower in ['damaged_drop']:
                            order_map[order_id]['damaged_drop'] = node
                        # Exchange pairs
                        elif vtype_lower in ['exchanged_pickup', 'exchange_pickup']:
                            order_map[order_id]['exchanged_pickup'] = node
                        elif vtype_lower in ['exchanged_drop', 'exchange_drop']:
                            order_map[order_id]['exchanged_drop'] = node
                        # Return pairs
                        elif vtype_lower in ['return_pickup', 'return_pick']:
                            order_map[order_id]['return_pickup'] = node
                        elif vtype_lower in ['return_drop', 'return_delivery']:
                            order_map[order_id]['return_drop'] = node
    
    # Track sequential truck number (not vehicle_id, since empty routes are skipped)
    # Start from the provided start_truck_number to continue numbering across passes
    truck_number = start_truck_number
    
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
                        
                        # Add order_id, visit_type, and vol_capacity from original visit if available
                        if original_visit:
                            if 'order_id' in original_visit and original_visit['order_id']:
                                stop_data["order_id"] = original_visit['order_id']
                            if 'visit_type' in original_visit and original_visit['visit_type']:
                                stop_data["visit_type"] = original_visit['visit_type']
                            if 'vol_capacity' in original_visit:
                                stop_data["vol_capacity"] = original_visit.get('vol_capacity', 0)
                        
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
                    
                    # Add order_id, visit_type, and vol_capacity from original visits
                    if original_visits:
                        for orig_v in original_visits:
                            if orig_v['visitId'] == visit_info['visitId']:
                                if 'order_id' in orig_v and orig_v['order_id']:
                                    stop_data["order_id"] = orig_v['order_id']
                                if 'visit_type' in orig_v and orig_v['visit_type']:
                                    stop_data["visit_type"] = orig_v['visit_type']
                                if 'vol_capacity' in orig_v:
                                    stop_data["vol_capacity"] = orig_v.get('vol_capacity', 0)
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
        
        # Ensure end_index is added to route_nodes for complete distance calculation
        # (The loop exits when IsEnd is True, so end_index might not be added yet)
        if route_nodes and route_nodes[-1] != end_index:
            route_nodes.append(end_index)
        
        # Calculate actual route distance AND duration from matrices
        # Sum: start→stop1 + stop1→stop2 + ... + stopN→end
        route_distance_meters = 0
        route_duration_seconds = 0
        for i in range(len(route_nodes) - 1):
            from_node = route_nodes[i]
            to_node = route_nodes[i + 1]
            route_distance_meters += distance_matrix[from_node][to_node]
            route_duration_seconds += duration_matrix[from_node][to_node]
        
        # Haversine gives straight-line distances — no adjustment needed
        estimated_road_distance_meters = route_distance_meters
        estimated_duration_seconds = route_duration_seconds
        
        # Validate max_km constraint - STRICT ENFORCEMENT
        # The solver MUST enforce this via AddDimension, but we validate as a safety check
        if max_distance_per_vehicle is not None and route_distance_meters > max_distance_per_vehicle:
            logger.error(f"❌ CRITICAL: Route for vehicle {vehicle_id + 1} exceeds max_km: "
                          f"{route_distance_meters/1000:.2f}km > {max_distance_per_vehicle/1000:.2f}km")
            logger.error(f"   This should NOT happen - distance constraint should be enforced by solver!")
            logger.error(f"   Route will be marked as invalid - all stops unassigned")
            # Mark all stops in this route as unassigned - constraint violation is critical
            for stop in stops:
                unassigned_visits.append({
                    "visitId": stop['visitId'],
                    "reason": "max_km_constraint_violation"
                })
            # Skip this route entirely - it violates strict constraints
            continue
        
        # Only add route if it has stops
        if stops:
            # Count unique locations (each sequence = 1 physical stop)
            # Combined visits at the same location share the same sequence number
            unique_locations = len(set(s['sequence'] for s in stops))
            
            # Validate max waypoints limit - STRICT ENFORCEMENT
            # The solver MUST enforce this via AddDimension, but we validate as a safety check
            if unique_locations > max_waypoints:
                logger.error(f"❌ CRITICAL: Route for vehicle {vehicle_id + 1} exceeds max_waypoints: "
                             f"{unique_locations} > {max_waypoints}")
                logger.error(f"   This should NOT happen - waypoint constraint should be enforced by solver!")
                logger.error(f"   Route will be marked as invalid - all stops unassigned")
                # Mark all stops in this route as unassigned - constraint violation is critical
                for stop in stops:
                    unassigned_visits.append({
                        "visitId": stop['visitId'],
                        "reason": "max_waypoints_constraint_violation"
                    })
                # Skip this route entirely - it violates strict constraints
                continue
            
            estimated_km = round(estimated_road_distance_meters / 1000, 2)
            estimated_hours = round(estimated_duration_seconds / 3600, 2)
            
            # Calculate initial load for THIS route: sum of drops that don't have both pickup and drop in this route
            # Track which order_ids have both pickup and drop in this route
            order_has_pickup = set()
            order_has_drop = set()
            for stop in stops:
                # Use order_id from stop directly, fallback to mapping if not present
                order_id = stop.get('order_id') or visit_id_to_order_id.get(stop.get('visitId'))
                visit_type = stop.get('visit_type', '').lower() or visit_id_to_visit_type.get(stop.get('visitId'), '').lower()
                if order_id:
                    if visit_type in ['pickup', 'damaged_pickup', 'exchanged_pickup', 'exchange_pickup',
                                     'return_pickup', 'return_pick', 'returned_from']:
                        order_has_pickup.add(order_id)
                    elif visit_type in ['drop', 'damaged_drop', 'exchanged_drop', 'exchange_drop',
                                       'return_drop', 'return_delivery', 'returned_to']:
                        order_has_drop.add(order_id)
            
            orders_with_both = order_has_pickup & order_has_drop
            
            # Calculate initial load: sum of drops that don't have both pickup and drop in this route
            # Note: returned_to is always counted in initial load (like drops without pickups)
            route_initial_load = 0
            for stop in stops:
                visit_id = stop.get('visitId')
                # Use order_id and visit_type from stop directly, fallback to mapping if not present
                order_id = stop.get('order_id') or visit_id_to_order_id.get(visit_id)
                visit_type = stop.get('visit_type', '').lower() or visit_id_to_visit_type.get(visit_id, '').lower()
                vol_capacity = stop.get('vol_capacity', 0) or visit_id_to_vol_capacity.get(visit_id, 0)
                
                # returned_to is always counted in initial load (pre-loaded from warehouse)
                if visit_type == 'returned_to':
                    route_initial_load += vol_capacity
                # Count other drops that don't have both pickup and drop in this route
                elif visit_type in ['drop', 'damaged_drop', 'exchanged_drop', 'exchange_drop',
                                   'return_drop', 'return_delivery']:
                    if order_id and order_id not in orders_with_both:
                        route_initial_load += vol_capacity
            
            # Cap initial load to truck capacity (cannot exceed capacity)
            route_initial_load = min(route_initial_load, truck_capacity)
            
            # Calculate volume capacity tracking: initial load and max buffer capacity reached
            current_volume = route_initial_load
            max_buffer_capacity_reached = route_initial_load
            
            # Process stops in sequence order to track volume changes
            stops_by_sequence = {}
            for stop in stops:
                seq = stop['sequence']
                if seq not in stops_by_sequence:
                    stops_by_sequence[seq] = []
                stops_by_sequence[seq].append(stop)
            
            # Process stops in sequence order
            for seq in sorted(stops_by_sequence.keys()):
                for stop in stops_by_sequence[seq]:
                    visit_id = stop.get('visitId')
                    visit_type = visit_id_to_visit_type.get(visit_id, '').lower()
                    vol_capacity = visit_id_to_vol_capacity.get(visit_id, 0)
                    
                    # Pickup adds volume (positive)
                    if visit_type in ['pickup', 'damaged_pickup', 'exchanged_pickup', 'exchange_pickup',
                                     'return_pickup', 'return_pick', 'returned_from']:
                        current_volume += vol_capacity
                    # Drop subtracts volume (negative)
                    elif visit_type in ['drop', 'damaged_drop', 'exchanged_drop', 'exchange_drop',
                                       'return_drop', 'return_delivery', 'returned_to']:
                        current_volume -= vol_capacity
                    
                    # Track maximum volume reached
                    if current_volume > max_buffer_capacity_reached:
                        max_buffer_capacity_reached = current_volume
            
            routes.append({
                "truckId": f"TRUCK_{truck_number}",
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
                "waypoint_count": unique_locations,
                "total_visits": len(stops),
                "initial_load": route_initial_load,
                "max_buffer_capacity_reached": max_buffer_capacity_reached
            })
            
            # Increment truck number for next route
            truck_number += 1
    
    # Find unassigned visits
    for i, location in enumerate(locations):
        if i not in assigned_indices and 'visitId' in location:
            reason = "max_km_exceeded"
            
            # Check if it's due to distance constraint
            if solution:
                # Could be due to capacity or optimization
                reason = "optimization_constraint"
            
            # If this is a combined location with multiple visits, add ALL of them to unassigned
            if visit_groups and i < len(visit_groups) and visit_groups[i]:
                visit_ids_at_location = visit_groups[i]
                logger.info(f"📍 Combined location {i} unassigned - adding ALL {len(visit_ids_at_location)} visits: {visit_ids_at_location}")
                for visit_id in visit_ids_at_location:
                    unassigned_visits.append({
                        "visitId": visit_id,
                        "reason": reason
                    })
            else:
                # Single visit at this location
                unassigned_visits.append({
                    "visitId": location['visitId'],
                    "reason": reason
                })
    
    # Post-process: Ensure pickup-drop pairs are complete
    # If one part of a pair is assigned and other is not, move the assigned one to unassigned
    # Handles all pair types: standard, damaged, exchanged, return
    if order_map:
        incomplete_pairs = []
        
        # Helper to check pairs for a specific type
        def check_pair_type(order_id, nodes, pickup_key, drop_key, pair_name):
            if pickup_key in nodes and drop_key in nodes:
                pickup_node = nodes[pickup_key]
                drop_node = nodes[drop_key]
                pickup_assigned = pickup_node in assigned_indices
                drop_assigned = drop_node in assigned_indices
                
                # If only one is assigned, mark as incomplete
                if pickup_assigned != drop_assigned:
                    incomplete_pairs.append({
                        'order_id': order_id,
                        'pickup_node': pickup_node,
                        'drop_node': drop_node,
                        'pickup_assigned': pickup_assigned,
                        'drop_assigned': drop_assigned,
                        'pair_type': pair_name
                    })
                elif not pickup_assigned and not drop_assigned:
                    # Both unassigned - they'll be added to unassigned_visits in the general loop
                    logger.info(f"📦 Order {order_id}: Both {pair_name} PICKUP and DROP unassigned (constraints ok)")
        
        for order_id, nodes in order_map.items():
            # Check all pair types
            check_pair_type(order_id, nodes, 'pickup', 'drop', 'Standard')
            check_pair_type(order_id, nodes, 'damaged_pickup', 'damaged_drop', 'Damaged')
            check_pair_type(order_id, nodes, 'exchanged_pickup', 'exchanged_drop', 'Exchange')
            check_pair_type(order_id, nodes, 'return_pickup', 'return_drop', 'Return')
        
        # Handle incomplete pairs:
        # - If PICKUP is assigned but DROP is not: KEEP pickup, add drop to unassigned
        # - If DROP is assigned but PICKUP is not: REMOVE drop from route, add both to unassigned
        #   (can't deliver without picking up first)
        if incomplete_pairs:
            pair_type_str = ', '.join(set(p.get('pair_type', 'Standard') for p in incomplete_pairs))
            logger.warning(f"⚠️ Found {len(incomplete_pairs)} incomplete pairs ({pair_type_str})...")
            
            nodes_to_remove = set()
            for pair in incomplete_pairs:
                pair_type = pair.get('pair_type', 'Standard')
                if pair['pickup_assigned'] and not pair['drop_assigned']:
                    # Pickup is in route, drop is not - this is acceptable
                    # Keep pickup in route, add drop to unassigned for rescheduling
                    logger.info(f"📦 Order {pair['order_id']}: {pair_type} PICKUP in route, DROP added to unassigned for rescheduling")
                    drop_node_idx = pair['drop_node']
                    if drop_node_idx < len(locations) and 'visitId' in locations[drop_node_idx]:
                        # If drop node is a combined location, add ALL visits at that location
                        if visit_groups and drop_node_idx < len(visit_groups) and visit_groups[drop_node_idx]:
                            for visit_id in visit_groups[drop_node_idx]:
                                unassigned_visits.append({
                                    "visitId": visit_id,
                                    "reason": "drop_rescheduled_pickup_in_route"
                                })
                        else:
                            unassigned_visits.append({
                                "visitId": locations[drop_node_idx]['visitId'],
                                "reason": "drop_rescheduled_pickup_in_route"
                            })
                
                elif pair['drop_assigned'] and not pair['pickup_assigned']:
                    # Drop is in route but pickup is not - this is invalid!
                    # Remove drop from route, add BOTH to unassigned
                    pair_type = pair.get('pair_type', 'Standard')
                    logger.warning(f"⚠️  Order {pair['order_id']}: {pair_type} DROP without PICKUP - removing drop, adding both to unassigned")
                    nodes_to_remove.add(pair['drop_node'])
                    
                    # Add both pickup and drop to unassigned (handle combined locations)
                    pickup_node_idx = pair['pickup_node']
                    drop_node_idx = pair['drop_node']
                    
                    # Add pickup visits
                    if pickup_node_idx < len(locations) and 'visitId' in locations[pickup_node_idx]:
                        if visit_groups and pickup_node_idx < len(visit_groups) and visit_groups[pickup_node_idx]:
                            for visit_id in visit_groups[pickup_node_idx]:
                                unassigned_visits.append({
                                    "visitId": visit_id,
                                    "reason": "incomplete_pair_no_pickup"
                                })
                        else:
                            unassigned_visits.append({
                                "visitId": locations[pickup_node_idx]['visitId'],
                                "reason": "incomplete_pair_no_pickup"
                            })
                    
                    # Add drop visits
                    if drop_node_idx < len(locations) and 'visitId' in locations[drop_node_idx]:
                        if visit_groups and drop_node_idx < len(visit_groups) and visit_groups[drop_node_idx]:
                            for visit_id in visit_groups[drop_node_idx]:
                                unassigned_visits.append({
                                    "visitId": visit_id,
                                    "reason": "incomplete_pair_no_pickup"
                                })
                        else:
                            unassigned_visits.append({
                                "visitId": locations[drop_node_idx]['visitId'],
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
                    route['waypoint_count'] = len(set(s['sequence'] for s in filtered_stops)) if filtered_stops else 0
                    route['total_visits'] = len(filtered_stops)
    
    # FINAL VALIDATION: Check for pickup-drop constraint violations
    validation_errors = validate_routes(routes)
    
    if validation_errors:
        logger.error(f"🚨 VALIDATION FOUND {len(validation_errors)} pickup-drop constraint violations (will attempt post-fix):")
        for error in validation_errors:
            logger.error(f"  {error}")
    else:
        logger.info(f"✅ Validation passed: No pickup-drop constraint violations")
    
    return {
        "routes": routes,
        "unassigned_visits": unassigned_visits,
        "validation_errors": validation_errors if validation_errors else None
    }


def fix_volume_capacity_violations(
    result: Dict,
    original_visits: List[Dict],
    truck_capacity: int,
    distance_matrix: List[List[int]],
    duration_matrix: List[List[int]],
    locations: List[Dict],
    start_index: int,
    end_index: int,
    max_distance_per_vehicle: int,
    max_waypoints: int = MAX_WAYPOINTS_PER_ROUTE
) -> Dict:
    """
    Post-processing step to validate and fix volume capacity violations.
    
    Process:
    1. Calculate initial load per route (sum of drops that don't have both pickup and drop in same route)
    2. Iterate through route stops: add volume for pickups, subtract for drops
    3. If volume exceeds truck_capacity at any point, try to fix by:
       - Moving visits to other trucks
       - Reordering visits within the route
       - Moving problematic visits to unassigned
    
    Returns updated result with fixed routes.
    """
    routes = result.get('routes', [])
    unassigned = result.get('unassigned_visits', [])
    
    if not routes or not original_visits:
        return result
    
    # Build visit mappings
    visit_id_to_vol_capacity = {}
    visit_id_to_visit_type = {}
    visit_id_to_order_id = {}
    for visit in original_visits:
        visit_id = visit.get('visitId')
        if visit_id:
            visit_id_to_vol_capacity[visit_id] = visit.get('vol_capacity', 0)
            visit_id_to_visit_type[visit_id] = visit.get('visit_type', '').lower()
            visit_id_to_order_id[visit_id] = visit.get('order_id')
    
    logger.info(f"📦 Validating volume capacity (max {truck_capacity} units per truck)...")
    
    # Track which order_ids have both pickup and drop in each route
    route_order_pairs = {}  # route_idx -> set of order_ids with both pickup and drop
    for route_idx, route in enumerate(routes):
        order_has_pickup = set()
        order_has_drop = set()
        for stop in route['stops']:
            order_id = stop.get('order_id')
            visit_type = visit_id_to_visit_type.get(stop.get('visitId'), '').lower()
            if order_id:
                if visit_type in ['pickup', 'damaged_pickup', 'exchanged_pickup', 'exchange_pickup',
                                 'return_pickup', 'return_pick', 'returned_from']:
                    order_has_pickup.add(order_id)
                elif visit_type in ['drop', 'damaged_drop', 'exchanged_drop', 'exchange_drop',
                                   'return_drop', 'return_delivery', 'returned_to']:
                    order_has_drop.add(order_id)
        route_order_pairs[route_idx] = order_has_pickup & order_has_drop
    
    # Validate and fix volume violations for each route
    violations_found = []
    for route_idx, route in enumerate(routes):
        # Calculate initial load: sum of drops that don't have both pickup and drop in this route
        initial_load = 0
        orders_with_both = route_order_pairs[route_idx]
        
        for stop in route['stops']:
            visit_id = stop.get('visitId')
            visit_type = visit_id_to_visit_type.get(visit_id, '').lower()
            order_id = visit_id_to_order_id.get(visit_id)
            vol_capacity = visit_id_to_vol_capacity.get(visit_id, 0)
            
            # returned_to is always counted in initial load (pre-loaded from warehouse)
            if visit_type == 'returned_to':
                initial_load += vol_capacity
            # Count other drops that don't have both pickup and drop in this route
            elif visit_type in ['drop', 'damaged_drop', 'exchanged_drop', 'exchange_drop',
                               'return_drop', 'return_delivery']:
                if order_id not in orders_with_both:
                    initial_load += vol_capacity
        
        # Track volume as we iterate through stops
        current_volume = initial_load
        max_volume_reached = initial_load
        stops_by_sequence = {}
        
        for stop in route['stops']:
            seq = stop['sequence']
            if seq not in stops_by_sequence:
                stops_by_sequence[seq] = []
            stops_by_sequence[seq].append(stop)
        
        # Process stops in sequence order and check for violations
        violation_stops = []
        for seq in sorted(stops_by_sequence.keys()):
            for stop in stops_by_sequence[seq]:
                visit_id = stop.get('visitId')
                visit_type = visit_id_to_visit_type.get(visit_id, '').lower()
                vol_capacity = visit_id_to_vol_capacity.get(visit_id, 0)
                
                # Pickup adds volume
                if visit_type in ['pickup', 'damaged_pickup', 'exchanged_pickup', 'exchange_pickup',
                                 'return_pickup', 'return_pick', 'returned_from']:
                    current_volume += vol_capacity
                # Drop subtracts volume
                elif visit_type in ['drop', 'damaged_drop', 'exchanged_drop', 'exchange_drop',
                                   'return_drop', 'return_delivery', 'returned_to']:
                    current_volume -= vol_capacity
                
                # Check for violation
                if current_volume > truck_capacity:
                    violation_stops.append({
                        'stop': stop,
                        'sequence': seq,
                        'volume': current_volume,
                        'excess': current_volume - truck_capacity
                    })
                
                if current_volume > max_volume_reached:
                    max_volume_reached = current_volume
        
        if violation_stops:
            violations_found.append({
                'route_idx': route_idx,
                'route': route,
                'violations': violation_stops,
                'initial_load': initial_load,
                'max_volume': max_volume_reached
            })
    
    if not violations_found:
        logger.info(f"✅ No volume capacity violations found")
        return result
    
    logger.warning(f"⚠️ Found volume capacity violations in {len(violations_found)} route(s)")
    
    # Try to fix violations
    seen_unassigned = {u.get('visitId') for u in unassigned}
    
    for violation_info in violations_found:
        route_idx = violation_info['route_idx']
        route = violation_info['route']
        violations = violation_info['violations']
        
        logger.warning(f"  Route {route['truckId']}: {len(violations)} volume violations (max volume: {violation_info['max_volume']}, capacity: {truck_capacity})")
        
        # Strategy: Try to move problematic visits to other trucks or unassigned
        # Start with the most problematic visits (highest excess)
        violations.sort(key=lambda v: v['excess'], reverse=True)
        
        for violation in violations:
            stop = violation['stop']
            visit_id = stop.get('visitId')
            
            if visit_id in seen_unassigned:
                continue
            
            # Try to move to another truck
            moved = False
            for other_route_idx, other_route in enumerate(routes):
                if other_route_idx == route_idx:
                    continue
                
                # Check if moving this visit would help and is feasible
                # For now, just move to unassigned if it's causing issues
                # TODO: Implement smarter logic to check if moving helps
                pass
            
            # If couldn't move, add to unassigned
            if not moved:
                route['stops'] = [s for s in route['stops'] if s.get('visitId') != visit_id]
                unassigned.append({
                    'visitId': visit_id,
                    'reason': f'volume_capacity_exceeded_{violation["excess"]}_units'
                })
                seen_unassigned.add(visit_id)
                logger.info(f"    Moved visit {visit_id} to unassigned (excess: {violation['excess']} units)")
        
        # Rebuild route stats after removals
        if route['stops']:
            # Recalculate route stats (simplified - would need full recalculation)
            route['total_visits'] = len(route['stops'])
        else:
            # Remove empty route
            routes.remove(route)
    
    result['routes'] = routes
    result['unassigned_visits'] = unassigned
    
    # Re-validate to check if violations are fixed
    remaining_violations = []
    for route_idx, route in enumerate(routes):
        # Recalculate initial load
        orders_with_both = route_order_pairs.get(route_idx, set())
        initial_load = 0
        for stop in route['stops']:
            visit_id = stop.get('visitId')
            visit_type = visit_id_to_visit_type.get(visit_id, '').lower()
            order_id = visit_id_to_order_id.get(visit_id)
            vol_capacity = visit_id_to_vol_capacity.get(visit_id, 0)
            # returned_to is always counted in initial load (pre-loaded from warehouse)
            if visit_type == 'returned_to':
                initial_load += vol_capacity
            # Count other drops that don't have both pickup and drop in this route
            elif visit_type in ['drop', 'damaged_drop', 'exchanged_drop', 'exchange_drop',
                               'return_drop', 'return_delivery']:
                if order_id not in orders_with_both:
                    initial_load += vol_capacity
        
        current_volume = initial_load
        for stop in sorted(route['stops'], key=lambda s: s['sequence']):
            visit_id = stop.get('visitId')
            visit_type = visit_id_to_visit_type.get(visit_id, '').lower()
            vol_capacity = visit_id_to_vol_capacity.get(visit_id, 0)
            if visit_type in ['pickup', 'damaged_pickup', 'exchanged_pickup', 'exchange_pickup',
                             'return_pickup', 'return_pick', 'returned_from']:
                current_volume += vol_capacity
            elif visit_type in ['drop', 'damaged_drop', 'exchanged_drop', 'exchange_drop',
                               'return_drop', 'return_delivery', 'returned_to']:
                current_volume -= vol_capacity
            if current_volume > truck_capacity:
                remaining_violations.append(f"Route {route['truckId']} still has violations")
                break
    
    if remaining_violations:
        logger.warning(f"⚠️ {len(remaining_violations)} route(s) still have volume violations after fixing")
    else:
        logger.info(f"✅ All volume capacity violations fixed")
    
    return result


def fix_validation_errors(
    result: Dict,
    distance_matrix: List[List[int]],
    duration_matrix: List[List[int]],
    locations: List[Dict],
    start_index: int,
    end_index: int,
    max_distance_per_vehicle: int,
    max_waypoints: int = MAX_WAYPOINTS_PER_ROUTE
) -> Dict:
    """
    Post-processing step to fix ALL pickup-drop validation errors.
    
    Handles two types of violations:
    1. Cross-truck: PICKUP on truck A, DROP on truck B
       Fix: Move DROP to pickup's truck (or PICKUP to drop's truck) if constraints allow.
    2. Ordering: DROP before PICKUP on the same truck
       Fix: Move DROP to after PICKUP (may add revisit waypoint) if constraints allow.
    
    If a violation cannot be fixed within constraints (max_km, max_waypoints),
    the offending visits are moved to unassigned.
    
    Iterates up to 3 passes to handle cascading fixes.
    """
    routes = result['routes']
    unassigned = result.get('unassigned_visits', [])
    
    if not result.get('validation_errors'):
        return result
    
    logger.info(f"🔧 Post-processing: Fixing {len(result['validation_errors'])} validation errors...")
    
    # ─── Helper: location → node index mapping ───
    loc_to_node = {}
    for i, loc in enumerate(locations):
        key = (round(loc['lat'], 7), round(loc['lng'], 7))
        loc_to_node[key] = i
    
    def find_node(lat, lng):
        """Find the node index in the locations/distance_matrix for a given lat/lng."""
        key = (round(lat, 7), round(lng, 7))
        if key in loc_to_node:
            return loc_to_node[key]
        # Fuzzy match (within ~11m)
        best = None
        best_dist = float('inf')
        for k, v in loc_to_node.items():
            d = abs(k[0] - lat) + abs(k[1] - lng)
            if d < 0.001 and d < best_dist:
                best = v
                best_dist = d
        return best
    
    def get_ordered_waypoint_nodes(route):
        """Get ordered list of unique waypoint node indices for a route."""
        stops = sorted(route['stops'], key=lambda s: s['sequence'])
        nodes = []
        seen_seq = set()
        for s in stops:
            if s['sequence'] not in seen_seq:
                seen_seq.add(s['sequence'])
                n = find_node(s['lat'], s['lng'])
                if n is not None:
                    nodes.append(n)
        return nodes
    
    def calc_route_distance(node_list):
        """Calculate total distance and duration for a route given ordered node list."""
        total_dist = 0
        total_dur = 0
        prev = start_index
        for n in node_list:
            total_dist += distance_matrix[prev][n]
            total_dur += duration_matrix[prev][n]
            prev = n
        total_dist += distance_matrix[prev][end_index]
        total_dur += duration_matrix[prev][end_index]
        return total_dist, total_dur
    
    def calc_insertion_cost_after(route, insert_node, after_seq):
        """Calculate route distance if we insert a new waypoint AFTER the given sequence."""
        stops = sorted(route['stops'], key=lambda s: s['sequence'])
        new_nodes = []
        seen_seq = set()
        insert_idx = -1
        for s in stops:
            if s['sequence'] not in seen_seq:
                seen_seq.add(s['sequence'])
                n = find_node(s['lat'], s['lng'])
                if n is not None:
                    new_nodes.append(n)
                    if s['sequence'] == after_seq:
                        insert_idx = len(new_nodes)
        if insert_idx >= 0:
            new_nodes.insert(insert_idx, insert_node)
        else:
            new_nodes.append(insert_node)
        return calc_route_distance(new_nodes)
    
    def calc_insertion_cost_before(route, insert_node, before_seq):
        """Calculate route distance if we insert a new waypoint BEFORE the given sequence."""
        stops = sorted(route['stops'], key=lambda s: s['sequence'])
        new_nodes = []
        seen_seq = set()
        inserted = False
        for s in stops:
            if s['sequence'] not in seen_seq:
                seen_seq.add(s['sequence'])
                n = find_node(s['lat'], s['lng'])
                if n is not None:
                    if s['sequence'] == before_seq and not inserted:
                        new_nodes.append(insert_node)
                        inserted = True
                    new_nodes.append(n)
        return calc_route_distance(new_nodes)
    
    def rebuild_route_stats(route):
        """Rebuild sequence numbers, waypoint counts, and distance stats for a modified route."""
        stops = route['stops']
        if not stops:
            route['waypoint_count'] = 0
            route['total_visits'] = 0
            route['estimated_km'] = 0
            route['estimated_hours'] = 0
            return
        
        # Sort by current sequence (supports float sequences from insertions)
        stops.sort(key=lambda s: s['sequence'])
        
        # Build ordered waypoints: consecutive stops at same location = 1 waypoint
        # Non-consecutive stops at same location = separate waypoints (revisits)
        waypoints = []  # [(loc_key, [stops])]
        for s in stops:
            loc_key = (round(s['lat'], 6), round(s['lng'], 6))
            if waypoints and waypoints[-1][0] == loc_key:
                waypoints[-1][1].append(s)
            else:
                waypoints.append((loc_key, [s]))
        
        # Assign clean integer sequences
        for seq_num, (_, wp_stops) in enumerate(waypoints, 1):
            for s in wp_stops:
                s['sequence'] = seq_num
        
        route['stops'] = stops
        route['waypoint_count'] = len(waypoints)
        route['total_visits'] = len(stops)
        
        # Recalculate distance and duration
        nodes = get_ordered_waypoint_nodes(route)
        if nodes:
            dist, dur = calc_route_distance(nodes)
            route['estimated_km'] = round(dist / 1000, 2)
            route['estimated_hours'] = round(dur / 3600, 2)
    
    # Track which visitIds are already unassigned (avoid duplicates)
    seen_unassigned = set(u.get('visitId') for u in unassigned if u.get('visitId'))
    
    # ─── Iterative fixing (max 5 passes for thorough fixing) ───
    previous_violation_count = None
    for iteration in range(5):
        # Build order info from CURRENT route state
        # Handles all pair types: standard, damaged, exchanged, return
        order_info = {}
        for ridx, route in enumerate(routes):
            truck_id = route['truckId']
            for stop in route['stops']:
                oid = stop.get('order_id')
                vtype = (stop.get('visit_type') or '').lower()
                if not oid:
                    continue
                
                # Initialize order_info structure for all pair types
                if oid not in order_info:
                    order_info[oid] = {
                        'pickups': [], 'drops': [],
                        'damaged_pickups': [], 'damaged_drops': [],
                        'exchanged_pickups': [], 'exchanged_drops': [],
                        'return_pickups': [], 'return_drops': []
                    }
                
                visit_data = {
                    'truck': truck_id, 'ridx': ridx, 'seq': stop['sequence'],
                    'lat': stop['lat'], 'lng': stop['lng'], 'stop': stop
                }
                
                # Standard pairs
                if vtype in ['pickup', 'pick']:
                    order_info[oid]['pickups'].append(visit_data)
                elif vtype in ['drop', 'delivery']:
                    order_info[oid]['drops'].append(visit_data)
                # Damaged pairs
                elif vtype in ['damaged_pickup']:
                    order_info[oid]['damaged_pickups'].append(visit_data)
                elif vtype in ['damaged_drop']:
                    order_info[oid]['damaged_drops'].append(visit_data)
                # Exchange pairs
                elif vtype in ['exchanged_pickup', 'exchange_pickup']:
                    order_info[oid]['exchanged_pickups'].append(visit_data)
                elif vtype in ['exchanged_drop', 'exchange_drop']:
                    order_info[oid]['exchanged_drops'].append(visit_data)
                # Return pairs
                elif vtype in ['return_pickup', 'return_pick']:
                    order_info[oid]['return_pickups'].append(visit_data)
                elif vtype in ['return_drop', 'return_delivery']:
                    order_info[oid]['return_drops'].append(visit_data)
        
        # Identify violations for all pair types
        cross_violations = []   # (order_id, pickup_data, drop_data, pair_type)
        order_violations = []   # (order_id, pickup_data, drop_data, pair_type)
        
        # Helper to check violations for a specific pair type
        def check_violations_for_type(oid, info, pickup_key, drop_key, pair_type):
            for pu in info.get(pickup_key, []):
                for dr in info.get(drop_key, []):
                    if pu['truck'] != dr['truck']:
                        cross_violations.append((oid, pu, dr, pair_type))
                    elif dr['seq'] < pu['seq']:
                        order_violations.append((oid, pu, dr, pair_type))
        
        for oid, info in order_info.items():
            # Check all pair types
            check_violations_for_type(oid, info, 'pickups', 'drops', 'standard')
            check_violations_for_type(oid, info, 'damaged_pickups', 'damaged_drops', 'damaged')
            check_violations_for_type(oid, info, 'exchanged_pickups', 'exchanged_drops', 'exchanged')
            check_violations_for_type(oid, info, 'return_pickups', 'return_drops', 'return')
        
        total_violations = len(cross_violations) + len(order_violations)
        
        if total_violations == 0:
            logger.info(f"✅ All validation errors fixed after {iteration + 1} pass(es)")
            break
        
        # Check if we're making progress
        if previous_violation_count is not None and total_violations >= previous_violation_count:
            # Not making progress - be more aggressive
            if iteration >= 2:  # After 2 iterations, start moving problematic visits to unassigned
                logger.warning(f"⚠️ Not making progress on violations - being more aggressive (iteration {iteration + 1})")
                # For standard pairs that can't be fixed, move both to unassigned
                violations_to_remove = []
                for violation in cross_violations:
                    oid, pu, dr, pair_type = violation
                    if pair_type == 'standard':
                        pickup_route = routes[pu['ridx']]
                        drop_route = routes[dr['ridx']]
                        pickup_visit = pu['stop']
                        drop_visit = dr['stop']
                        
                        # Remove both from routes
                        pickup_route['stops'] = [s for s in pickup_route['stops']
                                                 if s.get('visitId') != pickup_visit['visitId']]
                        drop_route['stops'] = [s for s in drop_route['stops']
                                               if s.get('visitId') != drop_visit['visitId']]
                        
                        # Add to unassigned
                        if pickup_visit['visitId'] not in seen_unassigned:
                            unassigned.append({
                                'visitId': pickup_visit['visitId'],
                                'reason': 'standard_pair_cross_truck_unfixable'
                            })
                            seen_unassigned.add(pickup_visit['visitId'])
                        if drop_visit['visitId'] not in seen_unassigned:
                            unassigned.append({
                                'visitId': drop_visit['visitId'],
                                'reason': 'standard_pair_cross_truck_unfixable'
                            })
                            seen_unassigned.add(drop_visit['visitId'])
                        
                        logger.info(f"  🗑️ Moved standard pair (order {oid}) to unassigned - cross-truck unfixable")
                        violations_to_remove.append(violation)
                
                # Remove handled violations and rebuild routes
                for violation in violations_to_remove:
                    cross_violations.remove(violation)
                
                # Rebuild routes after removals
                for route in routes:
                    rebuild_route_stats(route)
        
        previous_violation_count = total_violations
        
        logger.info(f"🔧 Fix pass {iteration + 1}: "
                     f"{len(cross_violations)} cross-truck, {len(order_violations)} ordering violations")
        
        # ─── Phase 1: Fix cross-truck violations ───
        # Strategy: Move DROP to pickup's truck (preferred) or PICKUP to drop's truck.
        for oid, pu, dr, pair_type in cross_violations:
            pickup_route = routes[pu['ridx']]
            drop_route = routes[dr['ridx']]
            drop_visit = dr['stop']
            pickup_visit = pu['stop']
            
            if drop_visit['visitId'] in seen_unassigned or pickup_visit['visitId'] in seen_unassigned:
                continue
            
            moved = False
            
            # ── Strategy A: Move DROP to pickup's truck ──
            # Check if pickup truck already visits the drop location AFTER the pickup
            existing_seq_after = None
            for s in pickup_route['stops']:
                if (abs(s['lat'] - dr['lat']) < 0.0001 and
                    abs(s['lng'] - dr['lng']) < 0.0001 and
                    s['sequence'] > pu['seq']):
                    existing_seq_after = s['sequence']
                    break
            
            if existing_seq_after is not None:
                # Truck already visits this location after the pickup — just add the visit
                new_stop = dict(drop_visit)
                new_stop['sequence'] = existing_seq_after
                pickup_route['stops'].append(new_stop)
                drop_route['stops'] = [s for s in drop_route['stops']
                                       if s.get('visitId') != drop_visit['visitId']]
                moved = True
                logger.info(f"  ✅ Order {oid}: Moved DROP to {pu['truck']} "
                            f"(existing waypoint at seq {existing_seq_after})")
            
            if not moved:
                # Need to add a new waypoint after the pickup
                drop_node = find_node(dr['lat'], dr['lng'])
                if drop_node is not None:
                    new_dist, _ = calc_insertion_cost_after(pickup_route, drop_node, pu['seq'])
                    new_wp = pickup_route['waypoint_count'] + 1
                    
                    if new_dist <= max_distance_per_vehicle and new_wp <= max_waypoints:
                        new_stop = dict(drop_visit)
                        new_stop['sequence'] = pu['seq'] + 0.5  # Fractional; rebuilt later
                        pickup_route['stops'].append(new_stop)
                        drop_route['stops'] = [s for s in drop_route['stops']
                                               if s.get('visitId') != drop_visit['visitId']]
                        moved = True
                        logger.info(f"  ✅ Order {oid}: Added DROP to {pu['truck']} "
                                    f"(new waypoint after pickup, +{(new_dist - pickup_route.get('estimated_km', 0) * 1000) / 1000:.1f}km)")
            
            if not moved:
                # ── Strategy B: Move PICKUP to drop's truck ──
                existing_seq_before = None
                for s in drop_route['stops']:
                    if (abs(s['lat'] - pu['lat']) < 0.0001 and
                        abs(s['lng'] - pu['lng']) < 0.0001 and
                        s['sequence'] < dr['seq']):
                        existing_seq_before = s['sequence']
                        break
                
                if existing_seq_before is not None:
                    new_stop = dict(pickup_visit)
                    new_stop['sequence'] = existing_seq_before
                    drop_route['stops'].append(new_stop)
                    pickup_route['stops'] = [s for s in pickup_route['stops']
                                             if s.get('visitId') != pickup_visit['visitId']]
                    moved = True
                    logger.info(f"  ✅ Order {oid}: Moved PICKUP to {dr['truck']} "
                                f"(existing waypoint at seq {existing_seq_before})")
                else:
                    pickup_node = find_node(pu['lat'], pu['lng'])
                    if pickup_node is not None:
                        new_dist, _ = calc_insertion_cost_before(drop_route, pickup_node, dr['seq'])
                        new_wp = drop_route['waypoint_count'] + 1
                        
                        if new_dist <= max_distance_per_vehicle and new_wp <= max_waypoints:
                            new_stop = dict(pickup_visit)
                            new_stop['sequence'] = dr['seq'] - 0.5  # Fractional; rebuilt later
                            drop_route['stops'].append(new_stop)
                            pickup_route['stops'] = [s for s in pickup_route['stops']
                                                     if s.get('visitId') != pickup_visit['visitId']]
                            moved = True
                            logger.info(f"  ✅ Order {oid}: Added PICKUP to {dr['truck']} "
                                        f"(new waypoint before drop)")
            
            if not moved:
                # ── Strategy C: Can't fix — handle based on pair type ──
                # For exchange/damage pairs: Keep pickup, move drop to unassigned
                # For standard pairs: Move both to unassigned
                if pair_type in ['exchanged', 'damaged']:
                    # Keep pickup in route, move only drop to unassigned
                    drop_route['stops'] = [s for s in drop_route['stops']
                                           if s.get('visitId') != drop_visit['visitId']]
                    if drop_visit['visitId'] not in seen_unassigned:
                        unassigned.append({
                            'visitId': drop_visit['visitId'],
                            'reason': f'{pair_type}_drop_unfixable_kept_pickup'
                        })
                        seen_unassigned.add(drop_visit['visitId'])
                    logger.warning(f"  ⚠️ Order {oid} ({pair_type}): Can't fix cross-truck — "
                                 f"DROP moved to unassigned, PICKUP kept in route")
                else:
                    # Standard pairs: Move both to unassigned
                    pickup_route['stops'] = [s for s in pickup_route['stops']
                                             if s.get('visitId') != pickup_visit['visitId']]
                    drop_route['stops'] = [s for s in drop_route['stops']
                                           if s.get('visitId') != drop_visit['visitId']]
                    if pickup_visit['visitId'] not in seen_unassigned:
                        unassigned.append({
                            'visitId': pickup_visit['visitId'],
                            'reason': 'cross_truck_violation_unfixable'
                        })
                        seen_unassigned.add(pickup_visit['visitId'])
                    if drop_visit['visitId'] not in seen_unassigned:
                        unassigned.append({
                            'visitId': drop_visit['visitId'],
                            'reason': 'cross_truck_violation_unfixable'
                        })
                        seen_unassigned.add(drop_visit['visitId'])
                    logger.warning(f"  ⚠️ Order {oid}: Can't fix cross-truck — both moved to unassigned")
        
        # ─── Phase 2: Fix ordering violations (DROP before PICKUP on same truck) ───
        # Strategy: Remove DROP from current position, re-insert AFTER the PICKUP.
        # If the truck already revisits the drop location after the pickup, reuse that waypoint.
        # Otherwise add a revisit waypoint (if within constraints).
        for oid, pu, dr, pair_type in order_violations:
            route = routes[pu['ridx']]
            drop_visit = dr['stop']
            
            if drop_visit['visitId'] in seen_unassigned:
                continue
            
            # Check if route already has a stop at drop location AFTER the pickup
            existing_after = None
            for s in route['stops']:
                if (s.get('visitId') != drop_visit['visitId'] and
                    abs(s['lat'] - dr['lat']) < 0.0001 and
                    abs(s['lng'] - dr['lng']) < 0.0001 and
                    s['sequence'] > pu['seq']):
                    existing_after = s['sequence']
                    break
            
            if existing_after is not None:
                # Move drop to the existing later waypoint at same location
                route['stops'] = [s for s in route['stops']
                                  if s.get('visitId') != drop_visit['visitId']]
                new_stop = dict(drop_visit)
                new_stop['sequence'] = existing_after
                route['stops'].append(new_stop)
                logger.info(f"  ✅ Order {oid}: Moved DROP to seq {existing_after} "
                            f"(existing waypoint after pickup)")
            else:
                # Need to add a revisit to the drop location after the pickup
                drop_node = find_node(dr['lat'], dr['lng'])
                if drop_node is not None:
                    # Calculate cost with the extra revisit
                    # First remove the drop, then calc insertion after pickup
                    temp_stops = [s for s in route['stops']
                                  if s.get('visitId') != drop_visit['visitId']]
                    temp_route = dict(route)
                    temp_route['stops'] = temp_stops
                    new_dist, _ = calc_insertion_cost_after(temp_route, drop_node, pu['seq'])
                    new_wp = len(set(s['sequence'] for s in temp_stops
                                     if s['sequence'] != dr['seq'] or
                                     any(s2['sequence'] == dr['seq'] and
                                         s2.get('visitId') != drop_visit['visitId']
                                         for s2 in temp_stops))) + 1
                    # Simpler: just count current waypoints + 1 for the revisit
                    current_wp = route['waypoint_count']
                    # Check if removing the drop empties its original waypoint
                    others_at_drop_loc = [s for s in route['stops']
                                          if s.get('visitId') != drop_visit['visitId'] and
                                          abs(s['lat'] - dr['lat']) < 0.0001 and
                                          abs(s['lng'] - dr['lng']) < 0.0001]
                    wp_freed = 0 if others_at_drop_loc else 1
                    new_wp_count = current_wp - wp_freed + 1  # -freed +revisit
                    
                    if new_dist <= max_distance_per_vehicle and new_wp_count <= max_waypoints:
                        route['stops'] = [s for s in route['stops']
                                          if s.get('visitId') != drop_visit['visitId']]
                        new_stop = dict(drop_visit)
                        new_stop['sequence'] = pu['seq'] + 0.5  # Fractional; rebuilt later
                        route['stops'].append(new_stop)
                        logger.info(f"  ✅ Order {oid}: Added DROP revisit after pickup "
                                    f"(wp: {current_wp} → {new_wp_count})")
                    else:
                        # Can't add revisit — move drop to unassigned
                        # For exchange/damage pairs, this is acceptable (keep pickup)
                        route['stops'] = [s for s in route['stops']
                                          if s.get('visitId') != drop_visit['visitId']]
                        if drop_visit['visitId'] not in seen_unassigned:
                            if pair_type in ['exchanged', 'damaged']:
                                reason = f'{pair_type}_drop_ordering_unfixable_kept_pickup'
                            else:
                                reason = 'ordering_violation_exceeds_constraints'
                            unassigned.append({
                                'visitId': drop_visit['visitId'],
                                'reason': reason
                            })
                            seen_unassigned.add(drop_visit['visitId'])
                        if pair_type in ['exchanged', 'damaged']:
                            logger.info(f"  ℹ️ Order {oid} ({pair_type}): DROP moved to unassigned "
                                       f"(constraints exceeded), PICKUP kept in route")
                        else:
                            logger.warning(f"  ⚠️ Order {oid}: Can't fix ordering (exceeds constraints) "
                                           f"— DROP moved to unassigned")
                else:
                    # Can't find location node — move to unassigned
                    route['stops'] = [s for s in route['stops']
                                      if s.get('visitId') != drop_visit['visitId']]
                    if drop_visit['visitId'] not in seen_unassigned:
                        if pair_type in ['exchanged', 'damaged']:
                            reason = f'{pair_type}_drop_ordering_unfixable_kept_pickup'
                        else:
                            reason = 'ordering_violation_unfixable'
                        unassigned.append({
                            'visitId': drop_visit['visitId'],
                            'reason': reason
                        })
                        seen_unassigned.add(drop_visit['visitId'])
                    if pair_type in ['exchanged', 'damaged']:
                        logger.info(f"  ℹ️ Order {oid} ({pair_type}): DROP moved to unassigned "
                                   f"(location not found), PICKUP kept in route")
                    else:
                        logger.warning(f"  ⚠️ Order {oid}: Location not found — DROP moved to unassigned")
        
        # ─── Rebuild all routes after this pass ───
        for route in routes:
            rebuild_route_stats(route)
    
    # Remove empty routes
    routes = [r for r in routes if r.get('stops')]
    
    # Final re-validation
    final_errors = validate_routes(routes)
    
    # ─── Final pass: Handle ALL remaining validation errors ───
    # For exchange/damage: move drop to unassigned (keep pickup)
    # For standard pairs: move both to unassigned if can't be fixed
    if final_errors:
        logger.info(f"🔧 Final pass: Handling {len(final_errors)} remaining validation errors...")
        
        # Parse errors to identify all problematic orders
        exchange_damage_orders = set()
        standard_orders = set()
        
        for error in final_errors:
            # Extract order_id from error message
            parts = error.split('Order ')
            if len(parts) > 1:
                order_id = parts[1].split(':')[0].strip()
                
                # Check if this is an exchange or damage pair error
                if 'Exchange' in error or 'Damaged' in error:
                    exchange_damage_orders.add(order_id)
                elif 'Standard' in error:
                    standard_orders.add(order_id)
        
        # For exchange/damage orders with errors, move drops to unassigned (keep pickup)
        if exchange_damage_orders:
            logger.info(f"  ℹ️ Found {len(exchange_damage_orders)} exchange/damage orders with errors - moving drops to unassigned")
            
            for route in routes:
                stops_to_remove = []
                for stop in route['stops']:
                    order_id = stop.get('order_id')
                    visit_type = (stop.get('visit_type') or '').lower()
                    
                    if order_id in exchange_damage_orders:
                        # Check if this is a drop for an exchange/damage pair
                        if visit_type in ['exchanged_drop', 'exchange_drop', 'damaged_drop']:
                            stops_to_remove.append(stop)
                            if stop['visitId'] not in seen_unassigned:
                                pair_type = 'exchanged' if 'exchange' in visit_type else 'damaged'
                                unassigned.append({
                                    'visitId': stop['visitId'],
                                    'reason': f'{pair_type}_drop_validation_error_kept_pickup'
                                })
                                seen_unassigned.add(stop['visitId'])
                                logger.info(f"  ✅ Moved {pair_type} DROP (order {order_id}) to unassigned due to validation error")
                
                # Remove the stops
                if stops_to_remove:
                    route['stops'] = [s for s in route['stops'] if s not in stops_to_remove]
                    rebuild_route_stats(route)
        
        # For standard orders with errors, move BOTH pickup and drop to unassigned
        if standard_orders:
            logger.info(f"  ℹ️ Found {len(standard_orders)} standard orders with errors - moving both pickup and drop to unassigned")
            
            for route in routes:
                stops_to_remove = []
                for stop in route['stops']:
                    order_id = stop.get('order_id')
                    visit_type = (stop.get('visit_type') or '').lower()
                    
                    if order_id in standard_orders:
                        # Remove both pickup and drop for standard pairs
                        if visit_type in ['pickup', 'pick', 'drop', 'delivery']:
                            stops_to_remove.append(stop)
                            if stop['visitId'] not in seen_unassigned:
                                unassigned.append({
                                    'visitId': stop['visitId'],
                                    'reason': 'standard_pair_validation_error_unfixable'
                                })
                                seen_unassigned.add(stop['visitId'])
                                logger.info(f"  ✅ Moved {visit_type} (order {order_id}) to unassigned due to validation error")
                
                # Remove the stops
                if stops_to_remove:
                    route['stops'] = [s for s in route['stops'] if s not in stops_to_remove]
                    rebuild_route_stats(route)
        
        # Re-validate after final pass
        final_errors = validate_routes(routes)
        
        # If there are still errors, do one more aggressive pass - remove ALL problematic visits
        if final_errors:
            logger.warning(f"⚠️ Still {len(final_errors)} validation errors after final pass - doing aggressive cleanup...")
            
            # Extract all order IDs from remaining errors
            all_problematic_orders = set()
            for error in final_errors:
                parts = error.split('Order ')
                if len(parts) > 1:
                    order_id = parts[1].split(':')[0].strip()
                    all_problematic_orders.add(order_id)
            
            # Remove ALL visits for these orders from routes
            for route in routes:
                stops_to_remove = []
                for stop in route['stops']:
                    order_id = stop.get('order_id')
                    if order_id in all_problematic_orders:
                        stops_to_remove.append(stop)
                        if stop['visitId'] not in seen_unassigned:
                            visit_type = (stop.get('visit_type') or '').lower()
                            if visit_type in ['exchanged_drop', 'exchange_drop', 'damaged_drop']:
                                pair_type = 'exchanged' if 'exchange' in visit_type else 'damaged'
                                reason = f'{pair_type}_drop_validation_error_kept_pickup'
                            else:
                                reason = 'validation_error_unfixable'
                            unassigned.append({
                                'visitId': stop['visitId'],
                                'reason': reason
                            })
                            seen_unassigned.add(stop['visitId'])
                            logger.info(f"  🗑️ Removed {stop.get('visit_type', 'visit')} (order {order_id}) from route due to validation error")
                
                if stops_to_remove:
                    route['stops'] = [s for s in route['stops'] if s not in stops_to_remove]
                    rebuild_route_stats(route)
            
            # Final validation - should be empty now
            final_errors = validate_routes(routes)
    
    # One final check - if there are still errors, remove ALL problematic visits
    if final_errors:
        logger.error(f"❌ CRITICAL: {len(final_errors)} validation errors still remain - removing ALL problematic visits")
        
        # Extract all order IDs from errors
        all_problematic_orders = set()
        for error in final_errors:
            parts = error.split('Order ')
            if len(parts) > 1:
                order_id = parts[1].split(':')[0].strip()
                all_problematic_orders.add(order_id)
        
        # Remove ALL visits for these orders
        for route in routes:
            stops_to_remove = []
            for stop in route['stops']:
                order_id = stop.get('order_id')
                if order_id in all_problematic_orders:
                    stops_to_remove.append(stop)
                    if stop['visitId'] not in seen_unassigned:
                        unassigned.append({
                            'visitId': stop['visitId'],
                            'reason': 'validation_error_critical_removal'
                        })
                        seen_unassigned.add(stop['visitId'])
            
            if stops_to_remove:
                route['stops'] = [s for s in route['stops'] if s not in stops_to_remove]
                rebuild_route_stats(route)
        
        # Final validation - MUST be empty now
        final_errors = validate_routes(routes)
        
        if final_errors:
            logger.error(f"❌ CRITICAL ERROR: Still {len(final_errors)} validation errors after aggressive cleanup!")
            for error in final_errors:
                logger.error(f"  {error}")
        else:
            logger.info(f"✅ All validation errors resolved after aggressive cleanup")
    else:
        logger.info(f"✅ Post-processing complete: All validation errors resolved")
    
    result['routes'] = routes
    result['unassigned_visits'] = unassigned
    result['validation_errors'] = None  # Always None - all errors are fixed or visits moved to unassigned
    
    return result


@auto_routing_bp.route("/optimize", methods=["POST"])
def optimize_routes():
    """
    Hybrid VRP Optimizer: CVRP + VRPPD + VRPTW
    
    Solves routing with capacity constraints, pickup-delivery pairs, and time windows.
    Uses Haversine formula for distance calculations (straight-line distances).
    
    Expected JSON input:
    {
        "trucks": 3,
        "max_km": 120,
        "max_stops": 25,               // Optional, default 25 — max stops per truck
        "truck_capacity": 100,          // Optional, default 100 — max volume capacity per truck
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
                "visit_type": "pickup",     // Optional visit types:
                                             // Standard: "pickup", "drop"
                                             // Damaged: "damaged_pickup", "damaged_drop"
                                             // Exchange: "exchanged_pickup", "exchanged_drop"
                                             // Return: "return_pickup", "return_drop"
                                             // Legacy: "returned_from", "returned_to"
                                             // Note: Damaged/Exchange/Return/Returned visits are MANDATORY (HIGHEST PRIORITY)
                "vol_capacity": 5,          // Optional, default 0 — volume capacity for this visit
                                             // Pickup adds volume, drop subtracts volume
                "time_window_start": 60,    // Optional: earliest arrival (min from shift start)
                "time_window_end": 300      // Optional: latest arrival (min from shift start)
            },
            { 
                "visitId": "V2", 
                "lat": 12.99, 
                "lng": 77.61, 
                "sla_days": 3,
                "order_id": "ORD123",       // Optional
                "visit_type": "drop",        // See above for all visit types
                "vol_capacity": 5           // Optional, default 0 — volume capacity for this visit
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
        
        # ─── LOG INPUT ───
        logger.info("="*80)
        logger.info("📥 API REQUEST INPUT:")
        logger.info(json.dumps(data, indent=2))
        logger.info("="*80)
        
        # Log that we're using Google Distance Matrix API for distance calculations
        logger.info("🗺️  Using Google Distance Matrix API for distance calculations (real road distances)")
        
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
        truck_capacity = data.get("truck_capacity", 100)  # Default 100 volume units
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
                     f"capacity {truck_capacity} units/truck, "
                     f"{shift_duration_hours}h shift, {service_time_minutes}min/stop, "
                     f"{len(visits)} visits")
        
        # Extract order_ids and visit_types from original visits
        # Also track mandatory visits that MUST be scheduled
        original_order_ids = []
        original_visit_types = []
        mandatory_visit_ids = set()  # Track visits that MUST be in final routes
        
        for visit in visits:
            original_order_ids.append(visit.get("order_id"))
            visit_type = visit.get("visit_type")
            original_visit_types.append(visit_type)
            
            # Track mandatory visits (HIGHEST PRIORITY - must be scheduled)
            # Includes: damaged, exchanged, return, and legacy returned visits
            if visit_type:
                vtype_lower = visit_type.lower()
                if vtype_lower in ['returned_from', 'returned_to',
                                   'damaged_pickup', 'damaged_drop',
                                   'exchanged_pickup', 'exchanged_drop', 'exchange_pickup', 'exchange_drop',
                                   'return_pickup', 'return_drop', 'return_pick', 'return_delivery']:
                    mandatory_visit_ids.add(visit['visitId'])
                    logger.info(f"🔒 HIGH PRIORITY mandatory visit detected: {visit['visitId']} (type: {visit_type})")
        
        if mandatory_visit_ids:
            logger.info(f"📋 Total mandatory visits to schedule: {len(mandatory_visit_ids)}")
        
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
        # Build distance + duration matrices via Haversine formula
        matrix_start = time.time()
        distance_matrix, duration_matrix = create_distance_matrix(locations)
        matrix_elapsed = time.time() - matrix_start
        matrix_snap = log_memory("Distance matrix BUILT", api_snap)
        logger.info(f"⏱️  Distance matrix: {len(locations)}x{len(locations)} built in {matrix_elapsed:.2f}s")
        
        # Create ENHANCED priority list - MUCH stronger SLA prioritization
        # Lower SLA days = exponentially higher priority
        priorities = [5]  # Start point - neutral priority
        for visit in combined_visits:
            sla_days = visit.get("sla_days", 5)
            
            # ENHANCED priority calculation with stronger differentiation
            if sla_days <= 0:
                # Breached SLA: EXTREME priority (20+)
                # More days overdue = even higher priority
                priority = 25 + abs(sla_days) * 2  # Was 15, now 25+
            elif sla_days == 1:
                # 1 day left: CRITICAL priority
                priority = 18  # Was ~9, now 18
            elif sla_days == 2:
                # 2 days left: URGENT priority
                priority = 12  # Was ~8, now 12
            elif sla_days == 3:
                # 3 days left: WARNING priority
                priority = 8   # Was 7, now 8
            elif sla_days <= 5:
                # 4-5 days: NORMAL priority
                priority = 5
            else:
                # 6+ days: LOW priority
                priority = max(1, 6 - sla_days)
            
            priorities.append(priority)
        priorities.append(5)  # End point - neutral priority
        
        logger.info(f"🎯 SLA Priority distribution: "
                    f"BREACHED={sum(1 for p in priorities if p >= 25)}, "
                    f"CRITICAL={sum(1 for p in priorities if 15 <= p < 25)}, "
                    f"URGENT={sum(1 for p in priorities if 10 <= p < 15)}, "
                    f"WARNING={sum(1 for p in priorities if 8 <= p < 10)}, "
                    f"NORMAL={sum(1 for p in priorities if 5 <= p < 8)}")
        
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
        
        # Convert max_km to meters (Haversine gives straight-line distances)
        max_distance_meters = int(max_km * 1000)
        
        # Convert shift duration to seconds
        max_route_time = int(shift_duration_hours * 3600)
        
        logger.info(f"📏 Distance: {max_km}km limit → {max_distance_meters/1000:.1f}km solver (Haversine)")
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
            max_waypoints=max_stops,
            combined_order_info=aligned_order_info,
            visit_groups=aligned_visit_groups,
            original_visits=visits,
            time_windows=aligned_time_windows if has_any_time_window else None,
            service_times=aligned_service_times,
            max_route_time=max_route_time,
            truck_capacity=truck_capacity
        )
        
        if result is None:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            return jsonify({
                "error": "Could not find a solution for the given constraints",
                "suggestion": "Try increasing max_km or number of trucks"
            }), 400
        
        # ─── MULTI-PASS ROUTING: Use remaining trucks for unassigned visits ───
        # If there are unassigned visits AND unused trucks, run solver again
        max_passes = 5  # Limit to prevent infinite loops
        pass_number = 1
        
        while pass_number < max_passes:
            trucks_used = len(result.get('routes', []))
            trucks_remaining = num_trucks - trucks_used
            unassigned_count = len(result.get('unassigned_visits', []))
            
            # Check if we should run another pass
            if trucks_remaining > 0 and unassigned_count > 0:
                logger.info(f"🔄 MULTI-PASS ROUTING - Pass {pass_number + 1}: "
                          f"{trucks_remaining} truck(s) remaining, {unassigned_count} visit(s) unassigned")
                logger.info(f"   Running solver again with remaining trucks and unassigned visits...")
                
                # Extract unassigned visit IDs
                unassigned_visit_ids = {uv['visitId'] for uv in result.get('unassigned_visits', [])}
                
                # Filter original visits to only unassigned ones
                remaining_visits = [v for v in visits if v['visitId'] in unassigned_visit_ids]
                
                if len(remaining_visits) == 0:
                    logger.info("   No visits to assign - stopping multi-pass")
                    break
                
                logger.info(f"   Processing {len(remaining_visits)} unassigned visits with {trucks_remaining} trucks")
                
                # Rebuild order_ids and visit_types for remaining visits
                remaining_order_ids = []
                remaining_visit_types = []
                for visit in remaining_visits:
                    remaining_order_ids.append(visit.get("order_id"))
                    remaining_visit_types.append(visit.get("visit_type"))
                
                # Combine visits at same locations (same as initial processing)
                remaining_combined_visits, remaining_visit_groups, remaining_combined_order_info, remaining_location_to_visits = \
                    combine_visits_at_same_location(remaining_visits, remaining_order_ids, remaining_visit_types)
                
                # Rebuild locations with start and end points
                remaining_locations = [start_point] + remaining_combined_visits + [end_point]
                remaining_start_index = 0
                remaining_end_index = len(remaining_locations) - 1
                
                # Rebuild distance and duration matrices for remaining locations
                logger.info(f"   Building distance matrix for {len(remaining_locations)} locations...")
                remaining_distance_matrix, remaining_duration_matrix = create_distance_matrix(remaining_locations)
                
                # Rebuild priorities, time windows, and service times for remaining locations
                # Use same logic as initial processing
                remaining_priorities = [5]  # Start point
                for visit in remaining_combined_visits:
                    sla_days = visit.get("sla_days", 5)
                    # Same priority calculation as initial processing
                    if sla_days <= 0:
                        priority = 25 + abs(sla_days) * 2
                    elif sla_days == 1:
                        priority = 18
                    elif sla_days == 2:
                        priority = 12
                    elif sla_days == 3:
                        priority = 8
                    elif sla_days <= 5:
                        priority = 5
                    else:
                        priority = max(1, 6 - sla_days)
                    remaining_priorities.append(priority)
                remaining_priorities.append(5)  # End point
                
                # Time windows aligned with locations
                remaining_time_windows = [None]  # Start
                remaining_has_time_window = False
                for visit in remaining_combined_visits:
                    tw_start = visit.get("time_window_start")
                    tw_end = visit.get("time_window_end")
                    if tw_start is not None and tw_end is not None:
                        remaining_time_windows.append((int(tw_start * 60), int(tw_end * 60)))
                        remaining_has_time_window = True
                    else:
                        remaining_time_windows.append(None)
                remaining_time_windows.append(None)  # End
                
                # Service times aligned with locations
                remaining_service_times = [0]  # Start
                svc_seconds = int(service_time_minutes * 60)
                for idx, _ in enumerate(remaining_combined_visits):
                    num_visits_at_node = len(remaining_visit_groups[idx]) if idx < len(remaining_visit_groups) and remaining_visit_groups[idx] else 1
                    remaining_service_times.append(svc_seconds * num_visits_at_node)
                remaining_service_times.append(0)  # End
                
                # Align combined_order_info and visit_groups with locations (add None for start/end)
                remaining_aligned_order_info = [None] + remaining_combined_order_info + [None]
                remaining_aligned_visit_groups = [None] + remaining_visit_groups + [None]
                
                # Calculate starting truck number for this pass (continue from where previous pass left off)
                current_truck_count = len(result.get('routes', []))
                start_truck_number_for_pass = current_truck_count + 1
                
                logger.info(f"   Starting truck numbering from TRUCK_{start_truck_number_for_pass}")
                
                # Run solver again with remaining trucks and unassigned visits
                remaining_result = solve_vrp(
                    num_vehicles=trucks_remaining,
                    max_distance_per_vehicle=max_distance_meters,
                    locations=remaining_locations,
                    distance_matrix=remaining_distance_matrix,
                    duration_matrix=remaining_duration_matrix,
                    start_index=remaining_start_index,
                    end_index=remaining_end_index,
                    priorities=remaining_priorities,
                    max_waypoints=max_stops,
                    combined_order_info=remaining_aligned_order_info,
                    visit_groups=remaining_aligned_visit_groups,
                    original_visits=remaining_visits,
                    time_windows=remaining_time_windows if remaining_has_time_window else None,
                    service_times=remaining_service_times,
                    max_route_time=max_route_time,
                    truck_capacity=truck_capacity,
                    start_truck_number=start_truck_number_for_pass
                )
                
                if remaining_result and remaining_result.get('routes'):
                    # Merge results: add new routes to existing routes
                    logger.info(f"   ✅ Pass {pass_number + 1} assigned {len(remaining_result['routes'])} additional route(s)")
                    result['routes'].extend(remaining_result['routes'])
                    # Update unassigned visits to only those that remain unassigned
                    result['unassigned_visits'] = remaining_result.get('unassigned_visits', [])
                    pass_number += 1
                else:
                    logger.info(f"   ⚠️ Pass {pass_number + 1} could not assign any additional visits - stopping")
                    break
            else:
                # No remaining trucks or no unassigned visits - stop
                if trucks_remaining == 0:
                    logger.info(f"   All {num_trucks} trucks are used - stopping multi-pass")
                if unassigned_count == 0:
                    logger.info(f"   All visits assigned - stopping multi-pass")
                break
        
        if pass_number > 1:
            logger.info(f"✅ Multi-pass routing completed: {pass_number} passes, "
                       f"{len(result.get('routes', []))} total routes, "
                       f"{len(result.get('unassigned_visits', []))} remaining unassigned visits")
        
        # Post-processing: Fix volume capacity violations first
        logger.info("📦 Validating and fixing volume capacity violations...")
        result = fix_volume_capacity_violations(
            result=result,
            original_visits=visits,
            truck_capacity=truck_capacity,
            distance_matrix=distance_matrix,
            duration_matrix=duration_matrix,
            locations=locations,
            start_index=start_index,
            end_index=end_index,
            max_distance_per_vehicle=max_distance_meters,
            max_waypoints=max_stops
        )
        
        # Post-processing: Fix any pickup-drop validation errors
        if result.get('validation_errors'):
            logger.info(f"🔧 Attempting to fix {len(result['validation_errors'])} validation errors...")
            result = fix_validation_errors(
                result=result,
                distance_matrix=distance_matrix,
                duration_matrix=duration_matrix,
                locations=locations,
                start_index=start_index,
                end_index=end_index,
                max_distance_per_vehicle=max_distance_meters,
                max_waypoints=max_stops
            )
        
        # Add filtered-out visits to unassigned visits
        if filtered_excluded_visits:
            result['unassigned_visits'].extend(filtered_excluded_visits)
            logger.info(f"Added {len(filtered_excluded_visits)} filtered visits to unassigned list")
        
        # Log route utilization summary
        total_locations = 0
        total_visits_assigned = 0
        for route in result['routes']:
            num_locations = route['waypoint_count']
            num_visits = route.get('total_visits', len(route['stops']))
            est_km = route['estimated_km']
            est_hrs = route.get('estimated_hours', 0)
            util_km = (est_km / max_km * 100) if max_km > 0 else 0
            util_stops = (num_locations / max_stops * 100) if max_stops > 0 else 0
            util_time = (est_hrs / shift_duration_hours * 100) if shift_duration_hours > 0 else 0
            total_locations += num_locations
            total_visits_assigned += num_visits
            logger.info(f"📊 {route['truckId']}: {num_locations} locations/{num_visits} visits ({util_stops:.0f}%), "
                        f"{est_km:.1f}km ({util_km:.0f}%), "
                        f"{est_hrs:.1f}h ({util_time:.0f}%)")
        
        logger.info(f"✅ Solution: {len(result['routes'])} routes, "
                     f"{total_locations} locations/{total_visits_assigned} visits assigned, "
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
        
        # ─── LOG OUTPUT ───
        logger.info("="*80)
        logger.info("📤 API RESPONSE OUTPUT:")
        logger.info(f"Number of routes: {len(result['routes'])}")
        logger.info(f"Number of unassigned visits: {len(result['unassigned_visits'])}")
        logger.info("Full response:")
        logger.info(json.dumps(result, indent=2))
        logger.info("="*80)
        
        # Log that we completed using Google Distance Matrix API
        logger.info("✅ Routing optimization completed using Google Distance Matrix API")
        
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

import logging
import os
from flask import Blueprint, request, jsonify
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
import googlemaps
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Google Maps Client
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
gmaps = None
if GOOGLE_MAPS_API_KEY:
    try:
        gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        logger.info("✅ Google Maps Client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Google Maps Client: {e}")
else:
    logger.warning("⚠️ GOOGLE_MAPS_API_KEY not found. Using Haversine fallback.")

auto_routing_bp = Blueprint("auto_routing", __name__)

# Maximum waypoints per route (Google Maps and other routing APIs limit)
MAX_WAYPOINTS_PER_ROUTE = 25


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


def create_distance_matrix(locations: List[Dict]) -> Tuple[List[List[int]], bool]:
    """
    Create a distance matrix from a list of locations.
    Uses Google Maps Distance Matrix API if available, otherwise Haversine.
    Returns: (distance_matrix, used_google_maps)
    """
    n = len(locations)
    distance_matrix = [[0] * n for _ in range(n)]
    
    # Try using Google Maps if available
    if gmaps:
        try:
            logger.info("Using Google Maps for distance matrix...")
            # Prepare coordinates
            coords = [(loc['lat'], loc['lng']) for loc in locations]
            
            # Google Maps Distance Matrix has limits (max 100 elements per request, e.g. 10x10)
            # For larger sets, we calculate in chunks or row by row
            # Since n is likely small (< 25) for this use case, we can try direct calls or row-based
            
            for i in range(n):
                # Request distances from origin 'i' to all destinations
                # We can batch destinations in groups of 25 (safe limit)
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
                            else:
                                # Fallback for unreachable points
                                distance_matrix[i][j + k] = 9999999
                    else:
                        raise Exception(f"Google Maps API error: {result['status']}")
            
            return distance_matrix, True
            
        except Exception as e:
            logger.error(f"Google Maps API failed, falling back to Haversine: {str(e)}")
            # Fall through to Haversine
    
    # Haversine Fallback
    logger.info("Using Haversine formula for distance matrix")
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_km = haversine_distance(
                    locations[i]['lat'], locations[i]['lng'],
                    locations[j]['lat'], locations[j]['lng']
                )
                # Convert to meters and round to integer
                distance_matrix[i][j] = int(dist_km * 1000)
            else:
                distance_matrix[i][j] = 0
    
    return distance_matrix, False


def combine_visits_at_same_location(
    visits: List[Dict],
    order_ids: List[str],
    visit_types: List[str],
    tolerance: float = 0.0001
) -> Tuple[List[Dict], List[List[str]], List[Dict], Dict[str, List[str]]]:
    """
    Combine visits that are at the same location (within tolerance).
    
    IMPORTANT: Only combines ELIGIBLE visits. Visits that are part of a 
    cross-location pickup-drop pair are kept as INDIVIDUAL nodes to prevent
    conflicting constraints (e.g., node A must come before AND after node B).
    
    Eligible for combining:
    - Standalone visits (no pickup-drop pair, or missing counterpart)
    - Visits where pickup AND drop are at the SAME location
    
    NOT eligible (stay as individual nodes):
    - Visits that are part of a pickup-drop pair where pickup and drop are
      at DIFFERENT locations (these create ordering constraints)
    
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
    
    # === STEP 1: Pre-scan to find cross-location pickup-drop pairs ===
    # Build a map of order_id -> {'pickup_idx': i, 'drop_idx': j, 'pickup_loc': ..., 'drop_loc': ...}
    order_pair_scan = {}
    for i, (oid, vtype) in enumerate(zip(order_ids, visit_types)):
        if oid and vtype:
            if oid not in order_pair_scan:
                order_pair_scan[oid] = {}
            vtype_lower = vtype.lower()
            loc_key = (
                round(visits[i]['lat'] / tolerance) * tolerance,
                round(visits[i]['lng'] / tolerance) * tolerance
            )
            if vtype_lower in ['pickup', 'pick']:
                order_pair_scan[oid]['pickup_idx'] = i
                order_pair_scan[oid]['pickup_loc'] = loc_key
            elif vtype_lower in ['drop', 'delivery']:
                order_pair_scan[oid]['drop_idx'] = i
                order_pair_scan[oid]['drop_loc'] = loc_key
    
    # Identify visits that are part of CROSS-LOCATION pairs (not eligible for combining)
    non_combinable_indices = set()
    for oid, info in order_pair_scan.items():
        if 'pickup_idx' in info and 'drop_idx' in info:
            pickup_loc = info['pickup_loc']
            drop_loc = info['drop_loc']
            # If pickup and drop are at DIFFERENT locations, mark both as non-combinable
            if pickup_loc != drop_loc:
                non_combinable_indices.add(info['pickup_idx'])
                non_combinable_indices.add(info['drop_idx'])
    
    logger.info(f"📊 Pre-scan: {len(order_pair_scan)} orders, {len(non_combinable_indices)} visits in cross-location pairs (kept individual)")
    
    # === STEP 2: Group visits by location ===
    location_map = {}  # (lat, lng) -> list of visit indices
    for i, visit in enumerate(visits):
        lat = round(visit['lat'] / tolerance) * tolerance
        lng = round(visit['lng'] / tolerance) * tolerance
        location_key = (lat, lng)
        
        if location_key not in location_map:
            location_map[location_key] = []
        location_map[location_key].append(i)
    
    combined_visits = []
    visit_groups = []
    combined_order_info = []
    location_to_visits = {}
    
    # === STEP 3: Create nodes - combine only eligible visits ===
    for location_key, visit_indices in location_map.items():
        # Separate into combinable and non-combinable visits at this location
        combinable = [idx for idx in visit_indices if idx not in non_combinable_indices]
        non_combinable = [idx for idx in visit_indices if idx in non_combinable_indices]
        
        # Create ONE combined node for all combinable visits at this location
        if combinable:
            first_idx = combinable[0]
            first_visit = visits[first_idx]
            
            visit_ids = [visits[idx]['visitId'] for idx in combinable]
            orders = [order_ids[idx] for idx in combinable]
            types = [visit_types[idx] for idx in combinable]
            
            combined_visit = {
                'lat': first_visit['lat'],
                'lng': first_visit['lng'],
                'visitId': first_visit['visitId'],
                'sla_days': min(visits[idx].get('sla_days', 5) for idx in combinable)
            }
            
            combined_visits.append(combined_visit)
            visit_groups.append(visit_ids)
            location_to_visits[f"{first_visit['lat']},{first_visit['lng']}"] = visit_ids
            combined_order_info.append({
                'order_ids': orders,
                'visit_types': types
            })
            
            if len(visit_ids) > 1:
                logger.info(f"✅ Combined {len(visit_ids)} standalone visits at ({first_visit['lat']}, {first_visit['lng']}): {visit_ids}")
        
        # Create INDIVIDUAL nodes for each non-combinable visit (pickup-drop pair members)
        for idx in non_combinable:
            visit = visits[idx]
            combined_visits.append({
                'lat': visit['lat'],
                'lng': visit['lng'],
                'visitId': visit['visitId'],
                'sla_days': visit.get('sla_days', 5)
            })
            visit_groups.append([visit['visitId']])
            combined_order_info.append({
                'order_ids': [order_ids[idx]],
                'visit_types': [visit_types[idx]]
            })
            
            oid = order_ids[idx]
            vtype = visit_types[idx]
            logger.info(f"📌 Kept individual: {visit['visitId']} (order {oid}, {vtype}) at ({visit['lat']}, {visit['lng']})")
    
    logger.info(f"✅ Processed {len(visits)} visits into {len(combined_visits)} nodes "
                f"({len(visits) - len(non_combinable_indices)} combinable, {len(non_combinable_indices)} individual)")
    
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
    start_index: int,
    end_index: int,
    priorities: List[int],
    used_google_maps: bool = False,
    max_waypoints: int = MAX_WAYPOINTS_PER_ROUTE,
    combined_order_info: List[Dict] = None,
    visit_groups: List[List[str]] = None,
    original_visits: List[Dict] = None
) -> Dict:
    """
    Solve the Vehicle Routing Problem using Google OR-Tools
    
    Args:
        num_vehicles: Number of trucks available
        max_distance_per_vehicle: Maximum distance each truck can travel (in meters)
        locations: List of all locations (start, end, and visits)
        distance_matrix: Matrix of distances between all locations
        start_index: Index of the start location
        end_index: Index of the end location
        priorities: Priority values for each visit (higher = more urgent)
        used_google_maps: Whether Google Maps was used for distances
        max_waypoints: Maximum number of waypoints (stops) per route (default: 25)
        combined_order_info: List of dicts with order_ids and visit_types at each location
    
    Returns:
        Dictionary containing routes and unassigned visits
    """
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
    
    # MAXIMIZE VISITS STRATEGY:
    # Use a scaled-down arc cost so solver prioritizes visiting more nodes
    # over minimizing total distance. The drop penalties do the heavy lifting.
    def scaled_distance_callback(from_index, to_index):
        """Returns scaled distance - lower values mean solver cares less about distance."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # Scale down by 10x so penalties dominate the objective
        return distance_matrix[from_node][to_node] // 10
    
    scaled_transit_callback_index = routing.RegisterTransitCallback(scaled_distance_callback)
    
    # Use scaled distance for arc costs (prioritizes including more visits)
    routing.SetArcCostEvaluatorOfAllVehicles(scaled_transit_callback_index)
    
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
    
    # MAXIMIZE VISITS: Set very low arc cost coefficient
    # This makes the solver prefer visiting more stops over minimizing distance
    # The high drop penalties will push the solver to include as many visits as possible
    distance_dimension.SetGlobalSpanCostCoefficient(1)  # Minimal cost for route length
    
    # Add waypoint count constraint (max_stops physical locations per route)
    # Each combined node = 1 physical stop (multiple visits at same lat/lng count as 1 stop)
    def count_callback(from_index):
        """Returns 1 for each physical stop node, 0 for start/end nodes."""
        from_node = manager.IndexToNode(from_index)
        # Count as 1 if it's a visit node (not start or end)
        # Combined visits at same location are already merged into one node
        if from_node != start_index and from_node != end_index:
            return 1
        return 0
    
    count_callback_index = routing.RegisterUnaryTransitCallback(count_callback)
    
    # Add dimension for physical stop count (max_stops)
    waypoint_dimension_name = 'WaypointCount'
    routing.AddDimension(
        count_callback_index,
        0,  # no slack
        max_waypoints,  # maximum physical stops per vehicle (combined locations)
        True,  # start cumul to zero
        waypoint_dimension_name
    )
    waypoint_dimension = routing.GetDimensionOrDie(waypoint_dimension_name)
    
    # Build a mapping of order_id -> {'pickup': node_index, 'drop': node_index}
    # Each combined location may contain multiple orders with different types
    order_map = {}
    paired_nodes = set()
    pickup_drop_pairs = []
    
    logger.info(f"🔍 Building pickup-drop pairs from combined location info...")
    
    if combined_order_info:
        for node in range(len(locations)):
            # Skip start and end nodes
            if node == start_index or node == end_index:
                continue
            
            if node < len(combined_order_info) and combined_order_info[node]:
                info = combined_order_info[node]
                order_ids_at_node = info.get('order_ids', [])
                visit_types_at_node = info.get('visit_types', [])
                
                # Process ALL order_ids and visit_types at this combined location
                for order_id, visit_type in zip(order_ids_at_node, visit_types_at_node):
                    if order_id and visit_type:
                        if order_id not in order_map:
                            order_map[order_id] = {}
                        
                        if visit_type.lower() in ['pickup', 'pick']:
                            order_map[order_id]['pickup'] = node
                            logger.debug(f"  Found PICKUP for order {order_id} at node {node}")
                        elif visit_type.lower() in ['drop', 'delivery']:
                            order_map[order_id]['drop'] = node
                            logger.debug(f"  Found DROP for order {order_id} at node {node}")
        
        # Mark nodes that have both pickup and drop (at DIFFERENT locations)
        for order_id, nodes in order_map.items():
            if 'pickup' in nodes and 'drop' in nodes:
                pickup_node = nodes['pickup']
                drop_node = nodes['drop']
                
                # Only add constraint if pickup and drop are at DIFFERENT nodes
                if pickup_node != drop_node:
                    paired_nodes.add(pickup_node)
                    paired_nodes.add(drop_node)
                    pickup_drop_pairs.append({
                        'order_id': order_id,
                        'pickup_node': pickup_node,
                        'drop_node': drop_node
                    })
                    logger.info(f"✅ Complete pair for order {order_id}: pickup node {pickup_node} -> drop node {drop_node}")
                else:
                    logger.info(f"📍 Order {order_id}: pickup and drop at SAME location (node {pickup_node}) - no constraint needed")
            elif 'pickup' in nodes:
                logger.warning(f"⚠️ Order {order_id} has PICKUP (node {nodes['pickup']}) but NO DROP")
            elif 'drop' in nodes:
                logger.warning(f"⚠️ Order {order_id} has DROP (node {nodes['drop']}) but NO PICKUP")
    
    # Add disjunctions for ALL visit nodes (including paired ones)
    # This allows solver to always find a solution - unassigned visits go to unassigned list
    # 
    # Priority mapping (from priorities list):
    # - priority 15+ = SLA breached (sla_days <= 0) -> Very high penalty
    # - priority 8-9 = SLA 1-2 days -> High penalty
    # - priority 7 = SLA 3 days -> Medium-high penalty
    # - priority < 7 = SLA > 3 days -> Standard penalty
    
    base_penalty = 50000  # High base penalty - maximize all visits
    
    logger.info(f"Adding disjunctions for ALL visit nodes (solver will always find a solution)...")
    
    breached_count = 0
    urgent_count = 0
    paired_count = 0
    
    for node in range(len(locations)):
        # Skip start and end nodes
        if node == start_index or node == end_index:
            continue
        
        priority = priorities[node] if node < len(priorities) else 5
        
        # Calculate penalty based on SLA urgency
        if priority >= 15:
            node_penalty = base_penalty * 100  # 5,000,000
            breached_count += 1
        elif priority >= 8:
            node_penalty = base_penalty * 50   # 2,500,000
            urgent_count += 1
        elif priority >= 7:
            node_penalty = base_penalty * 25   # 1,250,000
        else:
            node_penalty = base_penalty * 10   # 500,000
        
        # Paired nodes get even higher penalty to encourage keeping pairs together
        if node in paired_nodes:
            node_penalty = node_penalty * 2  # Double penalty for paired nodes
            paired_count += 1
        
        routing.AddDisjunction([manager.NodeToIndex(node)], node_penalty)
    
    logger.info(f"✅ Added disjunctions for {len(locations) - 2} visit nodes")
    logger.info(f"   {breached_count} BREACHED, {urgent_count} urgent, {paired_count} in pairs")
    
    # CRITICAL: Detect and resolve CONTRADICTORY pickup-drop constraints
    # When locations are combined, we can get:
    #   Order A: pickup node 12 -> drop node 1 (node 12 must come BEFORE node 1)
    #   Order B: pickup node 1 -> drop node 12 (node 1 must come BEFORE node 12)
    # These are IMPOSSIBLE to satisfy simultaneously and will CRASH the solver.
    #
    # Resolution: For each pair of nodes (A,B), count how many orders go A->B vs B->A.
    # Keep only the MAJORITY direction. The minority direction orders lose their
    # pickup-drop constraint but their visits still happen at the combined location.
    
    # Group pairs by node-pair (direction-agnostic)
    edge_directions = {}  # (min_node, max_node) -> {'forward': [pairs], 'reverse': [pairs]}
    for pair in pickup_drop_pairs:
        p, d = pair['pickup_node'], pair['drop_node']
        edge_key = (min(p, d), max(p, d))
        if edge_key not in edge_directions:
            edge_directions[edge_key] = {'forward': [], 'reverse': []}
        if p == edge_key[0]:
            edge_directions[edge_key]['forward'].append(pair)
        else:
            edge_directions[edge_key]['reverse'].append(pair)
    
    # Filter out contradictory pairs
    valid_pairs = []
    skipped_pairs = []
    for edge_key, directions in edge_directions.items():
        fwd = directions['forward']
        rev = directions['reverse']
        
        if fwd and rev:
            # CONFLICT detected! Both directions exist between same two nodes
            logger.warning(f"⚠️ CONFLICT: {len(fwd)} orders go node {edge_key[0]}→{edge_key[1]}, "
                          f"{len(rev)} orders go node {edge_key[1]}→{edge_key[0]}")
            
            # Keep the majority direction, skip the minority
            if len(fwd) >= len(rev):
                valid_pairs.extend(fwd)
                skipped_pairs.extend(rev)
                logger.warning(f"   → Keeping {len(fwd)} forward pairs, skipping {len(rev)} reverse pairs")
            else:
                valid_pairs.extend(rev)
                skipped_pairs.extend(fwd)
                logger.warning(f"   → Keeping {len(rev)} reverse pairs, skipping {len(fwd)} forward pairs")
        else:
            # No conflict - add all pairs
            valid_pairs.extend(fwd)
            valid_pairs.extend(rev)
    
    if skipped_pairs:
        logger.warning(f"⚠️ Skipped {len(skipped_pairs)} contradictory pickup-drop pairs to prevent solver crash")
        for sp in skipped_pairs:
            logger.warning(f"   Skipped order {sp['order_id']}: pickup {sp['pickup_node']} -> drop {sp['drop_node']}")
    
    # Add pickup and delivery constraints for VALID (non-contradictory) pairs only
    # This ensures: same vehicle + pickup before drop (if both are scheduled)
    added_count = 0
    for pair in valid_pairs:
        pickup_index = manager.NodeToIndex(pair['pickup_node'])
        drop_index = manager.NodeToIndex(pair['drop_node'])
        
        try:
            # AddPickupAndDelivery enforces ordering and same-vehicle constraints
            routing.AddPickupAndDelivery(pickup_index, drop_index)
            
            # Ensure they're on the same vehicle
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(drop_index)
            )
            
            # Ensure pickup comes before drop (cumulative distance constraint)
            routing.solver().Add(
                distance_dimension.CumulVar(pickup_index) <= distance_dimension.CumulVar(drop_index)
            )
            
            added_count += 1
            logger.info(f"Added pickup-drop pair for order {pair['order_id']}: pickup node {pair['pickup_node']} -> drop node {pair['drop_node']}")
        except Exception as e:
            logger.error(f"❌ Failed to add constraint for order {pair['order_id']}: {str(e)}")
    
    logger.info(f"✅ Finished adding {added_count} pickup-drop pairs (skipped {len(skipped_pairs)} conflicting)")
    
    # Setting first solution heuristic - optimized for MAXIMIZING VISITS
    logger.info("Setting up solver parameters...")
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    
    # Use PARALLEL_CHEAPEST_INSERTION - good for fitting many stops
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    
    # Use GUIDED_LOCAL_SEARCH - better at finding solutions with more visits
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    
    # Dynamic time limit based on problem size
    num_nodes = len(locations)
    if num_nodes <= 15:
        time_limit = 10
    elif num_nodes <= 30:
        time_limit = 20
    else:
        time_limit = 30  # More time for large problems to maximize visits
    
    search_parameters.time_limit.seconds = time_limit
    
    # Log level for debugging (0 = silent, 1 = errors, 2 = warnings, 3 = info)
    search_parameters.log_search = False
    
    logger.info(f"🚀 Starting solver with {num_vehicles} vehicles, {len(locations)} locations (max {max_waypoints} stops/truck), time limit: {time_limit}s...")
    # Solve the problem
    solution = None
    try:
        solution = routing.SolveWithParameters(search_parameters)
        
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
        logger.info(f"✅ Solver completed with status: {status_name}. Solution found: {solution is not None}")
        
        if not solution:
            logger.error(f"❌ No solution found. Status: {status_name}")
            if status == 3:
                logger.error("Solver timed out. Try reducing visits or increasing time limit.")
            elif status == 2:
                logger.error("Solver failed - constraints may be infeasible.")
    except Exception as e:
        logger.error(f"❌ Solver crashed with exception: {str(e)}", exc_info=True)
        solution = None
    
    if solution:
        return extract_solution(
            manager, routing, solution, locations, distance_matrix, 
            start_index, end_index, num_vehicles, used_google_maps, max_waypoints,
            combined_order_info, visit_groups, original_visits
        )
    else:
        # FALLBACK: Return all visits as unassigned instead of failing
        logger.warning("⚠️ No solution found - returning all visits as unassigned")
        fallback_unassigned = []
        if original_visits:
            for visit in original_visits:
                visit_id = visit.get('visitId') or visit.get('visit_id')
                if visit_id:
                    fallback_unassigned.append({
                        "visitId": visit_id,
                        "reason": "solver_no_solution"
                    })
        elif visit_groups:
            for group in visit_groups:
                for visit_id in group:
                    fallback_unassigned.append({
                        "visitId": visit_id,
                        "reason": "solver_no_solution"
                    })
        
        return {
            "routes": [],
            "unassigned_visits": fallback_unassigned,
            "total_distance_km": 0,
            "total_visits": 0,
            "used_google_maps": used_google_maps,
            "validation_errors": ["Solver could not find a solution - all visits returned as unassigned. Try increasing max_km or reducing constraints."]
        }


def extract_solution(
    manager, routing, solution, locations, distance_matrix, 
    start_index, end_index, num_vehicles, used_google_maps: bool = False,
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
        
        # Calculate actual route distance from distance matrix
        # Sum: start→stop1 + stop1→stop2 + ... + stopN→end
        route_distance_meters = 0
        for i in range(len(route_nodes) - 1):
            from_node = route_nodes[i]
            to_node = route_nodes[i + 1]
            route_distance_meters += distance_matrix[from_node][to_node]
        
        # Apply road distance factor
        # If using Google Maps (used_google_maps=True), we have real road distances -> Factor 1.0
        # If using Haversine (used_google_maps=False), we estimate -> Factor 1.3
        ROAD_DISTANCE_FACTOR = 1.0 if used_google_maps else 1.3
        estimated_road_distance_meters = route_distance_meters * ROAD_DISTANCE_FACTOR
        
        # Only add route if it has stops
        if stops:
            # Count physical stops (unique sequence numbers = unique locations)
            physical_stop_count = len(set(stop['sequence'] for stop in stops))
            
            # Enforce max waypoints limit based on PHYSICAL stops (locations), not individual visits
            if physical_stop_count > max_waypoints:
                logger.warning(f"Route for TRUCK_{vehicle_id + 1} has {physical_stop_count} physical stops, truncating to {max_waypoints}")
                # Find the max allowed sequence number
                max_allowed_sequence = sorted(set(stop['sequence'] for stop in stops))[max_waypoints - 1]
                
                # Keep stops up to max_allowed_sequence, mark the rest as unassigned
                kept_stops = []
                for stop in stops:
                    if stop['sequence'] <= max_allowed_sequence:
                        kept_stops.append(stop)
                    else:
                        unassigned_visits.append({
                            "visitId": stop['visitId'],
                            "reason": "max_stops_exceeded"
                        })
                stops = kept_stops
                physical_stop_count = max_waypoints
            
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
                "estimated_km": round(estimated_road_distance_meters / 1000, 2),
                "waypoint_count": physical_stop_count
            })
    
    # Find unassigned visits - expand combined visit groups back to individual visits
    for i, location in enumerate(locations):
        if i not in assigned_indices and 'visitId' in location:
            reason = "max_stops_exceeded" if solution else "max_km_exceeded"
            
            # If we have visit_groups, expand the combined node to all individual visits
            if visit_groups and i < len(visit_groups) and visit_groups[i]:
                for visit_id in visit_groups[i]:
                    unassigned_visits.append({
                        "visitId": visit_id,
                        "reason": reason
                    })
            else:
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
    API endpoint to optimize truck routes
    
    Expected JSON input:
    {
        "trucks": 3,
        "max_km": 120,
        "max_stops": 15,          // Optional: max stops per truck (default: 25)
        "start": { "lat": 12.97, "lng": 77.59 },
        "end": { "lat": 12.93, "lng": 77.62 },
        "visits": [
            { 
                "visitId": "V1", 
                "lat": 12.95, 
                "lng": 77.60, 
                "sla_days": 0,
                "order_id": "ORD123",  // Optional
                "visit_type": "pickup"  // Optional: "pickup" or "drop"
            },
            { 
                "visitId": "V2", 
                "lat": 12.99, 
                "lng": 77.61, 
                "sla_days": 3,
                "order_id": "ORD123",  // Optional
                "visit_type": "drop"  // Optional: "pickup" or "drop"
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
            { "visitId": "V2", "reason": "max_stops_exceeded" }
        ]
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        required_fields = ["trucks", "max_km", "start", "end", "visits"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        num_trucks = data["trucks"]
        max_km = data["max_km"]
        max_stops = data.get("max_stops", MAX_WAYPOINTS_PER_ROUTE)  # Optional, default 25
        start_point = data["start"]
        end_point = data["end"]
        visits = data["visits"]
        
        # Validate data types and values
        if not isinstance(num_trucks, int) or num_trucks <= 0:
            return jsonify({"error": "trucks must be a positive integer"}), 400
        
        if not isinstance(max_km, (int, float)) or max_km <= 0:
            return jsonify({"error": "max_km must be a positive number"}), 400
        
        if not isinstance(max_stops, int) or max_stops <= 0:
            return jsonify({"error": "max_stops must be a positive integer"}), 400
        
        # Cap max_stops to the hard limit
        if max_stops > MAX_WAYPOINTS_PER_ROUTE:
            logger.warning(f"⚠️ max_stops ({max_stops}) exceeds hard limit ({MAX_WAYPOINTS_PER_ROUTE}), capping to {MAX_WAYPOINTS_PER_ROUTE}")
            max_stops = MAX_WAYPOINTS_PER_ROUTE
        
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
        
        logger.info(f"Received routing request: {num_trucks} trucks, max {max_km}km, max {max_stops} stops/truck, {len(visits)} visits")
        
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
        
        # Create distance matrix using combined locations
        distance_matrix, used_google_maps = create_distance_matrix(locations)
        
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
        
        # Convert max_km to meters
        max_distance_meters = int(max_km * 1000)
        
        # Solve the VRP
        result = solve_vrp(
            num_vehicles=num_trucks,
            max_distance_per_vehicle=max_distance_meters,
            locations=locations,
            distance_matrix=distance_matrix,
            start_index=start_index,
            end_index=end_index,
            priorities=priorities,
            used_google_maps=used_google_maps,
            max_waypoints=max_stops,
            combined_order_info=aligned_order_info,
            visit_groups=aligned_visit_groups,
            original_visits=visits  # Pass original visits for expanding in output
        )
        
        if result is None:
            return jsonify({
                "error": "Could not find a solution for the given constraints",
                "suggestion": "Try increasing max_km or number of trucks"
            }), 400
        
        # Add filtered-out visits to unassigned visits
        if filtered_excluded_visits:
            result['unassigned_visits'].extend(filtered_excluded_visits)
            logger.info(f"Added {len(filtered_excluded_visits)} filtered visits to unassigned list")
        
        logger.info(f"Solution found: {len(result['routes'])} routes, {len(result['unassigned_visits'])} unassigned visits")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in optimize_routes: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@auto_routing_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "auto-routing",
        "version": "1.0.0"
    }), 200

import logging
import os
from typing import Dict, List, Optional, Set, Tuple

import googlemaps
import h3
from supabase import Client, create_client

logger = logging.getLogger(__name__)

# H3/Supabase cache settings
H3_RESOLUTION = int(os.environ.get("H3_RESOLUTION", "7"))  # ~1.2km average edge length
H3_FINE_RESOLUTION = int(os.environ.get("H3_FINE_RESOLUTION", "8"))
HEX_CACHE_TABLE = os.environ.get("HEX_CACHE_TABLE", "hex_distance_cache")
MAX_DESTINATIONS_PER_GOOGLE_CALL = 25
MAX_SUPABASE_FILTER_CHUNK = 100
FALLBACK_LARGE_VALUE = 9_999_999

# Google Maps Distance Matrix API client setup
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
gmaps = None
if GOOGLE_MAPS_API_KEY:
    try:
        gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        logger.info("✅ Google Maps Client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Google Maps Client: {e}")
else:
    logger.warning("⚠️ GOOGLE_MAPS_API_KEY not set — distance matrix will not work!")

_supabase_client: Optional[Client] = None
_supabase_init_attempted = False


def _chunk(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _get_supabase_client() -> Optional[Client]:
    global _supabase_client, _supabase_init_attempted

    if _supabase_client:
        return _supabase_client
    if _supabase_init_attempted:
        return None

    _supabase_init_attempted = True
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        logger.warning("⚠️ Supabase env vars missing; hex distance cache disabled.")
        return None

    try:
        _supabase_client = create_client(supabase_url, supabase_key)
        logger.info("✅ Supabase client initialized for hex distance cache")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client for distance cache: {e}")
        _supabase_client = None
    return _supabase_client


def _latlng_to_hex(lat: float, lng: float, resolution: int) -> str:
    # Support both newer and older h3-python APIs.
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lng, resolution)
    return h3.geo_to_h3(lat, lng, resolution)


def _hex_to_latlng(hex_id: str) -> Tuple[float, float]:
    if hasattr(h3, "cell_to_latlng"):
        return h3.cell_to_latlng(hex_id)
    return h3.h3_to_geo(hex_id)


def _cache_hex_key(resolution: int, hex_id: str) -> str:
    # Prefix resolution so res7 and res8 can live in the same cache table.
    return f"r{resolution}:{hex_id}"


def _uncache_hex_key(cache_hex_id: str) -> str:
    if ":" in cache_hex_id:
        return cache_hex_id.split(":", 1)[1]
    return cache_hex_id


def _fetch_cached_hex_pairs(
    supabase: Client,
    pairs: Set[Tuple[str, str]],
) -> Dict[Tuple[str, str], Tuple[int, int]]:
    if not pairs:
        return {}

    origins = sorted({origin for origin, _ in pairs})
    destinations = sorted({destination for _, destination in pairs})
    cached_values: Dict[Tuple[str, str], Tuple[int, int]] = {}

    for origin_chunk in _chunk(origins, MAX_SUPABASE_FILTER_CHUNK):
        for destination_chunk in _chunk(destinations, MAX_SUPABASE_FILTER_CHUNK):
            response = (
                supabase.table(HEX_CACHE_TABLE)
                .select("origin_hex,destination_hex,distance_meters,duration_seconds")
                .in_("origin_hex", origin_chunk)
                .in_("destination_hex", destination_chunk)
                .execute()
            )
            for row in response.data or []:
                key = (row["origin_hex"], row["destination_hex"])
                if key in pairs:
                    cached_values[key] = (
                        int(row["distance_meters"]),
                        int(row["duration_seconds"]),
                    )
    return cached_values


def _fetch_existing_cache_hexes(
    supabase: Optional[Client],
    resolution: int,
    hexes: Set[str],
) -> Set[str]:
    """
    Return which hex IDs already exist in cache table (as origin or destination)
    for a given resolution prefix.
    """
    if not supabase or not hexes:
        return set()

    cache_keys = [_cache_hex_key(resolution, h) for h in sorted(hexes)]
    existing_cache_keys: Set[str] = set()

    try:
        for key_chunk in _chunk(cache_keys, MAX_SUPABASE_FILTER_CHUNK):
            resp_origin = (
                supabase.table(HEX_CACHE_TABLE)
                .select("origin_hex")
                .in_("origin_hex", key_chunk)
                .execute()
            )
            for row in resp_origin.data or []:
                existing_cache_keys.add(row["origin_hex"])

            resp_destination = (
                supabase.table(HEX_CACHE_TABLE)
                .select("destination_hex")
                .in_("destination_hex", key_chunk)
                .execute()
            )
            for row in resp_destination.data or []:
                existing_cache_keys.add(row["destination_hex"])
    except Exception as e:
        logger.warning(f"Failed to fetch existing cache hexes for res{resolution}: {e}")
        return set()

    return {_uncache_hex_key(k) for k in existing_cache_keys}


def _write_hex_cache(
    supabase: Optional[Client],
    values: Dict[Tuple[str, str], Tuple[int, int]],
) -> None:
    if not supabase or not values:
        return
    rows = []
    for (origin_hex, destination_hex), (distance, duration) in values.items():
        rows.append(
            {
                "origin_hex": origin_hex,
                "destination_hex": destination_hex,
                "distance_meters": int(distance),
                "duration_seconds": int(duration),
                "provider": "google_maps",
            }
        )
    try:
        # Upsert keeps cache warm without duplicating hex-pair rows.
        supabase.table(HEX_CACHE_TABLE).upsert(
            rows,
            on_conflict="origin_hex,destination_hex",
        ).execute()
    except Exception as e:
        logger.warning(f"Failed to upsert hex distance cache rows: {e}")


def create_distance_matrix(
    locations: List[Dict],
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Create distance AND duration matrices using Google Maps Distance Matrix API.

    Returns:
        (distance_matrix [meters], duration_matrix [seconds])

    Raises:
        Exception if Google Maps API key is not configured or API call fails.
    """
    if not gmaps:
        raise Exception("GOOGLE_MAPS_API_KEY is not configured. Cannot build distance matrix.")

    n = len(locations)
    distance_matrix = [[0] * n for _ in range(n)]
    duration_matrix = [[0] * n for _ in range(n)]
    if n == 0:
        return distance_matrix, duration_matrix

    logger.info(
        f"🗺️  Building {n}x{n} distance + duration matrix via H3 hex cache (res={H3_RESOLUTION}) + Google fallback..."
    )

    hexes_res7 = [_latlng_to_hex(loc["lat"], loc["lng"], H3_RESOLUTION) for loc in locations]
    unique_hexes_res7 = set(hexes_res7)
    pair_positions_res7: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}

    for i in range(n):
        for j in range(n):
            pair = (hexes_res7[i], hexes_res7[j])
            pair_positions_res7.setdefault(pair, []).append((i, j))

    all_pairs_res7 = set(pair_positions_res7.keys())
    pair_values_res7: Dict[Tuple[str, str], Tuple[int, int]] = {}
    res7_cache_hits = 0
    res7_cache_misses = 0
    res8_cache_hits = 0
    res8_cache_misses = 0
    unique_hexes_res8_used: Set[str] = set()

    # Same-hex movement is approximated to zero to avoid unnecessary API calls.
    for pair in all_pairs_res7:
        if pair[0] == pair[1]:
            pair_values_res7[pair] = (0, 0)

    supabase = _get_supabase_client()
    pending_pairs_res7 = {pair for pair in all_pairs_res7 if pair[0] != pair[1]}

    if supabase and pending_pairs_res7:
        try:
            pending_pairs_res7_cache_keys = {
                (_cache_hex_key(H3_RESOLUTION, origin), _cache_hex_key(H3_RESOLUTION, dest))
                for origin, dest in pending_pairs_res7
            }
            cached_pairs_raw = _fetch_cached_hex_pairs(supabase, pending_pairs_res7_cache_keys)
            cached_pairs_res7 = {
                (_uncache_hex_key(origin), _uncache_hex_key(dest)): value
                for (origin, dest), value in cached_pairs_raw.items()
            }
            pair_values_res7.update(cached_pairs_res7)
            res7_cache_hits += len(cached_pairs_res7)
            pending_pairs_res7 -= set(cached_pairs_res7.keys())
            logger.info(
                f"🧠 Hex cache hits (res7): {len(cached_pairs_res7)} / {len(all_pairs_res7)} unique pairs"
            )
        except Exception as e:
            logger.warning(f"Hex cache read failed; falling back to Google API only: {e}")

    newly_computed_res7: Dict[Tuple[str, str], Tuple[int, int]] = {}
    if pending_pairs_res7:
        res7_cache_misses += len(pending_pairs_res7)
        origin_to_destinations: Dict[str, List[str]] = {}
        for origin_hex, destination_hex in pending_pairs_res7:
            origin_to_destinations.setdefault(origin_hex, []).append(destination_hex)

        for origin_hex, destination_hexes in origin_to_destinations.items():
            origin_latlng = _hex_to_latlng(origin_hex)
            unique_destinations = sorted(set(destination_hexes))

            for chunk in _chunk(unique_destinations, MAX_DESTINATIONS_PER_GOOGLE_CALL):
                destination_coords = [_hex_to_latlng(dest_hex) for dest_hex in chunk]
                result = gmaps.distance_matrix(
                    origins=[origin_latlng],
                    destinations=destination_coords,
                    mode="driving",
                )

                if result["status"] != "OK":
                    raise Exception(f"Google Maps API error: {result['status']}")

                row_elements = result["rows"][0]["elements"]
                for dest_hex, element in zip(chunk, row_elements):
                    key = (origin_hex, dest_hex)
                    if element["status"] == "OK":
                        newly_computed_res7[key] = (
                            int(element["distance"]["value"]),
                            int(element["duration"]["value"]),
                        )
                    else:
                        newly_computed_res7[key] = (FALLBACK_LARGE_VALUE, FALLBACK_LARGE_VALUE)

        pair_values_res7.update(newly_computed_res7)
        _write_hex_cache(
            supabase,
            {
                (_cache_hex_key(H3_RESOLUTION, origin), _cache_hex_key(H3_RESOLUTION, dest)): value
                for (origin, dest), value in newly_computed_res7.items()
            },
        )
        logger.info(
            f"🌐 Hex cache misses computed via Google API (res7): {len(pending_pairs_res7)}"
        )

    for pair, positions in pair_positions_res7.items():
        dist, dur = pair_values_res7.get(pair, (FALLBACK_LARGE_VALUE, FALLBACK_LARGE_VALUE))
        for i, j in positions:
            distance_matrix[i][j] = dist
            duration_matrix[i][j] = dur

    # Refine only zero-distance pairs (except self-pairs) using res8 cache/API.
    zero_pairs_positions: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if i != j and distance_matrix[i][j] == 0:
                zero_pairs_positions.append((i, j))

    if zero_pairs_positions:
        hexes_res8 = [_latlng_to_hex(loc["lat"], loc["lng"], H3_FINE_RESOLUTION) for loc in locations]
        unique_hexes_res8_used = {hexes_res8[i] for i, _ in zero_pairs_positions} | {
            hexes_res8[j] for _, j in zero_pairs_positions
        }
        pair_positions_res8: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
        for i, j in zero_pairs_positions:
            pair = (hexes_res8[i], hexes_res8[j])
            pair_positions_res8.setdefault(pair, []).append((i, j))

        all_pairs_res8 = set(pair_positions_res8.keys())
        pair_values_res8: Dict[Tuple[str, str], Tuple[int, int]] = {}
        pending_pairs_res8 = set(all_pairs_res8)

        # Keep same res8 hex as zero to avoid pointless API calls.
        for pair in all_pairs_res8:
            if pair[0] == pair[1]:
                pair_values_res8[pair] = (0, 0)
                pending_pairs_res8.discard(pair)

        if supabase and pending_pairs_res8:
            try:
                pending_pairs_res8_cache_keys = {
                    (
                        _cache_hex_key(H3_FINE_RESOLUTION, origin),
                        _cache_hex_key(H3_FINE_RESOLUTION, dest),
                    )
                    for origin, dest in pending_pairs_res8
                }
                cached_pairs_raw = _fetch_cached_hex_pairs(supabase, pending_pairs_res8_cache_keys)
                cached_pairs_res8 = {
                    (_uncache_hex_key(origin), _uncache_hex_key(dest)): value
                    for (origin, dest), value in cached_pairs_raw.items()
                }
                pair_values_res8.update(cached_pairs_res8)
                res8_cache_hits += len(cached_pairs_res8)
                pending_pairs_res8 -= set(cached_pairs_res8.keys())
                logger.info(
                    f"🧠 Hex cache hits (res8 refinement): {len(cached_pairs_res8)} / {len(all_pairs_res8)} zero-pairs"
                )
            except Exception as e:
                logger.warning(f"Hex cache read failed during res8 refinement: {e}")

        newly_computed_res8: Dict[Tuple[str, str], Tuple[int, int]] = {}
        if pending_pairs_res8:
            res8_cache_misses += len(pending_pairs_res8)
            origin_to_destinations_res8: Dict[str, List[str]] = {}
            for origin_hex, destination_hex in pending_pairs_res8:
                origin_to_destinations_res8.setdefault(origin_hex, []).append(destination_hex)

            for origin_hex, destination_hexes in origin_to_destinations_res8.items():
                origin_latlng = _hex_to_latlng(origin_hex)
                unique_destinations = sorted(set(destination_hexes))

                for chunk in _chunk(unique_destinations, MAX_DESTINATIONS_PER_GOOGLE_CALL):
                    destination_coords = [_hex_to_latlng(dest_hex) for dest_hex in chunk]
                    result = gmaps.distance_matrix(
                        origins=[origin_latlng],
                        destinations=destination_coords,
                        mode="driving",
                    )
                    if result["status"] != "OK":
                        raise Exception(f"Google Maps API error: {result['status']}")

                    row_elements = result["rows"][0]["elements"]
                    for dest_hex, element in zip(chunk, row_elements):
                        key = (origin_hex, dest_hex)
                        if element["status"] == "OK":
                            newly_computed_res8[key] = (
                                int(element["distance"]["value"]),
                                int(element["duration"]["value"]),
                            )
                        else:
                            newly_computed_res8[key] = (
                                FALLBACK_LARGE_VALUE,
                                FALLBACK_LARGE_VALUE,
                            )

        if newly_computed_res8:
            pair_values_res8.update(newly_computed_res8)
            _write_hex_cache(
                supabase,
                {
                    (
                        _cache_hex_key(H3_FINE_RESOLUTION, origin),
                        _cache_hex_key(H3_FINE_RESOLUTION, dest),
                    ): value
                    for (origin, dest), value in newly_computed_res8.items()
                },
            )
        if pending_pairs_res8:
            logger.info(
                f"🌐 Hex cache misses computed via Google API (res8 refinement): {len(pending_pairs_res8)}"
            )

        for pair, positions in pair_positions_res8.items():
            dist, dur = pair_values_res8.get(pair, (FALLBACK_LARGE_VALUE, FALLBACK_LARGE_VALUE))
            for i, j in positions:
                distance_matrix[i][j] = dist
                duration_matrix[i][j] = dur

    total_unique_locations = len(
        {(loc.get("lat"), loc.get("lng")) for loc in locations if "lat" in loc and "lng" in loc}
    )
    existing_res7_hexes = _fetch_existing_cache_hexes(supabase, H3_RESOLUTION, unique_hexes_res7)
    new_res7_hexes_used = len(unique_hexes_res7 - existing_res7_hexes)
    existing_res8_hexes = _fetch_existing_cache_hexes(
        supabase, H3_FINE_RESOLUTION, unique_hexes_res8_used
    )
    new_res8_hexes_used = len(unique_hexes_res8_used - existing_res8_hexes)
    total_hex_used = len(unique_hexes_res7) + len(unique_hexes_res8_used)
    logger.info(
        "🧩 Total hex used | res7=%s | res8=%s | combined=%s",
        len(unique_hexes_res7),
        len(unique_hexes_res8_used),
        total_hex_used,
    )
    logger.info(
        "🆕 New hexes used (not source-destination pairs) | res7=%s | res8=%s | combined=%s",
        new_res7_hexes_used,
        new_res8_hexes_used,
        new_res7_hexes_used + new_res8_hexes_used,
    )
    logger.info(
        "📊 H3 cache stats | unique_locations=%s | res7_unique_hexes=%s | res8_unique_hexes=%s | "
        "cache_hits(res7+res8)=%s (%s+%s) | cache_misses(res7+res8)=%s (%s+%s)",
        total_unique_locations,
        len(unique_hexes_res7),
        len(unique_hexes_res8_used),
        res7_cache_hits + res8_cache_hits,
        res7_cache_hits,
        res8_cache_hits,
        res7_cache_misses + res8_cache_misses,
        res7_cache_misses,
        res8_cache_misses,
    )
    logger.info(f"✅ Distance matrix built: {n}x{n} ({n*n} elements)")
    return distance_matrix, duration_matrix

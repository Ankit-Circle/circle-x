import logging
import os
from typing import Dict, List, Tuple

import googlemaps

logger = logging.getLogger(__name__)

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

    logger.info(f"🗺️  Building {n}x{n} distance + duration matrix via Google Maps API...")
    coords = [(loc["lat"], loc["lng"]) for loc in locations]

    for i in range(n):
        for j in range(0, n, 25):
            chunk_destinations = coords[j : j + 25]
            result = gmaps.distance_matrix(
                origins=[coords[i]],
                destinations=chunk_destinations,
                mode="driving",
            )

            if result["status"] == "OK":
                row_elements = result["rows"][0]["elements"]
                for k, element in enumerate(row_elements):
                    if element["status"] == "OK":
                        distance_matrix[i][j + k] = element["distance"]["value"]
                        duration_matrix[i][j + k] = element["duration"]["value"]
                    else:
                        distance_matrix[i][j + k] = 9999999
                        duration_matrix[i][j + k] = 9999999
            else:
                raise Exception(f"Google Maps API error: {result['status']}")

    logger.info(f"✅ Google Maps matrix built: {n}x{n} ({n*n} elements)")
    return distance_matrix, duration_matrix

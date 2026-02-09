# circle-x AI API Server

A Flask-based API server providing AI-powered services including vehicle route optimization, product pricing, and image enhancement.

## Running with Docker

Build the Docker image:

```bash
docker build -t circle-x .
```

Run the Docker container:

```bash
docker run -p 5000:5000 circle-x
```

## API Endpoints

### Auto Routing API — Hybrid VRP (CVRP + VRPPD + VRPTW)

**Endpoint:** `POST /api/auto-routing/optimize`

📖 **[See detailed Auto Routing API documentation →](AUTO_ROUTING_README.md)**

Optimizes delivery routes for multiple vehicles using a **Hybrid VRP** solver combining:
- **CVRP** — Capacity constraints (max km, max stops per truck)
- **VRPPD** — Pickup & delivery pairs (same vehicle, pickup before drop)
- **VRPTW** — Time windows and shift duration limits

Uses real road distances via Google Maps (paid) → OSRM (free) → Haversine (fallback).

### Pricing API

**Endpoint:** `POST /api/pricing`

```bash
curl --location '0.0.0.0:5000/api/pricing' \
--header 'Content-Type: application/json' \
--data '{
    "Category": "Audio",
    "Sub_Category": "Bluetooth Speaker",
    "Product_Age_Years1": 9,
    "brand_tier": "Premium",
    "warranty_period": "6 months",
    "Condition_Tier": "Very Good",
    "Functional_Status": "Fully Functional",
    "Best_Online_Price": 4500
}'
```

### Image Enhancement API

**Endpoint:** `POST /api/enhance`

```bash
curl --location '0.0.0.0:5000/api/enhance' \
--header 'Content-Type: application/json' \
--data '{
    "image_url": "https://res.cloudinary.com/dkjsiqjfr/image/upload/v1752215572/product_submissions/IMG_5593_1.jpg"
}'
```

## Environment Variables

Create a `.env` file with:

```bash
GOOGLE_MAPS_API_KEY=your_api_key_here   # Optional: enables Google Maps (paid, most accurate)
OSRM_BASE_URL=http://router.project-osrm.org  # Optional: OSRM server for free road distances
```

## Tech Stack

- **Flask**: Web framework
- **Google OR-Tools**: Hybrid VRP solver (CVRP + VRPPD + VRPTW)
- **Google Maps API**: Distance matrix (paid, primary)
- **OSRM**: Open Source Routing Machine (free, secondary)
- **Flask-CORS**: Cross-origin support

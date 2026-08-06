import importlib
import os
import sys
import types
import unittest

from flask import Flask


def _install_import_stubs() -> None:
    """Install lightweight stubs for optional provider modules used at import-time."""
    if "googlemaps" not in sys.modules:
        mod_googlemaps = types.ModuleType("googlemaps")

        class _DummyGMClient:
            def __init__(self, *args, **kwargs):
                pass

        mod_googlemaps.Client = _DummyGMClient
        sys.modules["googlemaps"] = mod_googlemaps

    if "h3" not in sys.modules:
        mod_h3 = types.ModuleType("h3")

        def _latlng_to_cell(lat, lng, resolution):
            return f"{resolution}:{lat:.4f}:{lng:.4f}"

        def _cell_to_latlng(cell):
            parts = cell.split(":")
            return float(parts[1]), float(parts[2])

        mod_h3.latlng_to_cell = _latlng_to_cell
        mod_h3.cell_to_latlng = _cell_to_latlng
        sys.modules["h3"] = mod_h3

    if "supabase" not in sys.modules:
        mod_supabase = types.ModuleType("supabase")

        class _DummyClient:
            pass

        def _create_client(*args, **kwargs):
            return _DummyClient()

        mod_supabase.Client = _DummyClient
        mod_supabase.create_client = _create_client
        sys.modules["supabase"] = mod_supabase


class WarehouseReloadRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_import_stubs()
        # Keep solver runtime short in tests.
        os.environ["AUTO_ROUTING_SOLVE_BUDGET_SEC"] = "5"
        os.environ["AUTO_ROUTING_MIN_PROFILE_SECONDS"] = "1"
        os.environ["AUTO_ROUTING_WORKER_TIMEOUT_SEC"] = "20"

        cls.routing = importlib.import_module("app.auto_routing.routing")

        def fake_matrix(locations):
            n = len(locations)
            dist = [[0 if i == j else 1000 for j in range(n)] for i in range(n)]
            dur = [[0 if i == j else 300 for j in range(n)] for i in range(n)]
            return dist, dur

        cls.routing.create_distance_matrix = fake_matrix

    def _client(self):
        app = Flask(__name__)
        app.register_blueprint(self.routing.auto_routing_bp)
        return app.test_client()

    def test_optimize_never_starts_with_warehouse_reload(self):
        client = self._client()

        payload = {
            "trucks": 1,
            "max_km": 200,
            "max_stops": 20,
            "truck_capacity": 5,
            "enable_warehouse_reload": True,
            "max_reloads_per_truck": 2,
            "warehouse": {"lat": 12.97, "lng": 77.59},
            "reload_service_time_minutes": 10,
            "start": {"lat": 12.97, "lng": 77.59},
            "end": {"lat": 12.97, "lng": 77.59},
            "visits": [
                {
                    "visitId": "D_PRE",
                    "lat": 12.971,
                    "lng": 77.591,
                    "sla_days": 0,
                    "order_id": "P1",
                    "visit_type": "drop",
                    "vol_capacity": 4,
                },
                {
                    "visitId": "PU_1",
                    "lat": 12.972,
                    "lng": 77.592,
                    "sla_days": 0,
                    "order_id": "O1",
                    "visit_type": "pickup",
                    "vol_capacity": 4,
                },
                {
                    "visitId": "DR_1",
                    "lat": 12.973,
                    "lng": 77.593,
                    "sla_days": 0,
                    "order_id": "O1",
                    "visit_type": "drop",
                    "vol_capacity": 4,
                },
                {
                    "visitId": "PU_2",
                    "lat": 12.974,
                    "lng": 77.594,
                    "sla_days": 0,
                    "order_id": "O2",
                    "visit_type": "pickup",
                    "vol_capacity": 4,
                },
                {
                    "visitId": "DR_2",
                    "lat": 12.975,
                    "lng": 77.595,
                    "sla_days": 0,
                    "order_id": "O2",
                    "visit_type": "drop",
                    "vol_capacity": 4,
                },
            ],
        }

        resp = client.post("/optimize", json=payload)
        self.assertEqual(resp.status_code, 200, resp.get_json())

        body = resp.get_json() or {}
        routes = body.get("routes", [])
        self.assertGreater(len(routes), 0)

        for route in routes:
            stops = route.get("stops", [])
            if not stops:
                continue
            self.assertNotEqual(stops[0].get("visit_type"), "warehouse_reload")

            for stop in stops:
                if stop.get("visit_type") == "warehouse_reload":
                    self.assertGreater(stop.get("sequence", 0), 1)

    def test_enforce_volume_capacity_does_not_insert_reload_at_sequence_one(self):
        routes = [
            {
                "truckId": "TRUCK_1",
                "start": {"lat": 12.97, "lng": 77.59},
                "end": {"lat": 12.97, "lng": 77.59},
                "stops": [
                    {
                        "visitId": "D1",
                        "lat": 12.971,
                        "lng": 77.591,
                        "sequence": 1,
                        "order_id": "P1",
                        "visit_type": "drop",
                    },
                    {
                        "visitId": "PU1",
                        "lat": 12.972,
                        "lng": 77.592,
                        "sequence": 2,
                        "order_id": "O1",
                        "visit_type": "pickup",
                    },
                    {
                        "visitId": "DR1",
                        "lat": 12.973,
                        "lng": 77.593,
                        "sequence": 3,
                        "order_id": "O1",
                        "visit_type": "drop",
                    },
                ],
                "estimated_km": 0.0,
                "estimated_hours": 0.0,
                "waypoint_count": 3,
                "total_visits": 3,
            }
        ]

        original_visits = [
            {
                "visitId": "D1",
                "lat": 12.971,
                "lng": 77.591,
                "sla_days": 10,
                "order_id": "P1",
                "visit_type": "drop",
                "vol_capacity": 8,
            },
            {
                "visitId": "PU1",
                "lat": 12.972,
                "lng": 77.592,
                "sla_days": 0,
                "order_id": "O1",
                "visit_type": "pickup",
                "vol_capacity": 4,
            },
            {
                "visitId": "DR1",
                "lat": 12.973,
                "lng": 77.593,
                "sla_days": 0,
                "order_id": "O1",
                "visit_type": "drop",
                "vol_capacity": 4,
            },
        ]

        locations = [
            {"lat": 12.97, "lng": 77.59},
            {"lat": 12.971, "lng": 77.591, "visitId": "D1"},
            {"lat": 12.972, "lng": 77.592, "visitId": "PU1"},
            {"lat": 12.973, "lng": 77.593, "visitId": "DR1"},
            {"lat": 12.97, "lng": 77.59},
        ]

        n = len(locations)
        distance_matrix = [[0 if i == j else 1000 for j in range(n)] for i in range(n)]
        duration_matrix = [[0 if i == j else 300 for j in range(n)] for i in range(n)]

        updated_routes, updated_unassigned = self.routing.enforce_volume_capacity(
            routes=routes,
            unassigned_visits=[],
            original_visits=original_visits,
            truck_capacity=5.0,
            distance_matrix=distance_matrix,
            duration_matrix=duration_matrix,
            locations=locations,
            start_index=0,
            end_index=4,
            max_distance_per_vehicle=100000,
            max_route_time=36000,
            enable_warehouse_reload=True,
            warehouse_location={"lat": 12.97, "lng": 77.59},
            max_reloads_per_truck=2,
            reload_service_time_seconds=600,
        )

        self.assertGreater(len(updated_routes), 0)
        for route in updated_routes:
            stops = route.get("stops", [])
            if not stops:
                continue
            self.assertNotEqual(stops[0].get("visit_type"), "warehouse_reload")

        first_route_stops = updated_routes[0].get("stops", [])
        self.assertTrue(
            all(stop.get("visit_type") != "warehouse_reload" for stop in first_route_stops)
        )

        reasons = [item.get("reason") for item in updated_unassigned]
        self.assertIn("volume_capacity_exceeded", reasons)


if __name__ == "__main__":
    unittest.main()

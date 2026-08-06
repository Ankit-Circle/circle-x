import unittest

from app.auto_routing.routing import (
    allowed_vehicle_classes_for_node,
    delivery_passes_qc,
    normalize_vehicle_specs,
)


class MixedFleetContractTests(unittest.TestCase):
    def test_legacy_payload_does_not_enable_mixed_fleet(self):
        self.assertIsNone(normalize_vehicle_specs({"trucks": 1, "max_km": 20}))

    def test_vehicle_specs_accept_per_vehicle_depots_and_limits(self):
        specs = normalize_vehicle_specs({
            "vehicles": [{
                "vehicle_id": "truck_1",
                "planner_vehicle_class": "large_truck",
                "start": {"lat": 12.97, "lng": 77.59},
                "end": {"lat": 12.93, "lng": 77.62},
                "max_km": 120,
                "max_points": 25,
                "capacity": 50,
                "shift_limit_hours": 10,
                "service_time_minutes": 8,
                "reload_limit": 1,
            }],
        })

        self.assertEqual(specs[0].vehicle_id, "truck_1")
        self.assertEqual(specs[0].max_stops, 25)
        self.assertEqual(specs[0].truck_capacity, 50)
        self.assertEqual(specs[0].max_reloads, 1)
        self.assertEqual(specs[0].end["lng"], 77.62)

    def test_combined_location_uses_strictest_size_class(self):
        visits = {
            "small": {"size": "S"},
            "medium": {"size": "M"},
            "large": {"size": "XL"},
        }
        self.assertEqual(
            allowed_vehicle_classes_for_node(["small"], visits),
            {"bike"},
        )
        self.assertEqual(
            allowed_vehicle_classes_for_node(["small", "medium"], visits),
            {"medium_truck"},
        )
        self.assertEqual(
            allowed_vehicle_classes_for_node(["medium", "large"], visits),
            {"large_truck"},
        )

    def test_qc_only_filters_deliveries(self):
        self.assertTrue(delivery_passes_qc({"visit_type": "pickup"}))
        self.assertTrue(delivery_passes_qc({"visit_type": "delivery", "qc_status": "Passed"}))
        self.assertFalse(delivery_passes_qc({"visit_type": "drop", "qc_status": "pending"}))


if __name__ == "__main__":
    unittest.main()

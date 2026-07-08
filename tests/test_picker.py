from zonewatch.picker import MIN_ZONE_SIZE, ZoneDrafter
from zonewatch.zones import Zone


def make_drafter(**kwargs):
    return ZoneDrafter(width=640, height=480, **kwargs)


class TestZoneDrafter:
    def test_drag_creates_zone(self):
        drafter = make_drafter()
        drafter.press(100, 100)
        drafter.drag(200, 150)
        zone = drafter.release(200, 150)
        assert zone == Zone("zone-1", 100, 100, 200, 150)
        assert drafter.zones == [zone]

    def test_reverse_drag_normalized(self):
        drafter = make_drafter()
        drafter.press(200, 150)
        zone = drafter.release(100, 100)
        assert (zone.x1, zone.y1, zone.x2, zone.y2) == (100, 100, 200, 150)

    def test_tiny_drag_ignored(self):
        drafter = make_drafter()
        drafter.press(100, 100)
        zone = drafter.release(100 + MIN_ZONE_SIZE - 1, 100 + MIN_ZONE_SIZE - 1)
        assert zone is None
        assert drafter.zones == []

    def test_release_without_press_is_noop(self):
        drafter = make_drafter()
        assert drafter.release(50, 50) is None

    def test_coordinates_clamped_to_frame(self):
        drafter = make_drafter()
        drafter.press(-20, -20)
        zone = drafter.release(9999, 9999)
        assert (zone.x1, zone.y1, zone.x2, zone.y2) == (0, 0, 639, 479)

    def test_live_box_tracks_drag_then_clears(self):
        drafter = make_drafter()
        drafter.press(10, 10)
        drafter.drag(60, 60)
        assert drafter.live_box == (10, 10, 60, 60)
        drafter.release(60, 60)
        assert drafter.live_box is None

    def test_names_are_sequential_and_unique(self):
        drafter = make_drafter()
        for i in range(2):
            drafter.press(0, i * 100)
            drafter.release(50, i * 100 + 50)
        assert [z.name for z in drafter.zones] == ["zone-1", "zone-2"]

    def test_names_avoid_existing_zones(self):
        drafter = make_drafter(zones=[Zone("zone-1", 0, 0, 10, 10)])
        drafter.press(100, 100)
        zone = drafter.release(200, 200)
        assert zone.name == "zone-2"

    def test_undo_and_reset(self):
        drafter = make_drafter()
        drafter.press(0, 0)
        drafter.release(50, 50)
        drafter.undo()
        assert drafter.zones == []
        drafter.undo()  # undo on empty is a no-op
        drafter.press(0, 0)
        drafter.release(50, 50)
        drafter.reset()
        assert drafter.zones == [] and drafter.live_box is None

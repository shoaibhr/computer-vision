from zonewatch.events import Cooldown, ZoneMonitor
from zonewatch.zones import Zone

ZONE = Zone("door", 100, 100, 200, 200)
INSIDE = (120, 120, 180, 180)
OUTSIDE = (300, 300, 400, 400)


def make_monitor(**kwargs):
    return ZoneMonitor([ZONE], **kwargs)


class TestZoneMonitor:
    def test_enter_requires_consecutive_hits(self):
        monitor = make_monitor(enter_frames=3, exit_frames=2)
        assert monitor.update([INSIDE]) == []
        assert monitor.update([INSIDE]) == []
        events = monitor.update([INSIDE])
        assert len(events) == 1 and events[0].kind == "enter"
        assert monitor.is_occupied(ZONE)

    def test_flicker_does_not_trigger_enter(self):
        monitor = make_monitor(enter_frames=3, exit_frames=2)
        monitor.update([INSIDE])
        monitor.update([INSIDE])
        monitor.update([])  # streak broken
        monitor.update([INSIDE])
        assert monitor.update([INSIDE]) == []
        assert not monitor.is_occupied(ZONE)

    def test_exit_requires_consecutive_misses(self):
        monitor = make_monitor(enter_frames=1, exit_frames=3)
        monitor.update([INSIDE])
        assert monitor.update([]) == []
        assert monitor.update([]) == []
        events = monitor.update([])
        assert len(events) == 1 and events[0].kind == "exit"
        assert not monitor.is_occupied(ZONE)

    def test_box_outside_zone_is_a_miss(self):
        monitor = make_monitor(enter_frames=1, exit_frames=1)
        assert monitor.update([OUTSIDE]) == []
        assert not monitor.is_occupied(ZONE)

    def test_no_repeat_enter_while_occupied(self):
        monitor = make_monitor(enter_frames=1, exit_frames=1)
        assert len(monitor.update([INSIDE])) == 1
        assert monitor.update([INSIDE]) == []

    def test_multiple_zones_tracked_independently(self):
        other = Zone("dock", 300, 300, 400, 400)
        monitor = ZoneMonitor([ZONE, other], enter_frames=1, exit_frames=1)
        events = monitor.update([INSIDE])
        assert [e.zone.name for e in events] == ["door"]
        assert monitor.is_occupied(ZONE)
        assert not monitor.is_occupied(other)

    def test_event_payload(self):
        monitor = make_monitor(enter_frames=1, exit_frames=1, clock=lambda: 1234.5)
        event = monitor.update([INSIDE])[0]
        assert event.to_payload() == {
            "event": "enter",
            "zone": "door",
            "box": [100, 100, 200, 200],
            "timestamp": 1234.5,
        }


class TestCooldown:
    def test_first_call_ready_then_blocked(self):
        now = {"t": 0.0}
        cooldown = Cooldown(30, clock=lambda: now["t"])
        assert cooldown.ready("door")
        now["t"] = 10
        assert not cooldown.ready("door")
        now["t"] = 31
        assert cooldown.ready("door")

    def test_keys_independent(self):
        cooldown = Cooldown(30, clock=lambda: 0.0)
        assert cooldown.ready("door")
        assert cooldown.ready("dock")

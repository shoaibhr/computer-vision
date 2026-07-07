from zonewatch.events import Cooldown, ZoneEvent
from zonewatch.notifiers import NotifierHub
from zonewatch.zones import Zone

ZONE = Zone("door", 1, 1, 10, 10)


class RecordingNotifier:
    def __init__(self):
        self.events = []

    def notify(self, event, frame=None):
        self.events.append(event.kind)


class ExplodingNotifier:
    def notify(self, event, frame=None):
        raise RuntimeError("boom")


def enter_event(t=0.0):
    return ZoneEvent("enter", ZONE, t)


def exit_event(t=0.0):
    return ZoneEvent("exit", ZONE, t)


def test_dispatch_fans_out():
    a, b = RecordingNotifier(), RecordingNotifier()
    hub = NotifierHub([a, b])
    assert hub.dispatch(enter_event())
    assert a.events == ["enter"] and b.events == ["enter"]


def test_cooldown_suppresses_repeat_enters():
    sink = RecordingNotifier()
    now = {"t": 0.0}
    hub = NotifierHub([sink], Cooldown(30, clock=lambda: now["t"]))
    assert hub.dispatch(enter_event())
    now["t"] = 5
    assert not hub.dispatch(enter_event())
    assert sink.events == ["enter"]


def test_exit_bypasses_cooldown():
    sink = RecordingNotifier()
    hub = NotifierHub([sink], Cooldown(30, clock=lambda: 0.0))
    hub.dispatch(enter_event())
    assert hub.dispatch(exit_event())
    assert sink.events == ["enter", "exit"]


def test_failing_notifier_does_not_break_others():
    sink = RecordingNotifier()
    hub = NotifierHub([ExplodingNotifier(), sink])
    assert hub.dispatch(enter_event())
    assert sink.events == ["enter"]

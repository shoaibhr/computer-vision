import pytest

from zonewatch.zones import Zone, parse_zones


class TestZoneIntersects:
    zone = Zone("z", 100, 100, 200, 200)

    def test_box_fully_inside(self):
        assert self.zone.intersects((120, 120, 180, 180))

    def test_box_fully_containing_zone(self):
        assert self.zone.intersects((0, 0, 500, 500))

    def test_partial_overlap(self):
        assert self.zone.intersects((150, 150, 300, 300))

    def test_disjoint(self):
        assert not self.zone.intersects((300, 300, 400, 400))

    def test_edge_touching_does_not_count(self):
        assert not self.zone.intersects((0, 0, 100, 100))

    def test_degenerate_zone_rejected(self):
        with pytest.raises(ValueError, match="degenerate"):
            Zone("bad", 10, 10, 10, 50)


class TestParseZones:
    def test_named_zone(self):
        zones = parse_zones("entrance:118,6,218,206")
        assert zones == [Zone("entrance", 118, 6, 218, 206)]

    def test_unnamed_zone_gets_default_name(self):
        zones = parse_zones("10,10,50,50")
        assert zones[0].name == "zone-1"

    def test_multiple_zones(self):
        zones = parse_zones("a:1,1,2,2; b:3,3,4,4")
        assert [z.name for z in zones] == ["a", "b"]

    def test_whitespace_tolerated(self):
        zones = parse_zones(" dock : 1 , 2 , 3 , 4 ")
        assert zones == [Zone("dock", 1, 2, 3, 4)]

    def test_wrong_coordinate_count(self):
        with pytest.raises(ValueError, match="expected 4"):
            parse_zones("a:1,2,3")

    def test_non_integer_coordinates(self):
        with pytest.raises(ValueError, match="integers"):
            parse_zones("a:1,2,3,x")

    def test_empty_spec(self):
        with pytest.raises(ValueError, match="empty"):
            parse_zones("  ;  ")

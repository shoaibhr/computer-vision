"""Detection zones (rectangular regions of interest)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Zone:
    """A named rectangular region of interest, in pixel coordinates."""

    name: str
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(
                f"Zone '{self.name}' is degenerate: ({self.x1},{self.y1})-({self.x2},{self.y2}); "
                "x2 must be > x1 and y2 must be > y1"
            )

    def intersects(self, box: tuple[int, int, int, int]) -> bool:
        """Return True if the (x1, y1, x2, y2) box overlaps this zone."""
        bx1, by1, bx2, by2 = box
        return bx2 > self.x1 and bx1 < self.x2 and by2 > self.y1 and by1 < self.y2


def zones_to_spec(zones: list[Zone]) -> str:
    """Serialize zones back into the ``parse_zones`` string format."""
    return ";".join(f"{z.name}:{z.x1},{z.y1},{z.x2},{z.y2}" for z in zones)


def parse_zones(spec: str) -> list[Zone]:
    """Parse a zone spec string into a list of zones.

    Format: ``name:x1,y1,x2,y2`` entries separated by ``;``. The name (and
    colon) may be omitted, in which case zones are named ``zone-1``,
    ``zone-2``, ... e.g. ``entrance:118,6,218,206;300,50,500,400``.
    """
    zones: list[Zone] = []
    for i, entry in enumerate(filter(None, (e.strip() for e in spec.split(";"))), start=1):
        name, _, coords = entry.rpartition(":")
        name = name.strip() or f"zone-{i}"
        parts = [p.strip() for p in coords.split(",")]
        if len(parts) != 4:
            raise ValueError(
                f"Invalid zone '{entry}': expected 4 comma-separated coordinates "
                "(x1,y1,x2,y2), optionally prefixed with 'name:'"
            )
        try:
            x1, y1, x2, y2 = (int(p) for p in parts)
        except ValueError as exc:
            raise ValueError(f"Invalid zone '{entry}': coordinates must be integers") from exc
        zones.append(Zone(name, x1, y1, x2, y2))
    if not zones:
        raise ValueError("Zone spec is empty")
    return zones

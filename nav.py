"""Coarse, conservative ground navigation over the supplied static arena map."""

from __future__ import annotations

from array import array
import heapq
import json
import math
from pathlib import Path
import sys
from typing import Any


def point_in_ring(x: float, y: float, ring: list[float]) -> bool:
    inside = False
    count = len(ring) // 2
    if count < 3:
        return False
    j = count - 1
    for i in range(count):
        xi, yi = float(ring[2 * i]), float(ring[2 * i + 1])
        xj, yj = float(ring[2 * j]), float(ring[2 * j + 1])
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _segment_distance(x: float, y: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 <= 1e-12:
        return math.hypot(x - ax, y - ay)
    t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length2))
    return math.hypot(x - (ax + t * dx), y - (ay + t * dy))


def _ring_distance(x: float, y: float, ring: list[float]) -> float:
    count = len(ring) // 2
    best = math.inf
    for i in range(count):
        j = (i + 1) % count
        best = min(best, _segment_distance(x, y, ring[2 * i], ring[2 * i + 1], ring[2 * j], ring[2 * j + 1]))
    return best


class MapNavigator:
    """Build a coarse walkable grid once, then reuse bounded A* paths between decisions."""

    def __init__(self, world: dict[str, Any], terrain_data: bytes, *, grid_step: float = 12.0, radius: float = 2.5, height: float = 9.0, max_slope_deg: float = 50.0):
        terrain = world["terrain"]
        self.min_x = float(terrain["minX"])
        self.min_y = float(terrain["minY"])
        self.step = float(terrain["step"])
        self.nx = int(terrain["nx"])
        self.ny = int(terrain["ny"])
        self.scale = float(terrain.get("scale", 0.01))
        values = array("h")
        values.frombytes(terrain_data)
        if sys.byteorder != "little":
            values.byteswap()
        if len(values) != self.nx * self.ny:
            raise ValueError("terrain size does not match world.json")
        self.heights = values
        self.grid_step = float(grid_step)
        self.radius = radius
        self.height = height
        self.tan_slope = math.tan(math.radians(max_slope_deg))
        bounds = world["bounds"]
        self.origin_x = float(bounds["minX"]) + self.grid_step * 0.5
        self.origin_y = float(bounds["minY"]) + self.grid_step * 0.5
        self.width = max(1, int(math.ceil((float(bounds["maxX"]) - float(bounds["minX"])) / self.grid_step)))
        self.height_count = max(1, int(math.ceil((float(bounds["maxY"]) - float(bounds["minY"])) / self.grid_step)))
        size = self.width * self.height_count
        self._ground = [0.0] * size
        self._blocked = bytearray(size)
        self._cover = bytearray(size)
        self._prisms: list[tuple[float, float, float, float, float, float, list[float], list[list[float]]]] = []
        for prism in world.get("prisms", []):
            ring = [float(v) for v in prism.get("ring", [])]
            if len(ring) < 6:
                continue
            holes = [[float(v) for v in hole] for hole in prism.get("holes", [])]
            self._prisms.append((
                min(ring[::2]), max(ring[::2]), min(ring[1::2]), max(ring[1::2]),
                float(prism["base"]), float(prism["top"]), ring, holes,
            ))
        self._build_grid()
        self.roofs = self._build_roofs()
        self._path: list[int] = []
        self._planned_goal: tuple[float, float] | None = None

    @classmethod
    def from_dir(cls, directory: str, *, max_slope_deg: float = 50.0, radius: float = 2.5, height: float = 9.0) -> "MapNavigator | None":
        if not directory:
            return None
        root = Path(directory)
        try:
            world = json.loads((root / "v2" / "world.json").read_text(encoding="utf-8"))
            terrain_path = root / "v2" / "terrain.i16"
            return cls(world, terrain_path.read_bytes(), max_slope_deg=max_slope_deg, radius=radius, height=height)
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            return None

    def _height_at(self, x: float, y: float) -> float | None:
        fx = (x - self.min_x) / self.step
        fy = (y - self.min_y) / self.step
        if fx < 0 or fy < 0 or fx >= self.nx - 1 or fy >= self.ny - 1:
            return None
        ix, iy = int(fx), int(fy)
        tx, ty = fx - ix, fy - iy
        i = iy * self.nx + ix
        a = self.heights[i] * self.scale
        b = self.heights[i + 1] * self.scale
        c = self.heights[i + self.nx] * self.scale
        d = self.heights[i + self.nx + 1] * self.scale
        return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty

    def _point_blocked(self, x: float, y: float, ground: float, cell_margin: float = 0.0) -> bool:
        safety = self.radius + 0.75 + cell_margin
        for min_x, max_x, min_y, max_y, base, top, ring, holes in self._prisms:
            if top <= ground or base >= ground + self.height:
                continue
            if x < min_x - safety or x > max_x + safety or y < min_y - safety or y > max_y + safety:
                continue
            in_outer = point_in_ring(x, y, ring)
            in_hole = any(point_in_ring(x, y, hole) for hole in holes)
            if in_outer and not in_hole:
                return True
            if _ring_distance(x, y, ring) <= safety:
                return True
            if any(_ring_distance(x, y, hole) <= safety for hole in holes):
                return True
        return False

    def _id(self, ix: int, iy: int) -> int:
        return iy * self.width + ix

    def _xy(self, node: int) -> tuple[float, float]:
        iy, ix = divmod(node, self.width)
        return self.origin_x + ix * self.grid_step, self.origin_y + iy * self.grid_step

    def _build_grid(self) -> None:
        # Inflate each obstacle slightly to account for the mech radius and the coarse cell spacing.
        margin = min(self.grid_step * 0.4, self.radius + 2.0)
        for iy in range(self.height_count):
            y = self.origin_y + iy * self.grid_step
            for ix in range(self.width):
                x = self.origin_x + ix * self.grid_step
                node = self._id(ix, iy)
                ground = self._height_at(x, y)
                if ground is None:
                    self._blocked[node] = 1
                    continue
                self._ground[node] = ground
                if self._point_blocked(x, y, ground, margin):
                    self._blocked[node] = 1
        for iy in range(self.height_count):
            for ix in range(self.width):
                node = self._id(ix, iy)
                if self._blocked[node]:
                    continue
                adjacent = 0
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height_count and self._blocked[self._id(nx, ny)]:
                        adjacent += 1
                self._cover[node] = adjacent

    def _build_roofs(self) -> list[dict[str, Any]]:
        sites: list[dict[str, Any]] = []
        for min_x, max_x, min_y, max_y, _base, top, ring, holes in self._prisms:
            if max_x - min_x < 12.0 or max_y - min_y < 12.0 or top < 5.0:
                continue
            cx = sum(ring[::2]) / (len(ring) // 2)
            cy = sum(ring[1::2]) / (len(ring) // 2)
            candidates = [(cx, cy)]
            if not point_in_ring(cx, cy, ring) or any(point_in_ring(cx, cy, hole) for hole in holes):
                candidates = []
            # Vertex means can fall outside concave footprints. Sample a compact lattice
            # and keep the interior point farthest from every edge.
            for row in range(5):
                for col in range(5):
                    x = min_x + (max_x - min_x) * (col + 0.5) / 5.0
                    y = min_y + (max_y - min_y) * (row + 0.5) / 5.0
                    if point_in_ring(x, y, ring) and not any(point_in_ring(x, y, hole) for hole in holes):
                        candidates.append((x, y))
            if not candidates:
                continue
            def clearance(point: tuple[float, float]) -> float:
                x, y = point
                return min([_ring_distance(x, y, ring)] + [_ring_distance(x, y, hole) for hole in holes])
            center_x, center_y = max(candidates, key=clearance)
            clear = clearance((center_x, center_y))
            ground = self._height_at(center_x, center_y)
            if ground is None:
                continue
            gain = top - ground
            if not 8.0 <= gain <= 35.0 or clear < 5.0:
                continue
            overhead = any(
                other_ring is not ring and other_top > top + 1.0 and other_base < top + self.height
                and point_in_ring(center_x, center_y, other_ring)
                and not any(point_in_ring(center_x, center_y, hole) for hole in other_holes)
                for _, _, _, _, other_base, other_top, other_ring, other_holes in self._prisms
            )
            if overhead:
                continue
            # Merge near-identical roof prisms so only one climb target is considered.
            if any(math.hypot(center_x - r["x"], center_y - r["y"]) < 8.0 and abs(top - r["top"]) < 3.0 for r in sites):
                continue
            sites.append({"x": center_x, "y": center_y, "top": top, "gain": gain, "clearance": clear, "ring": ring, "holes": holes})
        return sites

    def nearby_roof(self, position: list[float], center: list[float], radius: float, energy: float, max_distance: float = 65.0) -> dict[str, Any] | None:
        safe_radius = max(0.0, radius - min(22.0, max(8.0, radius * 0.08)))
        best: tuple[float, dict[str, Any]] | None = None
        for site in self.roofs:
            if site["gain"] * 1.5 + 25.0 > energy:
                continue
            if math.hypot(site["x"] - center[0], site["y"] - center[1]) > safe_radius:
                continue
            boundary: tuple[float, float] | None = None
            boundary_distance = math.inf
            ring = site["ring"]
            count = len(ring) // 2
            for i in range(count):
                j = (i + 1) % count
                ax, ay, bx, by = ring[2 * i], ring[2 * i + 1], ring[2 * j], ring[2 * j + 1]
                distance = _segment_distance(position[0], position[1], ax, ay, bx, by)
                if distance < boundary_distance:
                    boundary_distance = distance
                    dx, dy = bx - ax, by - ay
                    length2 = dx * dx + dy * dy
                    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((position[0] - ax) * dx + (position[1] - ay) * dy) / length2))
                    boundary = (ax + t * dx, ay + t * dy)
            if boundary is None:
                continue
            ox, oy = boundary[0] - site["x"], boundary[1] - site["y"]
            length = max(1.0, math.hypot(ox, oy))
            takeoff = [boundary[0] + ox / length * (self.radius + 4.0), boundary[1] + oy / length * (self.radius + 4.0)]
            distance = math.hypot(takeoff[0] - position[0], takeoff[1] - position[1])
            if distance > max_distance:
                continue
            score = distance + site["gain"] * 0.6 - min(8.0, site["clearance"] * 0.5)
            candidate = {**site, "takeoff": takeoff}
            if best is None or score < best[0]:
                best = (score, candidate)
        return best[1] if best else None

    def _nearest_open(self, x: float, y: float) -> int | None:
        ix = round((x - self.origin_x) / self.grid_step)
        iy = round((y - self.origin_y) / self.grid_step)
        for radius in range(0, 9):
            choices = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if radius and max(abs(dx), abs(dy)) != radius:
                        continue
                    xx, yy = ix + dx, iy + dy
                    if 0 <= xx < self.width and 0 <= yy < self.height_count:
                        node = self._id(xx, yy)
                        if not self._blocked[node]:
                            px, py = self._xy(node)
                            choices.append(((px - x) ** 2 + (py - y) ** 2, node))
            if choices:
                return min(choices)[1]
        return None

    def _neighbors(self, node: int):
        ix, iy = node % self.width, node // self.width
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            x, y = ix + dx, iy + dy
            if not (0 <= x < self.width and 0 <= y < self.height_count):
                continue
            nxt = self._id(x, y)
            if self._blocked[nxt]:
                continue
            if dx and dy and (self._blocked[self._id(ix + dx, iy)] or self._blocked[self._id(ix, iy + dy)]):
                continue
            horizontal = self.grid_step * (math.sqrt(2.0) if dx and dy else 1.0)
            rise = self._ground[nxt] - self._ground[node]
            if abs(rise) > horizontal * self.tan_slope:
                continue
            penalty = 1.0 + min(0.75, abs(rise) / max(1.0, horizontal))
            yield nxt, horizontal * penalty

    def _astar(self, start: int, goal: int) -> list[int]:
        if start == goal:
            return [start]
        gx, gy = goal % self.width, goal // self.width
        frontier: list[tuple[float, float, int]] = [(0.0, 0.0, start)]
        came_from = {start: -1}
        cost = {start: 0.0}
        while frontier:
            _, current_cost, current = heapq.heappop(frontier)
            if current_cost != cost.get(current):
                continue
            if current == goal:
                break
            for nxt, edge in self._neighbors(current):
                new_cost = current_cost + edge
                if new_cost >= cost.get(nxt, math.inf):
                    continue
                cost[nxt] = new_cost
                came_from[nxt] = current
                nx, ny = nxt % self.width, nxt // self.width
                heuristic = self.grid_step * math.hypot(nx - gx, ny - gy)
                heapq.heappush(frontier, (new_cost + heuristic, new_cost, nxt))
        if goal not in came_from:
            return []
        path = [goal]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        return path

    def next_waypoint(self, position: list[float], goal: list[float]) -> list[float] | None:
        start = self._nearest_open(position[0], position[1])
        target = self._nearest_open(goal[0], goal[1])
        if start is None or target is None:
            return None
        goal_xy = self._xy(target)
        if not self._path or self._planned_goal is None or math.hypot(goal_xy[0] - self._planned_goal[0], goal_xy[1] - self._planned_goal[1]) > self.grid_step * 1.5:
            self._path = self._astar(start, target)
            self._planned_goal = goal_xy
        if not self._path:
            return None
        try:
            at = self._path.index(start)
            self._path = self._path[at:]
        except ValueError:
            self._path = self._astar(start, target)
        if len(self._path) < 2:
            return list(self._xy(self._path[-1])) if self._path else None
        # Aim a few cells ahead to avoid twitching from one grid center to the next.
        lookahead = min(len(self._path) - 1, 3)
        return list(self._xy(self._path[lookahead]))

    def is_blocked(self, x: float, y: float) -> bool:
        node = self._nearest_open(x, y)
        if node is None:
            return True
        px, py = self._xy(node)
        return math.hypot(px - x, py - y) > self.grid_step * 0.55

    def cover_score(self, x: float, y: float) -> int:
        node = self._nearest_open(x, y)
        return int(self._cover[node]) if node is not None else 0

    def _blocks_ground_sight(self, start: list[float], end: list[float]) -> bool:
        """Return whether a static building blocks a ground-level shot segment."""
        dx, dy = end[0] - start[0], end[1] - start[1]
        distance = math.hypot(dx, dy)
        steps = max(2, int(math.ceil(distance / max(2.0, self.grid_step))))
        start_z = float(start[2]) + self.height * 0.5 if len(start) > 2 else self.height * 0.5
        end_z = float(end[2]) + self.height * 0.5 if len(end) > 2 else self.height * 0.5
        min_x, max_x = sorted((start[0], end[0]))
        min_y, max_y = sorted((start[1], end[1]))
        for pmin_x, pmax_x, pmin_y, pmax_y, base, top, ring, holes in self._prisms:
            if top <= min(start_z, end_z) or base >= max(start_z, end_z) + self.height:
                continue
            if pmax_x < min_x or pmin_x > max_x or pmax_y < min_y or pmin_y > max_y:
                continue
            for step in range(1, steps):
                t = step / steps
                x, y = start[0] + t * dx, start[1] + t * dy
                z = start_z + t * (end_z - start_z)
                if z < base or z > top:
                    continue
                if point_in_ring(x, y, ring) and not any(point_in_ring(x, y, hole) for hole in holes):
                    return True
        return False

    def nearby_cover_goal(
        self,
        position: list[float],
        center: list[float],
        radius: float,
        max_distance: float = 72.0,
        min_cover_score: int = 2,
        threats: list[dict[str, Any]] | None = None,
    ) -> list[float] | None:
        safe_radius = max(0.0, radius - min(22.0, max(8.0, radius * 0.08)))
        best: tuple[float, int] | None = None
        ix0 = max(0, int(math.floor((position[0] - max_distance - self.origin_x) / self.grid_step)))
        ix1 = min(self.width - 1, int(math.ceil((position[0] + max_distance - self.origin_x) / self.grid_step)))
        iy0 = max(0, int(math.floor((position[1] - max_distance - self.origin_y) / self.grid_step)))
        iy1 = min(self.height_count - 1, int(math.ceil((position[1] + max_distance - self.origin_y) / self.grid_step)))
        close_threats = sorted(
            (t for t in (threats or []) if len(t.get("pos", [])) >= 2
             and math.hypot(t["pos"][0] - position[0], t["pos"][1] - position[1]) <= 180.0),
            key=lambda t: math.hypot(t["pos"][0] - position[0], t["pos"][1] - position[1]),
        )[:5]
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                node = self._id(ix, iy)
                cover = self._cover[node]
                if cover < min_cover_score or self._blocked[node]:
                    continue
                x, y = self._xy(node)
                travel = math.hypot(x - position[0], y - position[1])
                if travel > max_distance or math.hypot(x - center[0], y - center[1]) > safe_radius:
                    continue
                # Prefer nearby positions protected by more than one obstruction.
                score = travel - min(12.0, float(cover) * 3.0)
                if close_threats:
                    blocked_weight = 0.0
                    end = [x, y, self._ground[node]]
                    for threat in close_threats:
                        source = threat["pos"]
                        if self._blocks_ground_sight(source, end):
                            distance_to_threat = math.hypot(source[0] - x, source[1] - y)
                            blocked_weight += 1.0 + max(0.0, 120.0 - distance_to_threat) / 120.0
                    if blocked_weight:
                        score -= 70.0 * blocked_weight
                if best is None or score < best[0]:
                    best = (score, node)
        return list(self._xy(best[1])) if best else None


def load_navigator(directory: str, *, max_slope_deg: float = 50.0, radius: float = 2.5, height: float = 9.0) -> MapNavigator | None:
    return MapNavigator.from_dir(directory, max_slope_deg=max_slope_deg, radius=radius, height=height)

"""Quiet Iron: a placement-first, stateful mech controller."""

from __future__ import annotations

import math
from typing import Any

import mechfight as mf
from nav import MapNavigator, load_navigator, point_in_ring


def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _angle_delta(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def _normalize(x: float, y: float) -> list[float]:
    length = math.hypot(x, y)
    if length < 1e-9:
        return [0.0, 0.0]
    return [x / length, y / length]


def _lead_point(me: dict[str, Any], target: dict[str, Any], speed: float, target_height: float) -> tuple[float, float]:
    """Return yaw and pitch for a straight projectile, leading horizontal motion."""
    me_pos = me["pos"]
    target_pos = target["pos"]
    velocity = target.get("vel", [0.0, 0.0, 0.0])
    rx, ry = target_pos[0] - me_pos[0], target_pos[1] - me_pos[1]
    vx, vy = velocity[0], velocity[1]
    aa = vx * vx + vy * vy - speed * speed
    bb = 2.0 * (rx * vx + ry * vy)
    cc = rx * rx + ry * ry
    disc = bb * bb - 4.0 * aa * cc
    flight = math.sqrt(cc) / max(speed, 1.0)
    if abs(aa) > 1e-8 and disc >= 0:
        roots = [r for r in ((-bb - math.sqrt(disc)) / (2 * aa), (-bb + math.sqrt(disc)) / (2 * aa)) if r > 0]
        if roots:
            flight = min(roots)
    tx = target_pos[0] + vx * flight
    ty = target_pos[1] + vy * flight
    muzzle = me_pos[2] + me.get("stats", {}).get("muzzle_height", 7.0)
    target_z = target_pos[2] + target_height
    yaw = math.atan2(ty - me_pos[1], tx - me_pos[0])
    pitch = math.atan2(target_z - muzzle, max(1e-6, math.hypot(tx - me_pos[0], ty - me_pos[1])))
    return yaw, pitch


def _mortar_aim(me: dict[str, Any], target: dict[str, Any], speed: float, gravity: float, target_height: float) -> tuple[float, float] | None:
    """Solve the low ballistic arc, with one short lead refinement for target motion."""
    me_pos, target_pos = me["pos"], target["pos"]
    velocity = target.get("vel", [0.0, 0.0, 0.0])
    muzzle = me_pos[2] + me.get("stats", {}).get("muzzle_height", 7.0) + 2.5
    pitch = 0.0
    yaw = 0.0
    flight = math.hypot(target_pos[0] - me_pos[0], target_pos[1] - me_pos[1]) / max(speed, 1.0)
    for _ in range(3):
        dx = target_pos[0] + velocity[0] * flight - me_pos[0]
        dy = target_pos[1] + velocity[1] * flight - me_pos[1]
        horizontal = math.hypot(dx, dy)
        dz = target_pos[2] + target_height - muzzle
        if horizontal < 1e-6:
            return None
        v2 = speed * speed
        discriminant = v2 * v2 - gravity * (gravity * horizontal * horizontal + 2.0 * dz * v2)
        if discriminant < 0.0:
            return None
        tangent = (v2 - math.sqrt(discriminant)) / (gravity * horizontal)
        pitch = math.atan(tangent)
        yaw = math.atan2(dy, dx)
        flight = horizontal / max(1e-6, speed * math.cos(pitch))
    return yaw, pitch


class Bot:
    name = "Quiet Iron"

    def __init__(self, hello: mf.Hello):
        self.config = hello["config"]
        self.slot = int(hello["slot"])
        self.last_replan = -100.0
        self.last_zone_key: tuple[float, float, float] | None = None
        self.last_goal: list[float] | None = None
        self.last_threat_bearing: float | None = None
        self.last_threat_time = -100.0
        self.last_hit_time = -100.0
        self.last_attacker: int | None = None
        mech_config = self.config.get("mech", {})
        self.navigator: MapNavigator | None = load_navigator(
            hello.get("map", {}).get("dir", ""),
            max_slope_deg=float(mech_config.get("max_slope_deg", 50.0)),
            radius=float(mech_config.get("radius", 2.5)),
            height=float(mech_config.get("height", 9.0)),
        )
        self.cover_goal: list[float] | None = None
        self.cover_recheck = 0.0
        self.roof_state: dict[str, Any] | None = None
        self.roof_recheck = 0.0

    def _future_zone(self, obs: mf.Observation) -> tuple[list[float], float]:
        zone = obs["zone"]
        t = float(obs["t"])
        start, end = float(zone["shrink_start"]), float(zone["shrink_end"])
        # Move toward the announced next circle before it starts shrinking, rather than
        # waiting until the edge is already taking damage.
        alpha = 1.0 if t < start else (1.0 if end <= start or t >= end else (t - start) / (end - start))
        center = [zone["center"][i] + alpha * (zone["next_center"][i] - zone["center"][i]) for i in range(2)]
        radius = float(zone["radius"]) + alpha * (float(zone["next_radius"]) - float(zone["radius"]))
        return center, max(0.0, radius)

    def _zone_goal(self, obs: mf.Observation) -> list[float] | None:
        me = obs["self"]
        center, radius = self._future_zone(obs)
        x, y = me["pos"][:2]
        dx, dy = x - center[0], y - center[1]
        distance = math.hypot(dx, dy)
        margin = min(22.0, max(8.0, radius * 0.08))
        safe_radius = max(0.0, radius - margin)
        if distance <= safe_radius:
            return None
        if distance < 1e-6:
            return center
        return [center[0] + dx * safe_radius / distance, center[1] + dy * safe_radius / distance]

    def _evasion_goal(self, position: list[float], bearing: float, center: list[float], radius: float, goal: list[float] | None) -> list[float]:
        """Break line of fire while keeping the escape point inside the projected zone."""
        distance = 64.0
        point = [position[0] - distance * math.cos(bearing), position[1] - distance * math.sin(bearing)]
        if goal is not None:
            point = [0.75 * point[i] + 0.25 * goal[i] for i in range(2)]
        safe_radius = max(0.0, radius - min(22.0, max(8.0, radius * 0.08)))
        dx, dy = point[0] - center[0], point[1] - center[1]
        radial_distance = math.hypot(dx, dy)
        if radial_distance > safe_radius and radial_distance > 1e-6:
            point = [center[0] + dx * safe_radius / radial_distance, center[1] + dy * safe_radius / radial_distance]
        return point

    def _choose_upgrade(self, obs: mf.Observation) -> int:
        offer = obs.get("offer")
        if not offer:
            return 0
        held = {w["kind"] for w in obs["self"].get("weapons", []) if w}
        hp_ratio = float(obs["self"]["hp"]) / max(1.0, float(obs["self"]["max_hp"]))
        score = {
            "armor": 12.0,
            "plating": 10.5 + (2.0 if hp_ratio < 0.75 else 0.0),
            "nanites": 13.0 + (1.0 if hp_ratio < 0.8 else 0.0),
            "servos": 9.0,
            "jets": 8.5,
            "reactor": 8.0,
            "capacitor": 7.5,
            "sensors": 7.0,
            "gyros": 6.5,
            "gun_rate": 8.0,
            "gun_damage": 7.8,
            "gun_velocity": 7.5,
            "gun_mag": 5.5,
            "mortar_splash": 7.0,
            "mortar_reload": 7.5,
            "mortar_velocity": 6.0,
            "laser_charge": 7.0,
            "laser_damage": 8.0,
            "laser_pierce": 6.0,
            "lightning_chain": 8.0,
            "lightning_range": 7.5,
            "lightning_efficiency": 8.5,
        }
        best_index, best_score = 0, -1.0
        for index, choice in enumerate(offer.get("choices", [])):
            ident = str(choice.get("id", ""))
            entry = next((item for item in self.config.get("upgrades", []) if item.get("id") == ident), {})
            required = entry.get("requires")
            value = score.get(ident, 0.0) if required is None or required in held else -1.0
            if value > best_score:
                best_index, best_score = index, value
        return best_index

    def _target(self, obs: mf.Observation) -> dict[str, Any] | None:
        me = obs["self"]
        candidates = [m for m in obs.get("mechs", []) if m.get("kind") == "mech"]
        if not candidates:
            return None
        recent = self.last_attacker if obs["t"] - self.last_threat_time < 3.0 else None

        def key(target: dict[str, Any]) -> float:
            distance = _dist(me["pos"], target["pos"])
            hp = float(target.get("hp", target.get("max_hp", 1400)))
            threat_bonus = 45.0 if int(target["id"]) == recent else 0.0
            charge_bonus = 25.0 if target.get("charging") else 0.0
            return distance + 0.025 * hp - threat_bonus - charge_bonus

        return min(candidates, key=key)

    def _select_weapon(self, obs: mf.Observation, target: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
        me = obs["self"]
        weapons = me.get("weapons", [])
        distance = _dist(me["pos"], target["pos"])
        speed = math.hypot(*target.get("vel", [0.0, 0.0])[:2])
        for i, weapon in enumerate(weapons[:2]):
            if weapon and weapon.get("kind") == "laser" and float(weapon.get("charge", 0.0)) > 0.0:
                return i, weapon
        for index, weapon in enumerate(weapons[:2]):
            if not weapon:
                continue
            kind = weapon.get("kind")
            stats = weapon.get("stats", {})
            if kind == "lightning" and distance <= float(stats.get("range", 45.0)) - 3.0:
                return index, weapon
            if kind == "laser" and 55.0 <= distance <= min(180.0, float(stats.get("range", 280.0))) and speed <= 2.5:
                return index, weapon
            if kind == "mortar" and 38.0 <= distance <= min(125.0, float(stats.get("range", 235.0))) and speed <= 7.0:
                return index, weapon
        for index, weapon in enumerate(weapons[:2]):
            if weapon and weapon.get("kind") == "gun":
                return index, weapon
        return None

    def _weapon_fire(self, obs: mf.Observation, target: dict[str, Any] | None, aim: list[float], action: dict[str, Any], selected: tuple[int, dict[str, Any]] | None) -> list[bool]:
        weapons = obs["self"].get("weapons", [])
        fire = [False, False]
        if not target or selected is None:
            return fire
        me = obs["self"]
        distance = _dist(me["pos"], target["pos"])
        slot, weapon = selected
        kind = str(weapon.get("kind", ""))
        stats = weapon.get("stats", {})
        actual = me.get("aim", aim)
        yaw_error = abs(_angle_delta(aim[0], actual[0]))
        pitch_error = abs(aim[1] - actual[1])
        mech_stats = me.get("stats", {})
        aligned = yaw_error <= float(mech_stats.get("turn_rate", 4.0)) * 0.12 + 0.06 and pitch_error <= float(mech_stats.get("pitch_rate", 3.0)) * 0.12 + 0.08
        if kind == "gun":
            if float(weapon.get("reload", 0.0)) <= 0.0 and (float(weapon.get("ammo", 0.0)) <= 0.0 or (float(weapon.get("ammo", 0.0)) <= 2.0 and distance > 35.0)):
                action["reload"] = True
            elif distance <= float(stats.get("range", 230.0)) and aligned and float(weapon.get("ammo", 0.0)) > 0.0:
                fire[slot] = True
        elif kind == "lightning":
            cone = math.radians(float(stats.get("cone_deg", 30.0)))
            if distance <= float(stats.get("range", 45.0)) and yaw_error <= cone * 0.4 and float(me.get("energy", 0.0)) >= float(stats.get("energy", 3.0)):
                fire[slot] = True
        elif kind == "laser":
            charge = float(weapon.get("charge", 0.0))
            charge_time = float(stats.get("charge", 0.75))
            recently_hit = obs["t"] - self.last_threat_time < 0.45
            if charge > 0.0:
                fire[slot] = charge < min(0.55, charge_time * 0.8) and not recently_hit
            elif distance <= float(stats.get("range", 280.0)) and aligned and not recently_hit:
                fire[slot] = True
        elif kind == "mortar" and 22.0 <= distance <= float(stats.get("range", 235.0)) and aligned:
            fire[slot] = True
        return fire

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        t = float(obs["t"])
        x, y, z = me["pos"]
        events = obs.get("events", [])
        for event in events:
            if event.get("type") == "damaged":
                self.last_hit_time = t
                bearing = event.get("bearing")
                if bearing is not None:
                    self.last_threat_bearing = float(bearing)
                    self.last_threat_time = t
                attacker = event.get("by")
                if attacker is not None:
                    self.last_attacker = int(attacker)
        recent_sounds = [s for s in obs.get("sounds", []) if float(s.get("age", 99.0)) <= 0.8 and s.get("bearing") is not None]
        if recent_sounds:
            sound_priority = {"explosion": 0, "mortar": 1, "laser": 2, "lightning": 3, "gun": 4, "npc_gun": 5, "boost": 6, "jet": 7, "step": 8}
            sound = min(recent_sounds, key=lambda s: (float(s.get("age", 99.0)), sound_priority.get(s.get("kind"), 9)))
            sound_time = t - float(sound.get("age", 0.0))
            if sound_time > self.last_threat_time:
                self.last_threat_bearing = float(sound["bearing"])
                self.last_threat_time = sound_time
        recent_beams = [b for b in obs.get("beams", []) if float(b.get("age", 99.0)) <= 0.8 and b.get("path")]
        if recent_beams:
            beam = min(recent_beams, key=lambda b: float(b.get("age", 99.0)))
            source = beam["path"][0]
            if t - float(beam.get("age", 0.0)) > self.last_threat_time:
                self.last_threat_bearing = math.atan2(source[1] - y, source[0] - x)
                self.last_threat_time = t - float(beam.get("age", 0.0))
                self.last_attacker = int(beam["by"]) if beam.get("by") is not None else self.last_attacker

        action: mf.Action = {"tick": obs["tick"]}
        if obs.get("offer"):
            action["pick"] = self._choose_upgrade(obs)

        center, radius = self._future_zone(obs)
        goal = self._zone_goal(obs)
        target = self._target(obs)
        nearest = _dist(me["pos"], target["pos"]) if target else math.inf

        # A nearby mech is a threat first. Retreat while keeping a path toward safety.
        if target and nearest < 40.0:
            dx, dy = x - target["pos"][0], y - target["pos"][1]
            length = max(1.0, math.hypot(dx, dy))
            away = [x + 48.0 * dx / length, y + 48.0 * dy / length]
            if goal:
                away[0] = 0.8 * away[0] + 0.2 * goal[0]
                away[1] = 0.8 * away[1] + 0.2 * goal[1]
            goal = away
        if t - self.last_hit_time < 1.5 and self.last_threat_bearing is not None:
            bearing = self.last_threat_bearing
            if target is not None and nearest < 40.0:
                bearing = math.atan2(target["pos"][1] - y, target["pos"][0] - x)
            goal = self._evasion_goal([x, y], bearing, center, radius, goal)

        if self.navigator and goal is None and nearest > 60.0:
            if self.navigator.cover_score(x, y) < 2:
                if t >= self.cover_recheck:
                    self.cover_goal = self.navigator.nearby_cover_goal(me["pos"], center, radius)
                    self.cover_recheck = t + 8.0
                if self.cover_goal and _dist([x, y], self.cover_goal) > 15.0:
                    goal = self.cover_goal
                elif self.cover_goal:
                    self.cover_goal = None
                    self.cover_recheck = t + 12.0
            else:
                self.cover_goal = None
                self.cover_recheck = t

        # Opportunistic pickups only when no visible mech is close enough to punish the stop.
        weapons = me.get("weapons", [])
        held = {w["kind"] for w in weapons if w}
        if nearest > 60.0 and (len(weapons) < 2 or not weapons[1]):
            reach = float(self.config["loot"].get("pickup_radius", 4.0))
            choices = [p for p in obs.get("pickups", []) if p.get("kind") not in held and _dist(me["pos"], p["pos"]) <= 45.0]
            if choices:
                pickup = min(choices, key=lambda p: _dist(me["pos"], p["pos"]))
                if _dist(me["pos"], pickup["pos"]) <= reach:
                    action["take"] = pickup["id"]
                else:
                    goal = pickup["pos"][:2]

        # Cubes are valuable while below the ten-cube damage/health cap, but not worth
        # crossing an unsafe circle or walking into a nearby fight to collect.
        if nearest > 60.0 and float(me.get("cubes", 0.0)) < 10.0:
            safe_radius = max(0.0, radius - 18.0)
            cubes = [c for c in obs.get("cubes", []) if math.hypot(c["pos"][0] - center[0], c["pos"][1] - center[1]) <= safe_radius and _dist(me["pos"], c["pos"]) <= 24.0]
            if cubes:
                cube = min(cubes, key=lambda c: _dist(me["pos"], c["pos"]))
                if _dist(me["pos"], cube["pos"]) > 3.0:
                    goal = cube["pos"][:2]

        # Supply drops heal and award three cubes. Only take a quiet, uncontested drop
        # late, when few players remain and it is already inside the future circle.
        if nearest > 120.0 and float(obs.get("alive", 16.0)) <= 4.0:
            drops = [d for d in obs.get("drops", []) if float(d.get("lands_in", 99.0)) <= 0.0 and d.get("holder") is None and math.hypot(d["pos"][0] - center[0], d["pos"][1] - center[1]) <= max(0.0, radius - 12.0)]
            if drops:
                drop = min(drops, key=lambda d: _dist(me["pos"], d["pos"]))
                if _dist(me["pos"], drop["pos"]) > float(drop.get("radius", 8.0)) - 2.0:
                    goal = drop["pos"][:2]

        roof_override: tuple[list[float], bool, bool] | None = None
        if self.roof_state is None and self.navigator and goal is None and nearest > 120.0 and float(obs.get("alive", 16.0)) > 4.0 and self.navigator.cover_score(x, y) < 2 and t >= self.roof_recheck:
            self.roof_recheck = t + 8.0
            nearby_activity = any(float(s.get("age", 99.0)) < 1.0 and s.get("dist") == "near" for s in obs.get("sounds", []))
            if not nearby_activity:
                site = self.navigator.nearby_roof(me["pos"], center, radius, float(me.get("energy", 0.0)))
                if site:
                    self.roof_state = {"site": site, "phase": "approach", "started": t}

        if self.roof_state:
            state = self.roof_state
            site = state["site"]
            phase = state["phase"]
            safe_radius = max(0.0, radius - min(22.0, max(8.0, radius * 0.08)))
            site_safe = math.hypot(site["x"] - center[0], site["y"] - center[1]) <= safe_radius
            active_threat = (target is not None and nearest < 80.0) or self.last_hit_time >= t - 0.8 or any(float(s.get("age", 99.0)) < 0.6 and s.get("dist") == "near" for s in obs.get("sounds", []))
            if phase != "landed" and (goal is not None or not site_safe or active_threat):
                self.roof_state = None
                self.roof_recheck = t + 30.0
            elif (phase == "landed" and (goal is not None or (target and nearest < 60.0) or self.last_hit_time >= t - 0.8)):
                self.roof_state = None
                self.roof_recheck = t + 25.0
            elif phase == "approach":
                goal = site["takeoff"]
                if t - state["started"] > 15.0:
                    self.roof_state = None
                    self.roof_recheck = t + 30.0
                    goal = None
                elif math.hypot(x - site["takeoff"][0], y - site["takeoff"][1]) <= 4.5 and me.get("grounded", False):
                    state["phase"] = "ascend"
                    state["started"] = t
                    roof_override = ([0.0, 0.0], True, True)
            elif phase == "ascend":
                if z >= site["top"] + 1.5:
                    state["phase"] = "cross"
                elif t - state["started"] > 8.0 or float(me.get("energy", 0.0)) < 20.0:
                    self.roof_state = None
                    self.roof_recheck = t + 30.0
                else:
                    roof_override = ([0.0, 0.0], True, bool(me.get("grounded", False)))
            if self.roof_state and state["phase"] == "cross":
                inside = point_in_ring(x, y, site["ring"]) and not any(point_in_ring(x, y, hole) for hole in site["holes"])
                if inside and z >= site["top"] + 1.2:
                    state["phase"] = "land"
                    state["started"] = t
                else:
                    goal = [site["x"], site["y"]]
                    roof_override = (_normalize(site["x"] - x, site["y"] - y), True, False)
            if self.roof_state and state["phase"] == "land":
                if me.get("grounded", False) and abs(z - site["top"]) <= 3.0:
                    state["phase"] = "landed"
                elif t - state["started"] > 6.0 or z < site["top"] - 2.0:
                    self.roof_state = None
                    self.roof_recheck = t + 30.0
                else:
                    roof_override = ([0.0, 0.0], False, False)
            if self.roof_state and state["phase"] == "landed":
                goal = None
                roof_override = ([0.0, 0.0], False, False)

        move = [0.0, 0.0]
        if goal:
            waypoint = self.navigator.next_waypoint(me["pos"], goal) if self.navigator else None
            if self.roof_state and self.roof_state["phase"] == "approach" and waypoint is None:
                self.roof_state = None
                self.roof_recheck = t + 30.0
                goal = None
            if waypoint is None:
                waypoint = goal
            if goal:
                move = _normalize(waypoint[0] - x, waypoint[1] - y)
        if roof_override:
            move = roof_override[0]
        urgent = not bool(me.get("in_zone", True)) or (target is not None and nearest < 40.0) or self.last_hit_time >= t - 1.5
        action["move"] = move
        action["boost"] = roof_override[1] if roof_override else bool(urgent and float(me.get("energy", 0.0)) > 15.0)
        if roof_override and roof_override[2]:
            action["jump"] = True

        if target is None and t - self.last_hit_time > 0.5:
            gun = next((w for w in weapons if w and w.get("kind") == "gun"), None)
            if gun and float(gun.get("ammo", 0.0)) <= 2.0 and float(gun.get("reload", 0.0)) <= 0.0:
                action["reload"] = True

        aim = list(me.get("aim", [0.0, 0.0]))
        if target:
            selected = self._select_weapon(obs, target)
            selected_kind = selected[1].get("kind") if selected else "gun"
            target_height = float(self.config["mech"].get("height", 9.0)) * (1.0 + 0.05 * float(target.get("cubes", 0))) * 0.5
            if selected_kind == "mortar" and selected:
                solution = _mortar_aim(me, target, float(selected[1].get("stats", {}).get("speed", 72.0)), float(self.config["weapons"]["mortar"].get("gravity", 22.0)), target_height)
                if solution:
                    yaw, pitch = solution
                else:
                    selected = next(((i, w) for i, w in enumerate(weapons[:2]) if w and w.get("kind") == "gun"), None)
                    yaw, pitch = _lead_point(me, target, float(selected[1].get("stats", {}).get("speed", 252.0)) if selected else 252.0, target_height)
            elif selected_kind == "laser":
                yaw, pitch = _lead_point(me, target, 1e9, target_height)
            elif selected_kind == "lightning":
                target_pos = target["pos"]
                yaw = math.atan2(target_pos[1] - y, target_pos[0] - x)
                pitch = math.atan2(target_pos[2] + target_height - (z + float(me.get("stats", {}).get("muzzle_height", 7.0))), max(1e-6, math.hypot(target_pos[0] - x, target_pos[1] - y)))
            else:
                gun = selected[1] if selected and selected[1].get("kind") == "gun" else None
                yaw, pitch = _lead_point(me, target, float(gun.get("stats", {}).get("speed", 252.0)) if gun else 252.0, target_height)
            mech_config = self.config.get("mech", {})
            pitch = max(float(mech_config.get("min_pitch", -0.7)), min(float(mech_config.get("max_pitch", 1.3)), pitch))
            aim = [yaw, pitch]
            fire = self._weapon_fire(obs, target, aim, action, selected)
            # Long-range shots advertise your position and delay health regeneration.
            if nearest > 70.0:
                fire = [False, False]
            action["fire"] = fire
        elif self.last_threat_bearing is not None and t - self.last_threat_time <= 2.0:
            aim = [self.last_threat_bearing, 0.0]
        elif goal:
            aim = [math.atan2(goal[1] - y, goal[0] - x), 0.0]
        action["aim"] = aim

        # Avoid camping at the edge of the future circle due to small coordinate errors.
        if radius < 1.0 and math.hypot(x - center[0], y - center[1]) > 1.0:
            action["boost"] = float(me.get("energy", 0.0)) > 10.0
        return action


mf.run(Bot)

"""Quiet Iron: a placement-first, stateful mech controller."""

from __future__ import annotations

import math
import random
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
        self.rng = random.Random(int(hello.get("seed", self.slot)))
        self.evasion_side = 1.0
        self.evasion_until = -1.0
        self.last_replan = -100.0
        self.last_zone_key: tuple[float, float, float] | None = None
        self.last_goal: list[float] | None = None
        self.last_threat_bearing: float | None = None
        self.last_threat_time = -100.0
        self.last_hit_time = -100.0
        self.last_attacker: int | None = None
        self.current_target_id: int | None = None
        mech_config = self.config.get("mech", {})
        self.navigator: MapNavigator | None = load_navigator(
            hello.get("map", {}).get("dir", ""),
            max_slope_deg=float(mech_config.get("max_slope_deg", 50.0)),
            radius=float(mech_config.get("radius", 2.5)),
            height=float(mech_config.get("height", 9.0)),
        )
        self.cover_goal: list[float] | None = None
        self.cover_recheck = 0.0
        self.threat_cover_goal: list[float] | None = None
        self.threat_cover_recheck = 0.0
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

    def _evasion_side(self, t: float) -> float:
        if t >= self.evasion_until:
            if self.evasion_until >= 0.0:
                self.evasion_side *= -1.0
            else:
                self.evasion_side = self.rng.choice((-1.0, 1.0))
            self.evasion_until = t + self.rng.uniform(0.3, 0.55)
        return self.evasion_side

    def _multi_threat_evasion(self, me: dict[str, Any], threats: list[dict[str, Any]], t: float) -> list[float]:
        """Choose a lateral direction that opens distance from every aimed attack line."""
        if not threats:
            return [0.0, 0.0]
        pos = me["pos"]
        stats = me.get("stats", {})
        speed = float(stats.get("boost_speed", 19.5) if float(me.get("energy", 0.0)) > 15.0 else stats.get("walk_speed", 11.7))
        preferred_side = self._evasion_side(t)
        primary = threats[0]
        pdx = float(pos[0]) - float(primary["pos"][0])
        pdy = float(pos[1]) - float(primary["pos"][1])
        plen = max(1.0, math.hypot(pdx, pdy))
        preferred = [-preferred_side * pdy / plen, preferred_side * pdx / plen]
        best_direction = preferred
        best_score = -math.inf
        for index in range(16):
            angle = 2.0 * math.pi * index / 16.0
            direction = [math.cos(angle), math.sin(angle)]
            score = 0.001 * (direction[0] * preferred[0] + direction[1] * preferred[1])
            for threat in threats:
                source = threat["pos"]
                yaw = float(threat.get("aim", [0.0, 0.0])[0])
                line = [math.cos(yaw), math.sin(yaw)]
                rx, ry = float(pos[0]) - float(source[0]), float(pos[1]) - float(source[1])
                along = rx * line[0] + ry * line[1]
                if along < 0.0:
                    continue
                offset = rx * line[1] - ry * line[0]
                distance = math.hypot(rx, ry)
                horizon = 0.45 if threat.get("charging") else min(0.65, max(0.15, distance / 252.0))
                cross = direction[0] * line[1] - direction[1] * line[0]
                miss_distance = abs(offset + speed * horizon * cross)
                weight = 1.4 if threat.get("charging") else 1.0
                score += weight * min(60.0, miss_distance)
            if score > best_score:
                best_score, best_direction = score, direction
        return best_direction

    def _escape_goal(self, me: dict[str, Any], threats: list[dict[str, Any]]) -> list[float]:
        """Move toward the safest direction when several mechs close in."""
        x, y = float(me["pos"][0]), float(me["pos"][1])
        nearby = []
        for threat in threats:
            dx = x - float(threat["pos"][0])
            dy = y - float(threat["pos"][1])
            distance = math.hypot(dx, dy)
            if distance <= 135.0:
                vel = threat.get("vel", [0.0, 0.0])
                nearby.append((threat, distance, dx, dy, vel))
        if not nearby:
            return [x, y]

        # Project briefly ahead to avoid choosing a route the opponents' current
        # motion will immediately close. Favor opening space from the nearest mechs.
        best_score = -math.inf
        best_goal = [x, y]
        for index in range(24):
            angle = 2.0 * math.pi * index / 24.0
            dx, dy = math.cos(angle), math.sin(angle)
            goal_x, goal_y = x + 52.0 * dx, y + 52.0 * dy
            score = 0.002 * sum(dx * threat_dx + dy * threat_dy for _, _, threat_dx, threat_dy, _ in nearby)
            for threat, current, _, _, velocity in nearby:
                vx = max(-20.0, min(20.0, float(velocity[0])))
                vy = max(-20.0, min(20.0, float(velocity[1])))
                future_x = float(threat["pos"][0]) + 1.2 * vx
                future_y = float(threat["pos"][1]) + 1.2 * vy
                future = math.hypot(goal_x - future_x, goal_y - future_y)
                weight = 1.0 + max(0.0, 135.0 - current) / 45.0
                score += weight * (min(160.0, future) - min(160.0, current))
            if score > best_score:
                best_score, best_goal = score, [goal_x, goal_y]
        return best_goal

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

    def _clamp_evasion_to_zone(self, point: list[float], center: list[float], radius: float) -> list[float]:
        if radius < 40.0:
            return point
        margin = min(22.0, max(8.0, radius * 0.08))
        safe_radius = max(0.0, radius - margin)
        dx, dy = point[0] - center[0], point[1] - center[1]
        distance = math.hypot(dx, dy)
        if distance <= safe_radius or distance < 1e-6:
            return point
        scale = safe_radius / distance
        return [center[0] + dx * scale, center[1] + dy * scale]

    def _threat_cover_goal(
        self,
        me: dict[str, Any],
        threats: list[dict[str, Any]],
        center: list[float],
        radius: float,
        t: float,
    ) -> list[float] | None:
        if not self.navigator or t < self.threat_cover_recheck:
            return self.threat_cover_goal
        self.threat_cover_recheck = t + 0.5
        self.threat_cover_goal = None
        nearby = [
            actor for actor in threats
            if actor.get("kind") == "mech" and _dist(me["pos"], actor["pos"]) <= 180.0
        ]
        if not nearby:
            return None
        self.threat_cover_goal = self.navigator.nearby_cover_goal(
            me["pos"], center, radius, max_distance=96.0,
            min_cover_score=1, threats=nearby,
        )
        return self.threat_cover_goal

    def _choose_upgrade(self, obs: mf.Observation) -> int:
        offer = obs.get("offer")
        if not offer:
            return 0
        held = {w["kind"] for w in obs["self"].get("weapons", []) if w}
        hp_ratio = float(obs["self"]["hp"]) / max(1.0, float(obs["self"]["max_hp"]))
        score = {
            "armor": 12.0,
            "plating": 11.2 + (2.0 if hp_ratio < 0.75 else 0.0),
            "nanites": 10.2 + (3.5 if hp_ratio < 0.9 else 0.0),
            "servos": 9.5,
            "jets": 9.0,
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

    def _projectile_threat(self, obs: mf.Observation) -> tuple[float, list[float]] | None:
        me = obs["self"]
        pos = me["pos"]
        velocity = me.get("vel", [0.0, 0.0, 0.0])
        height = float(self.config.get("mech", {}).get("height", 9.0))
        body_radius = float(self.config.get("mech", {}).get("radius", 2.5))
        projectile_config = self.config.get("projectile", {})
        threats: list[tuple[float, float, list[float]]] = []
        for projectile in obs.get("projectiles", []):
            projectile_pos = projectile["pos"]
            projectile_velocity = projectile["vel"]
            rx = float(projectile_pos[0]) - float(pos[0])
            ry = float(projectile_pos[1]) - float(pos[1])
            vx = float(projectile_velocity[0]) - float(velocity[0])
            vy = float(projectile_velocity[1]) - float(velocity[1])
            speed_squared = vx * vx + vy * vy
            if speed_squared < 1e-6:
                continue
            impact_time = -(rx * vx + ry * vy) / speed_squared
            if impact_time < 0.0:
                impact_time = 0.0
            if impact_time > 0.65:
                continue
            closest_x = rx + vx * impact_time
            closest_y = ry + vy * impact_time
            closest_distance = math.hypot(closest_x, closest_y)
            kind = str(projectile.get("kind", "gun"))
            projectile_radius = float(projectile_config.get(f"{kind}_radius", 0.6))
            collision_distance = body_radius + projectile_radius + 1.5
            if closest_distance > collision_distance:
                continue
            projectile_z = float(projectile_pos[2]) + float(projectile_velocity[2]) * impact_time
            mech_z = float(pos[2]) + float(velocity[2]) * impact_time
            if abs(projectile_z - (mech_z + height * 0.5)) > height * 0.5 + projectile_radius:
                continue
            speed = math.sqrt(speed_squared)
            perpendicular = [-vy / speed, vx / speed]
            side_of_path = closest_x * perpendicular[0] + closest_y * perpendicular[1]
            side = -1.0 if side_of_path > 0.0 else 1.0
            if abs(side_of_path) < 0.1:
                side = 1.0 if self.slot % 2 == 0 else -1.0
            direction = [side * perpendicular[0], side * perpendicular[1]]
            threats.append((impact_time, closest_distance, direction))
        if not threats:
            return None
        impact_time, _, direction = min(threats, key=lambda threat: (threat[0], threat[1]))
        return impact_time, direction

    def _target(self, obs: mf.Observation) -> dict[str, Any] | None:
        me = obs["self"]
        actors = obs.get("mechs", [])
        candidates = [m for m in actors if m.get("kind") == "mech"]
        recent = self.last_attacker if obs["t"] - self.last_threat_time < 3.0 else None

        def distance(target: dict[str, Any]) -> float:
            return _dist(me["pos"], target["pos"])

        npc_range = float(self.config.get("npc", {}).get("aggro_range", 45.0))
        npcs = [m for m in actors if m.get("kind") == "npc" and (distance(m) <= npc_range or int(m["id"]) == recent)]
        mech = min(candidates, key=distance) if candidates else None
        npc = min(npcs, key=distance) if npcs else None

        def key(target: dict[str, Any]) -> float:
            target_distance = distance(target)
            hp = float(target.get("hp", target.get("max_hp", 1400)))
            threat_bonus = 45.0 if int(target["id"]) == recent else 0.0
            charge_bonus = 25.0 if target.get("charging") else 0.0
            # Prefer a damaged mech enough to convert a good shot into a kill and cubes.
            return target_distance + 0.05 * hp - threat_bonus - charge_bonus

        if npc and (mech is None or distance(mech) >= 60.0):
            selected = npc
        elif mech and distance(mech) <= 170.0:
            selected = min(candidates, key=key)
        else:
            crate = min(
                (
                    c for c in obs.get("crates", [])
                    if float(c.get("hp", 0.0)) > 0.0
                    and _dist(me["pos"], c["pos"]) <= 100.0
                    and float(me.get("cubes", 0.0)) < 6.0
                ),
                key=lambda c: _dist(me["pos"], c["pos"]),
                default=None,
            )
            if crate is None:
                self.current_target_id = None
                return None
            selected = {**crate, "kind": "crate"}

        current = next((m for m in actors if int(m["id"]) == self.current_target_id and m.get("kind") == selected.get("kind")), None)
        if current and key(current) <= key(selected) + 20.0 and not (
            current.get("kind") == "npc" and selected.get("kind") == "mech" and distance(selected) < 60.0
        ):
            selected = current
        self.current_target_id = int(selected["id"])
        return selected

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
        warning_sounds = [
            s for s in recent_sounds
            if s.get("kind") in ("laser_charge", "mortar") and float(s.get("age", 99.0)) <= 0.45
        ]
        warning_sound = min(warning_sounds, key=lambda s: float(s.get("age", 99.0))) if warning_sounds else None
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
        projectile_threat = self._projectile_threat(obs)
        incoming_attack = False
        incoming_threat: dict[str, Any] | None = None
        incoming_threats: list[dict[str, Any]] = []
        incoming_aim_error = math.inf
        incoming_charging = False
        threat_score = math.inf
        for actor in obs.get("mechs", []):
            if actor.get("kind") != "mech":
                continue
            threat_distance = _dist(me["pos"], actor["pos"])
            if threat_distance >= 150.0:
                continue
            actor_aim = actor.get("aim")
            aim_error = math.inf
            if actor_aim:
                toward_us = math.atan2(y - actor["pos"][1], x - actor["pos"][0])
                aim_error = abs(_angle_delta(toward_us, float(actor_aim[0])))
            charging = bool(actor.get("charging"))
            if aim_error < 0.35:
                incoming_threats.append(actor)
                score = (0.0 if charging else 1.0) + aim_error + threat_distance * 0.001
                if score < threat_score:
                    incoming_threat = actor
                    incoming_aim_error = aim_error
                    incoming_charging = charging
                    threat_score = score

        if projectile_threat:
            impact_time, direction = projectile_threat
            dodge = [x + 36.0 * direction[0], y + 36.0 * direction[1]]
            if goal:
                dodge[0] = 0.8 * dodge[0] + 0.2 * goal[0]
                dodge[1] = 0.8 * dodge[1] + 0.2 * goal[1]
            goal = self._clamp_evasion_to_zone(dodge, center, radius)
            incoming_attack = impact_time < 0.55
        # A nearby mech is a threat first. Retreat while keeping a path toward safety.
        elif target and target.get("kind") == "mech" and nearest < 60.0:
            cover = self._threat_cover_goal(me, obs.get("mechs", []), center, radius, t)
            if cover:
                goal = cover
            else:
                dx, dy = x - target["pos"][0], y - target["pos"][1]
                length = max(1.0, math.hypot(dx, dy))
                away = [x + 48.0 * dx / length, y + 48.0 * dy / length]
                if goal:
                    away[0] = 0.8 * away[0] + 0.2 * goal[0]
                    away[1] = 0.8 * away[1] + 0.2 * goal[1]
                goal = self._clamp_evasion_to_zone(away, center, radius)
        elif incoming_threat:
            incoming_attack = incoming_aim_error < 0.16 or incoming_charging
            cover = self._threat_cover_goal(me, obs.get("mechs", []), center, radius, t)
            if cover:
                goal = cover
            else:
                # Move across a visible attack line before the shot lands. Keep half
                # of an existing zone objective so evasion does not undo routing.
                direction = self._multi_threat_evasion(me, incoming_threats, t)
                evade = [x + 36.0 * direction[0], y + 36.0 * direction[1]]
                if goal:
                    evade[0] = 0.5 * evade[0] + 0.5 * goal[0]
                    evade[1] = 0.5 * evade[1] + 0.5 * goal[1]
                goal = self._clamp_evasion_to_zone(evade, center, radius)
        elif any(actor.get("kind") == "mech" and _dist(me["pos"], actor["pos"]) < 110.0 for actor in obs.get("mechs", [])):
            # Don't wait for the nearest opponent to finish aiming before leaving.
            nearby_mechs = [actor for actor in obs.get("mechs", []) if actor.get("kind") == "mech"]
            cover = self._threat_cover_goal(me, nearby_mechs, center, radius, t)
            if cover:
                goal = cover
            else:
                escape = self._escape_goal(me, nearby_mechs)
                if goal:
                    escape[0] = 0.8 * escape[0] + 0.2 * goal[0]
                    escape[1] = 0.8 * escape[1] + 0.2 * goal[1]
                goal = self._clamp_evasion_to_zone(escape, center, radius)
        elif warning_sound:
            # An audible charge or mortar launch can reveal an attacker outside
            # vision. Cross that firing line briefly while preserving zone travel.
            bearing = float(warning_sound["bearing"])
            side = self._evasion_side(t)
            evade = [x - side * 36.0 * math.sin(bearing), y + side * 36.0 * math.cos(bearing)]
            if goal:
                evade[0] = 0.65 * evade[0] + 0.35 * goal[0]
                evade[1] = 0.65 * evade[1] + 0.35 * goal[1]
            goal = self._clamp_evasion_to_zone(evade, center, radius)
            incoming_attack = (
                float(warning_sound.get("age", 99.0)) < 0.35
                and (warning_sound.get("kind") == "laser_charge" or warning_sound.get("dist") == "near")
            )
        elif target is None and self.last_hit_time >= t - 0.6 and self.last_threat_bearing is not None and goal is None:
            goal = [x - 42.0 * math.cos(self.last_threat_bearing), y - 42.0 * math.sin(self.last_threat_bearing)]

        if self.navigator and goal is None and nearest > 60.0:
            cover = self.navigator.cover_score(x, y)
            if t >= self.cover_recheck:
                min_cover = 2 if cover < 2 else cover + 1
                max_distance = 72.0 if cover < 2 else 48.0
                self.cover_goal = self.navigator.nearby_cover_goal(
                    me["pos"], center, radius, max_distance=max_distance, min_cover_score=min_cover,
                )
                self.cover_recheck = t + (8.0 if cover < 2 else 18.0)
            if self.cover_goal and _dist([x, y], self.cover_goal) > 15.0:
                goal = self.cover_goal
            elif self.cover_goal:
                self.cover_goal = None
                self.cover_recheck = t + (12.0 if cover < 2 else 24.0)

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

        # A landed drop can restore a damaged mech and provide three upgrade
        # cubes. Take a nearby one during a lull, but never chase it outside the
        # projected safe circle or toward a visible fight.
        drop_capture = False
        damaged = float(me["max_hp"]) - float(me["hp"])
        if nearest > 100.0 and t - self.last_hit_time > 2.5 and (damaged >= 300.0 or float(me.get("cubes", 0.0)) < 4.0):
            my_id = int(me.get("id", -1))
            safe_radius = max(0.0, radius - min(22.0, max(8.0, radius * 0.08)))
            drops = [
                d for d in obs.get("drops", [])
                if float(d.get("lands_in", 99.0)) <= 0.0
                and d.get("holder") in (None, my_id)
                and math.hypot(d["pos"][0] - center[0], d["pos"][1] - center[1]) <= safe_radius
                and _dist(me["pos"], d["pos"]) <= 75.0
            ]
            if drops:
                drop = min(drops, key=lambda d: _dist(me["pos"], d["pos"]))
                if _dist(me["pos"], drop["pos"]) > float(drop.get("radius", 8.0)) - 2.0:
                    goal = drop["pos"][:2]
                else:
                    # Remain inside the capture ring for its full hold time.
                    goal = None
                    drop_capture = True

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
        urgent = not bool(me.get("in_zone", True)) or (target is not None and nearest < 40.0) or self.last_hit_time >= t - 0.6 or incoming_attack
        action["move"] = move
        action["boost"] = roof_override[1] if roof_override else bool(urgent and float(me.get("energy", 0.0)) > 15.0)
        if drop_capture and not urgent:
            action["move"] = [0.0, 0.0]
            action["boost"] = False
        if roof_override and roof_override[2]:
            action["jump"] = True

        if target is None and t - self.last_hit_time > 0.5:
            gun = next((w for w in weapons if w and w.get("kind") == "gun"), None)
            if gun and float(gun.get("ammo", 0.0)) <= 2.0 and float(gun.get("reload", 0.0)) <= 0.0:
                action["reload"] = True

        aim = list(me.get("aim", [0.0, 0.0]))
        if target:
            if target.get("kind") == "crate":
                selected = next(((i, w) for i, w in enumerate(weapons[:2]) if w and w.get("kind") == "gun"), None)
            else:
                selected = self._select_weapon(obs, target)
            selected_kind = selected[1].get("kind") if selected else "gun"
            if target.get("kind") == "npc":
                target_height = float(self.config.get("npc", {}).get("height", 3.5)) * 0.5
            elif target.get("kind") == "crate":
                target_height = float(self.config.get("crates", {}).get("height", 2.4)) * 0.5
            else:
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
            action["fire"] = self._weapon_fire(obs, target, aim, action, selected)
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

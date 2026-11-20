"""Quiet Iron: a placement-first, stateful mech controller."""

from __future__ import annotations

import math
from typing import Any

import mechfight as mf


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
        self.last_attacker: int | None = None

    def _future_zone(self, obs: mf.Observation) -> tuple[list[float], float]:
        zone = obs["zone"]
        t = float(obs["t"])
        start, end = float(zone["shrink_start"]), float(zone["shrink_end"])
        alpha = 0.0 if t <= start else (1.0 if end <= start or t >= end else (t - start) / (end - start))
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

    def _choose_upgrade(self, obs: mf.Observation) -> int:
        offer = obs.get("offer")
        if not offer:
            return 0
        held = {w["kind"] for w in obs["self"].get("weapons", []) if w}
        hp_ratio = float(obs["self"]["hp"]) / max(1.0, float(obs["self"]["max_hp"]))
        score = {
            "armor": 12.0,
            "plating": 10.5 + (2.0 if hp_ratio < 0.75 else 0.0),
            "nanites": 9.5 + (1.0 if hp_ratio < 0.8 else 0.0),
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

    def _weapon_fire(self, obs: mf.Observation, target: dict[str, Any] | None, aim: list[float], action: dict[str, Any]) -> list[bool]:
        weapons = obs["self"].get("weapons", [])
        fire = [False, False]
        if not target:
            return fire
        me = obs["self"]
        distance = _dist(me["pos"], target["pos"])
        target_height = float(self.config["mech"].get("height", 9.0)) * (1.0 + 0.05 * float(target.get("cubes", 0))) * 0.5
        gun_slot = next((i for i, w in enumerate(weapons) if w and w.get("kind") == "gun"), None)
        if gun_slot is not None:
            gun = weapons[gun_slot]
            actual = me.get("aim", aim)
            yaw_error = abs(_angle_delta(aim[0], actual[0]))
            pitch_error = abs(aim[1] - actual[1])
            aligned = yaw_error <= float(me.get("stats", {}).get("turn_rate", 4.0)) * 0.12 + 0.06 and pitch_error <= float(me.get("stats", {}).get("pitch_rate", 3.0)) * 0.12 + 0.08
            if float(gun.get("reload", 0.0)) > 0:
                pass
            elif float(gun.get("ammo", 0.0)) <= 2 and distance > 35:
                action["reload"] = True
            elif distance <= float(gun.get("stats", {}).get("range", self.config["weapons"]["gun"].get("range", 230.0))) and aligned:
                fire[gun_slot] = True

        for index, weapon in enumerate(weapons[:2]):
            if not weapon or weapon.get("kind") == "gun":
                continue
            kind = weapon.get("kind")
            stats = weapon.get("stats", {})
            if kind == "lightning":
                yaw_error = abs(_angle_delta(aim[0], me.get("aim", aim)[0]))
                cone = math.radians(float(stats.get("cone_deg", 30.0)))
                if distance <= float(stats.get("range", 45.0)) and yaw_error <= cone * 0.45 and float(me.get("energy", 0.0)) >= float(stats.get("energy", 3.0)):
                    fire[index] = True
            elif kind == "laser":
                charging = float(weapon.get("charge", 0.0)) > 0.0
                danger = any(e.get("type") == "damaged" and obs["t"] - self.last_threat_time < 0.5 for e in obs.get("events", []))
                if charging:
                    fire[index] = float(weapon.get("charge", 0.0)) < min(0.55, float(stats.get("charge", 0.75))) and not danger
                elif distance <= float(stats.get("range", 280.0)) and not danger:
                    yaw_error = abs(_angle_delta(aim[0], me.get("aim", aim)[0]))
                    if yaw_error < 0.18:
                        fire[index] = True
            elif kind == "mortar":
                if 22.0 <= distance <= float(stats.get("range", 235.0)):
                    fire[index] = True
        return fire

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        t = float(obs["t"])
        x, y, z = me["pos"]
        events = obs.get("events", [])
        for event in events:
            if event.get("type") == "damaged":
                bearing = event.get("bearing")
                if bearing is not None:
                    self.last_threat_bearing = float(bearing)
                    self.last_threat_time = t
                attacker = event.get("by")
                if attacker is not None:
                    self.last_attacker = int(attacker)

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

        move = [0.0, 0.0]
        if goal:
            move = _normalize(goal[0] - x, goal[1] - y)
        urgent = not bool(me.get("in_zone", True)) or (target is not None and nearest < 40.0)
        action["move"] = move
        action["boost"] = bool(urgent and float(me.get("energy", 0.0)) > 15.0)

        aim = list(me.get("aim", [0.0, 0.0]))
        if target:
            gun = next((w for w in weapons if w and w.get("kind") == "gun"), None)
            if gun:
                yaw, pitch = _lead_point(me, target, float(gun.get("stats", {}).get("speed", 252.0)), float(self.config["mech"].get("height", 9.0)) * (1.0 + 0.05 * float(target.get("cubes", 0))) * 0.5)
            else:
                target_pos = target["pos"]
                yaw = math.atan2(target_pos[1] - y, target_pos[0] - x)
                pitch = math.atan2(target_pos[2] + 4.5 - (z + float(me.get("stats", {}).get("muzzle_height", 7.0))), max(1e-6, math.hypot(target_pos[0] - x, target_pos[1] - y)))
            aim = [yaw, pitch]
            action["fire"] = self._weapon_fire(obs, target, aim, action)
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

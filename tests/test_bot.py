import math
import unittest
from array import array

from bot import Bot, _angle_delta, _lead_point, _mortar_aim, _normalize
from nav import MapNavigator, point_in_ring


class GeometryTests(unittest.TestCase):
    def test_angle_delta_wraps_at_pi(self):
        self.assertAlmostEqual(_angle_delta(-math.pi + 0.1, math.pi - 0.1), 0.2)

    def test_normalize_handles_zero_and_direction(self):
        self.assertEqual(_normalize(0.0, 0.0), [0.0, 0.0])
        x, y = _normalize(3.0, 4.0)
        self.assertAlmostEqual(math.hypot(x, y), 1.0)
        self.assertAlmostEqual(x, 0.6)

    def test_intercept_leads_a_moving_target(self):
        me = {"pos": [0.0, 0.0, 0.0], "stats": {"muzzle_height": 7.0}}
        target = {"pos": [100.0, 0.0, 0.0], "vel": [0.0, 10.0, 0.0]}
        yaw, pitch = _lead_point(me, target, 100.0, 4.5)
        self.assertGreater(yaw, 0.0)
        self.assertLess(yaw, 0.2)
        self.assertLess(pitch, 0.0)

    def test_mortar_solution_is_finite_for_reachable_target(self):
        me = {"pos": [0.0, 0.0, 0.0], "stats": {"muzzle_height": 7.0}}
        target = {"pos": [100.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0]}
        yaw, pitch = _mortar_aim(me, target, 72.0, 22.0, 4.5)
        self.assertAlmostEqual(yaw, 0.0)
        self.assertTrue(math.isfinite(pitch))


class PolicyTests(unittest.TestCase):
    def make_bot(self):
        hello = {
            "slot": 0,
            "map": {"dir": ""},
            "config": {
                "mech": {},
                "weapons": {"gun": {}, "mortar": {"gravity": 22.0}},
                "loot": {"pickup_radius": 4.0},
                "upgrades": [],
            },
        }
        return Bot(hello)

    def test_zone_controller_moves_early_to_announced_circle(self):
        bot = self.make_bot()
        obs = {
            "t": 5.0,
            "zone": {
                "center": [0.0, 0.0], "radius": 100.0,
                "next_center": [50.0, 0.0], "next_radius": 50.0,
                "shrink_start": 10.0, "shrink_end": 20.0,
            },
        }
        center, radius = bot._future_zone(obs)
        self.assertEqual(center, [50.0, 0.0])
        self.assertEqual(radius, 50.0)

    def test_empty_gun_reloads_even_in_close_combat(self):
        bot = self.make_bot()
        gun = {"kind": "gun", "ammo": 0.0, "reload": 0.0, "stats": {"range": 230.0}}
        me = {"pos": [0.0, 0.0, 0.0], "aim": [0.0, 0.0], "stats": {"turn_rate": 4.0, "pitch_rate": 3.0}, "weapons": [gun, None]}
        target = {"id": 1, "pos": [20.0, 0.0, 0.0]}
        obs = {"self": me, "t": 1.0, "events": []}
        action = {}
        fire = bot._weapon_fire(obs, target, [0.0, 0.0], action, (0, gun))
        self.assertEqual(fire, [False, False])
        self.assertTrue(action["reload"])

    def test_recent_damage_triggers_evasion_even_with_a_visible_target(self):
        bot = self.make_bot()
        bot.last_hit_time = 10.0
        bot.last_threat_time = 10.0
        bot.last_threat_bearing = 0.0
        gun = {"kind": "gun", "ammo": 12.0, "reload": 0.0, "stats": {"range": 230.0, "speed": 252.0}}
        me = {
            "pos": [0.0, 0.0, 0.0], "aim": [0.0, 0.0], "stats": {"turn_rate": 4.0, "pitch_rate": 3.0, "muzzle_height": 7.0},
            "hp": 800.0, "max_hp": 1400.0, "energy": 100.0, "grounded": True, "in_zone": True, "cubes": 0,
            "weapons": [gun, None],
        }
        target = {"id": 1, "kind": "mech", "pos": [100.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0], "hp": 1400.0}
        obs = {
            "t": 10.1, "tick": 101, "alive": 16.0, "self": me, "mechs": [target], "events": [], "sounds": [], "beams": [],
            "pickups": [], "cubes": [], "drops": [], "offer": None,
            "zone": {"center": [0.0, 0.0], "radius": 100.0, "next_center": [0.0, 0.0], "next_radius": 100.0, "shrink_start": 20.0, "shrink_end": 30.0},
        }

        action = bot.act(obs)

        self.assertLess(action["move"][0], 0.0)
        self.assertEqual(action["fire"], [False, False])

    def test_evasion_goal_stays_in_projected_zone(self):
        bot = self.make_bot()

        goal = bot._evasion_goal([0.0, 0.0], 0.0, [0.0, 0.0], 60.0, None)

        self.assertAlmostEqual(goal[0], -52.0)
        self.assertAlmostEqual(goal[1], 0.0)

    def test_upgrade_choice_prioritizes_health_regeneration(self):
        bot = self.make_bot()
        obs = {
            "self": {"hp": 800.0, "max_hp": 1400.0, "weapons": [{"kind": "gun"}, None]},
            "offer": {"choices": [{"id": "armor"}, {"id": "nanites"}, {"id": "plating"}]},
        }

        self.assertEqual(bot._choose_upgrade(obs), 1)


class NavigationTests(unittest.TestCase):
    def make_navigator(self):
        nx = ny = 40
        world = {
            "bounds": {"minX": 0.0, "maxX": 39.0, "minY": 0.0, "maxY": 39.0},
            "terrain": {"minX": 0.0, "minY": 0.0, "step": 1.0, "nx": nx, "ny": ny, "scale": 0.01},
            "prisms": [
                {"base": 0.0, "top": 10.0, "ring": [18.0, 0.0, 22.0, 0.0, 22.0, 28.0, 18.0, 28.0], "holes": []},
                {"base": 0.0, "top": 10.0, "ring": [6.0, 0.0, 10.0, 0.0, 10.0, 28.0, 6.0, 28.0], "holes": []},
            ],
        }
        heights = array("h", [0] * (nx * ny)).tobytes()
        return MapNavigator(world, heights, grid_step=4.0, radius=1.0, height=3.0)

    def test_point_in_ring_handles_inside_and_outside(self):
        square = [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]
        self.assertTrue(point_in_ring(5.0, 5.0, square))
        self.assertFalse(point_in_ring(12.0, 5.0, square))

    def test_astar_routes_around_wall(self):
        nav = self.make_navigator()
        start = nav._nearest_open(2.0, 20.0)
        goal = nav._nearest_open(38.0, 20.0)
        self.assertIsNotNone(start)
        self.assertIsNotNone(goal)
        path = nav._astar(start, goal)
        self.assertTrue(path)
        ys = [nav._xy(node)[1] for node in path]
        self.assertGreater(max(ys), 28.0)

    def test_blocked_goal_terminates_at_nearest_walkable_node(self):
        nav = self.make_navigator()
        start = nav._nearest_open(2.0, 20.0)
        waypoint = nav.next_waypoint(list(nav._xy(start)), [20.0, 15.0])
        self.assertIsNotNone(waypoint)
        self.assertNotEqual(waypoint, [20.0, 15.0])

    def test_roof_candidates_respect_energy_reserve(self):
        nx = ny = 40
        world = {
            "bounds": {"minX": 0.0, "maxX": 39.0, "minY": 0.0, "maxY": 39.0},
            "terrain": {"minX": 0.0, "minY": 0.0, "step": 1.0, "nx": nx, "ny": ny, "scale": 0.01},
            "prisms": [{
                "base": 0.0,
                "top": 15.0,
                "ring": [10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0],
                "holes": [],
            }],
        }
        heights = array("h", [0] * (nx * ny)).tobytes()
        nav = MapNavigator(world, heights, grid_step=4.0, radius=1.0, height=3.0)
        site = nav.nearby_roof([5.0, 20.0], [20.0, 20.0], 40.0, 100.0)
        self.assertIsNotNone(site)
        self.assertEqual(site["gain"], 15.0)
        self.assertIsNone(nav.nearby_roof([5.0, 20.0], [20.0, 20.0], 40.0, 40.0))

    def test_roof_candidate_under_an_overhang_is_rejected(self):
        nx = ny = 40
        world = {
            "bounds": {"minX": 0.0, "maxX": 39.0, "minY": 0.0, "maxY": 39.0},
            "terrain": {"minX": 0.0, "minY": 0.0, "step": 1.0, "nx": nx, "ny": ny, "scale": 0.01},
            "prisms": [
                {"base": 0.0, "top": 15.0, "ring": [10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0], "holes": []},
                {"base": 15.0, "top": 25.0, "ring": [16.0, 16.0, 24.0, 16.0, 24.0, 24.0, 16.0, 24.0], "holes": []},
            ],
        }
        heights = array("h", [0] * (nx * ny)).tobytes()
        nav = MapNavigator(world, heights, grid_step=4.0, radius=1.0, height=3.0)
        self.assertIsNone(nav.nearby_roof([5.0, 20.0], [20.0, 20.0], 40.0, 100.0))


if __name__ == "__main__":
    unittest.main()

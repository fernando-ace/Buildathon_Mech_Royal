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
                "npc": {"aggro_range": 45.0, "height": 3.5},
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

    def test_engages_a_close_drone_without_retreating_from_it(self):
        bot = self.make_bot()
        gun = {"kind": "gun", "ammo": 12.0, "reload": 0.0, "stats": {"range": 230.0, "speed": 252.0}}
        me = {
            "pos": [0.0, 0.0, 0.0], "aim": [0.0, 0.0],
            "stats": {"turn_rate": 4.0, "pitch_rate": 3.0, "muzzle_height": 7.0},
            "hp": 1400.0, "max_hp": 1400.0, "energy": 100.0, "grounded": False,
            "in_zone": True, "weapons": [gun, None], "cubes": 0,
        }
        drone = {"id": 1001, "kind": "npc", "pos": [30.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0], "hp": 350.0}
        obs = {
            "tick": 1, "t": 0.1, "alive": 16.0, "self": me, "mechs": [drone],
            "events": [], "sounds": [], "beams": [], "pickups": [], "cubes": [], "drops": [], "offer": None,
            "zone": {"center": [0.0, 0.0], "radius": 100.0, "next_center": [0.0, 0.0], "next_radius": 100.0,
                     "shrink_start": 20.0, "shrink_end": 30.0},
        }

        action = bot.act(obs)

        self.assertEqual(action["move"], [0.0, 0.0])
        self.assertTrue(action["fire"][0])

    def test_close_mech_keeps_priority_over_a_drone(self):
        bot = self.make_bot()
        mech = {"id": 2, "kind": "mech", "pos": [30.0, 0.0, 0.0]}
        drone = {"id": 1001, "kind": "npc", "pos": [20.0, 0.0, 0.0]}
        obs = {"t": 1.0, "self": {"pos": [0.0, 0.0, 0.0]}, "mechs": [mech, drone]}

        self.assertEqual(bot._target(obs)["kind"], "mech")

    def test_target_selection_prefers_a_damaged_mech_for_a_finishing_shot(self):
        bot = self.make_bot()
        healthy = {"id": 2, "kind": "mech", "pos": [30.0, 0.0, 0.0], "hp": 1400.0}
        weakened = {"id": 3, "kind": "mech", "pos": [80.0, 0.0, 0.0], "hp": 200.0}
        obs = {"t": 1.0, "self": {"pos": [0.0, 0.0, 0.0]}, "mechs": [healthy, weakened]}

        self.assertEqual(bot._target(obs)["id"], 3)

    def test_target_selection_keeps_an_equally_good_target_to_steady_aim(self):
        bot = self.make_bot()
        first = {"id": 2, "kind": "mech", "pos": [31.0, 0.0, 0.0], "hp": 1400.0}
        second = {"id": 3, "kind": "mech", "pos": [30.0, 0.0, 0.0], "hp": 1400.0}
        obs = {"t": 1.0, "self": {"pos": [0.0, 0.0, 0.0]}, "mechs": [first, second]}
        bot.current_target_id = 2

        self.assertEqual(bot._target(obs)["id"], 2)

    def test_strafes_after_a_ranged_hit_instead_of_standing_still(self):
        bot = self.make_bot()
        me = {
            "pos": [0.0, 0.0, 0.0], "aim": [0.0, 0.0],
            "stats": {"turn_rate": 4.0, "pitch_rate": 3.0, "muzzle_height": 7.0},
            "hp": 1000.0, "max_hp": 1400.0, "energy": 100.0, "grounded": False,
            "in_zone": True, "weapons": [None, None], "cubes": 0,
        }
        attacker = {"id": 2, "kind": "mech", "pos": [60.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0], "hp": 1400.0}
        obs = {
            "tick": 1, "t": 1.0, "alive": 12.0, "self": me, "mechs": [attacker],
            "events": [{"type": "damaged", "bearing": 0.0, "by": 2}],
            "sounds": [], "beams": [], "pickups": [], "cubes": [], "drops": [], "offer": None,
            "zone": {"center": [0.0, 0.0], "radius": 100.0, "next_center": [0.0, 0.0], "next_radius": 100.0,
                     "shrink_start": 20.0, "shrink_end": 30.0},
        }

        move = bot.act(obs)["move"]

        self.assertAlmostEqual(move[0], 0.0, delta=0.01)
        self.assertAlmostEqual(abs(move[1]), 1.0, delta=0.01)

    def test_does_not_leave_the_zone_for_an_early_supply_drop(self):
        bot = self.make_bot()
        me = {
            "pos": [0.0, 0.0, 0.0], "aim": [0.0, 0.0], "stats": {"walk_speed": 11.7},
            "hp": 500.0, "max_hp": 1400.0, "energy": 100.0, "grounded": False,
            "in_zone": True, "weapons": [None, None], "cubes": 0,
        }
        obs = {
            "tick": 300, "t": 30.0, "alive": 16.0, "self": me, "mechs": [],
            "events": [], "sounds": [], "beams": [], "pickups": [], "cubes": [],
            "drops": [{"pos": [80.0, 0.0, 0.0], "lands_in": 10.0, "holder": None, "radius": 8.0}],
            "offer": None,
            "zone": {"center": [0.0, 0.0], "radius": 200.0, "next_center": [0.0, 0.0], "next_radius": 100.0,
                     "shrink_start": 40.0, "shrink_end": 50.0},
        }

        self.assertEqual(bot.act(obs)["move"], [0.0, 0.0])


class NavigationTests(unittest.TestCase):
    def make_navigator(self):
        nx = ny = 40
        world = {
            "bounds": {"minX": 0.0, "maxX": 39.0, "minY": 0.0, "maxY": 39.0},
            "terrain": {"minX": 0.0, "minY": 0.0, "step": 1.0, "nx": nx, "ny": ny, "scale": 0.01},
            "prisms": [{
                "base": 0.0,
                "top": 10.0,
                "ring": [18.0, 0.0, 22.0, 0.0, 22.0, 28.0, 18.0, 28.0],
                "holes": [],
            }],
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

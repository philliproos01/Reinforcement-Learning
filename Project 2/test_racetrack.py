import random
import unittest

from racetrack import ACTION_DELTAS, Track, route_initialized_q, episode


class RacetrackTests(unittest.TestCase):
    def test_digitized_tracks(self):
        left, right = Track("left"), Track("right")
        self.assertEqual((left.width, left.height, len(left.cells)), (17, 32, 298))
        self.assertEqual((right.width, right.height, len(right.cells)), (32, 30, 515))
        self.assertEqual((len(left.starts), len(left.finishes)), (6, 6))
        self.assertEqual((len(right.starts), len(right.finishes)), (23, 9))

    def test_velocity_limits_and_zero_at_start(self):
        track = Track("left")
        self.assertIn(ACTION_DELTAS.index((0, 0)), track.legal[0, 0, True])
        self.assertNotIn(ACTION_DELTAS.index((-1, 0)), track.legal[1, 0, False])
        self.assertNotIn(ACTION_DELTAS.index((1, 0)), track.legal[4, 0, False])

    def test_swept_path_hits_finish_and_outer_wall(self):
        track = Track("left")
        self.assertEqual(track.motion[13, 29, 4, 0].kind, "finish")
        self.assertEqual(track.motion[8, 10, 4, 4].kind, "crash")
        # The terminal crossing is detected before the projected endpoint.
        self.assertEqual((track.motion[13, 29, 4, 0].x,
                          track.motion[13, 29, 4, 0].y), (16, 29))
        right = Track("right")
        self.assertIn((1, 3), right.cells)
        # At this exact corner the diagonal destination is on track, but a
        # side cell is outside; the swept path must still crash.
        self.assertEqual(right.motion[0, 2, 1, 1].kind, "crash")

    def test_action_failure_and_crash_reset(self):
        track = Track("left")
        action = ACTION_DELTAS.index((1, 1))
        state = (3, 0, 0, 0)
        next_state, done, crashed = track.step(state, action, random.Random(1), 1.0)
        self.assertEqual(next_state, state)
        self.assertFalse(done or crashed)
        next_state, done, crashed = track.step((8, 0, 0, 0), action,
                                               random.Random(1), 0.0)
        self.assertIn(next_state[:2], track.starts)
        self.assertEqual(next_state[2:], (0, 0))
        self.assertTrue(crashed)
        self.assertFalse(done)

    def test_route_initialization_reaches_finish_from_every_start(self):
        for name in ("left", "right"):
            track = Track(name)
            q = route_initialized_q(track)
            for start in track.starts:
                _, steps, crashes, done, _ = episode(
                    track, q, random.Random(7), 0.0, 100,
                    failure_probability=0.0, start=start,
                )
                self.assertTrue(done, (name, start))
                self.assertEqual(crashes, 0, (name, start))
                self.assertLess(steps, 20)


if __name__ == "__main__":
    unittest.main()

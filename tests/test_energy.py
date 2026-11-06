"""Check energy accounting against simple analytical states, independently of dynamics."""
import math
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp
import newton.viewer

from cable_pendulum import ENERGY_COLUMNS, Pendulum


class EnergyAccounting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        wp.set_device('cpu')
        cls.viewer = newton.viewer.ViewerNull(num_frames=1)
        args = SimpleNamespace(length=10, linear_density=.03, bob_mass=1, radius=.003,
                               bob_radius=.15, angle=90, ei=.05, gj=.02, ea=1e5,
                               angular_damping=0, segments=40, fps=60, substeps=20,
                               iterations=40, lambda_retention=0, viewer='null')
        cls.p = Pendulum(args, cls.viewer)
        cls.rest = cls.p.state_0.body_q.numpy().copy()

    @classmethod
    def tearDownClass(cls):
        cls.viewer.close()

    def setUp(self):
        self.p.state_0.body_q.assign(self.rest)
        self.p.state_0.body_qd.zero_()
        for samples in [self.p.rows, self.p.transforms, self.p.nodes, self.p.velocities,
                        self.p.energies, self.p.endpoint_history]:
            samples.clear()
        self.p.args.strict_validation = False

    def measure(self):
        self.p.sample()
        return dict(zip(ENERGY_COLUMNS, self.p.energies[-1]))

    def test_initial_gravity_and_zero_motion(self):
        e = self.measure()
        self.assertAlmostEqual(e['cable_gravity_J'], .3 * 9.81 * 11, places=5)
        self.assertAlmostEqual(e['bob_gravity_J'], 9.81 * 11, places=5)
        self.assertEqual(e['cable_kinetic_J'] + e['bob_kinetic_J'], 0)

    def test_translation_kinetic_energy(self):
        qd = self.p.state_0.body_qd.numpy()
        qd[:, 0] = 2
        self.p.state_0.body_qd.assign(qd)
        e = self.measure()
        self.assertAlmostEqual(e['cable_kinetic_J'], .5 * .3 * 2**2, places=6)
        self.assertAlmostEqual(e['bob_kinetic_J'], .5 * 2**2, places=6)

    def test_bob_rotation_kinetic_energy(self):
        qd = self.p.state_0.body_qd.numpy()
        qd[self.p.bob, 3] = 2
        self.p.state_0.body_qd.assign(qd)
        self.assertAlmostEqual(self.measure()['bob_kinetic_J'], .5 * (.4 * .15**2) * 2**2, places=6)

    def test_height_reference(self):
        before = self.measure()
        q = self.rest.copy()
        q[:, 2] += 1
        self.p.state_0.body_q.assign(q)
        after = self.measure()
        self.assertAlmostEqual(after['cable_gravity_J'] - before['cable_gravity_J'], .3 * 9.81, places=5)
        self.assertAlmostEqual(after['bob_gravity_J'] - before['bob_gravity_J'], 9.81, places=5)

    def test_finite_bending(self):
        q = self.rest.copy()
        q[20, 3:] = np.array(wp.mul(wp.quat(*q[20, 3:]), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), .1)))
        self.p.state_0.body_q.assign(q)
        # Two neighboring joints bend; DER curvature is 2 tan(theta / 2).
        expected = .05 / self.p.ds * (2 * math.tan(.05))**2
        self.assertAlmostEqual(self.measure()['cable_bend_J'], expected, places=7)

    def test_material_twist(self):
        q = self.rest.copy()
        q[20, 3:] = np.array(wp.mul(wp.quat(*q[20, 3:]), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), .1)))
        self.p.state_0.body_q.assign(q)
        expected = .02 / self.p.ds * .1**2
        self.assertAlmostEqual(self.measure()['cable_twist_J'], expected, places=7)

    def test_endpoint_loads_in_static_hanging_configuration(self):
        q = self.rest.copy()
        n = len(self.p.bodies)
        segment_mass = .3 / n
        tension = 1.3 * 9.81
        z = -tension / self.p.attachment_stiffness
        for i, body in enumerate(self.p.bodies):
            if i:
                tension = (1 + (n - i) * segment_mass) * 9.81
                z -= tension / (self.p.args.ea / self.p.ds)
            q[body, :3] = (0, 0, z - self.p.ds / 2)
            q[body, 3:] = (0, 1, 0, 0)
            z -= self.p.ds
        q[self.p.bob, :3] = (0, 0, z - 9.81 / self.p.attachment_stiffness)
        q[self.p.bob, 3:] = (0, 1, 0, 0)
        self.p.state_0.body_q.assign(q)
        self.measure()
        load = self.p.endpoint_history[-1][-1]
        self.assertAlmostEqual(load[1], 1.3 * 9.81, delta=.01)
        self.assertAlmostEqual(load[2], 9.81, delta=.1)
        self.assertGreater(load[3], 0)
        self.assertGreater(load[4], 0)

    def test_endpoint_compression_has_positive_magnitude_and_negative_axial_load(self):
        q = self.rest.copy()
        q[self.p.bob, 0] -= .001
        self.p.state_0.body_q.assign(q)
        self.measure()
        load = self.p.endpoint_history[-1][-1]
        self.assertAlmostEqual(load[2], 100, delta=.1)
        self.assertLess(load[4], 0)

    def test_failed_validation_preserves_results(self):
        q = self.rest.copy()
        q[0, 0] += .02
        self.p.state_0.body_q.assign(q)
        self.measure()
        with tempfile.TemporaryDirectory() as folder:
            self.p.args.output = folder
            with contextlib.redirect_stdout(io.StringIO()):
                self.p.save()
            summary = json.loads((Path(folder) / 'summary.json').read_text())
            self.assertFalse(summary['constraint_validation']['passed'])
            self.assertTrue((Path(folder) / 'pendulum.html').is_file())
            self.assertTrue((Path(folder) / 'energies.csv').is_file())
            self.assertTrue((Path(folder) / 'tensions.csv').is_file())

    def test_strict_validation_still_raises_after_saving(self):
        q = self.rest.copy()
        q[0, 0] += .02
        self.p.state_0.body_q.assign(q)
        self.measure()
        self.p.args.strict_validation = True
        with tempfile.TemporaryDirectory() as folder:
            self.p.args.output = folder
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'failed constraint validation'):
                    self.p.save()
            self.assertTrue((Path(folder) / 'summary.json').is_file())


if __name__ == '__main__':
    unittest.main()

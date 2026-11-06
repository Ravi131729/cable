"""Newton cable pendulum: SI units, Z up, horizontal release from rest."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp
import newton
import newton.solvers
import newton.viewer
from newton._src.solvers.vbd.rigid_vbd_kernels import compute_geometric_rod_kappa_cached_z


@wp.kernel
def rod_strains(body_q: wp.array[wp.transform], bodies: wp.array[int],
                strains: wp.array[wp.vec3]):
    i = wp.tid()
    qa = wp.transform_get_rotation(body_q[bodies[i]])
    qb = wp.transform_get_rotation(body_q[bodies[i + 1]])
    strains[i] = compute_geometric_rod_kappa_cached_z(qa, qb, wp.vec3(0.0), 0.0)


ENERGY_COLUMNS = ('cable_kinetic_J', 'cable_gravity_J', 'cable_stretch_J',
                  'cable_shear_J', 'cable_bend_J', 'cable_twist_J', 'cable_elastic_J',
                  'cable_total_J', 'bob_kinetic_J', 'bob_gravity_J', 'bob_total_J',
                  'attachment_elastic_J', 'system_total_J', 'system_drift_J')


def rotate(q, v):
    u = q[..., :3]
    return v + 2 * np.cross(u, np.cross(u, v) + q[..., 3:4] * v)


class Pendulum:
    def __init__(self, args, viewer):
        self.args, self.viewer = args, viewer
        self.dt = 1 / (args.fps * args.substeps)
        self.ds = args.length / args.segments
        self.time = 0.0
        builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
        cfg = newton.ModelBuilder.ShapeConfig(density=0, has_shape_collision=False)
        # Local Z is the cable tangent. Release angle is measured from downward vertical.
        angle = math.radians(args.angle)
        direction = np.array([math.sin(angle), 0, -math.cos(angle)])
        q = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.pi - angle)
        # Place release height at zero to reduce float32 position cancellation.
        self.pivot = np.array([0.0, 0.0, 0.0])
        mass = args.linear_density * self.ds
        transverse = mass * (3 * args.radius**2 + self.ds**2) / 12
        axial = mass * args.radius**2 / 2
        inertia = wp.mat33(transverse, 0, 0, 0, transverse, 0, 0, 0, axial)
        bodies, joints = [], []
        for i in range(args.segments):
            center = self.pivot + (i + 0.5) * self.ds * direction
            body = builder.add_link(xform=wp.transform(wp.vec3(*center), q),
                                    mass=mass, inertia=inertia, label=f"cable_{i}")
            builder.add_shape_capsule(body, radius=args.radius, half_height=self.ds / 2,
                                      cfg=cfg)
            builder.shape_color[-1] = (1.0, 0.67, 0.12)
            bodies.append(body)
            if i == 0:
                joints.append(builder.add_joint_ball(
                    -1, body, parent_xform=wp.transform(wp.vec3(*self.pivot), q),
                    child_xform=wp.transform(wp.vec3(0, 0, -self.ds / 2), wp.quat_identity()),
                    damping=0, label="frictionless_top_pin"))
            else:
                # Continuum rigidities become discrete rotational springs R / ds.
                joints.append(builder.add_joint_rod(
                    bodies[i - 1], body,
                    parent_xform=wp.transform(wp.vec3(0, 0, self.ds / 2), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(0, 0, -self.ds / 2), wp.quat_identity()),
                    stretch_stiffness=args.ea / self.ds,
                    shear_stiffness=args.ea / self.ds,
                    bend_stiffness=args.ei / self.ds,
                    twist_stiffness=args.gj / self.ds,
                    bend_damping=args.angular_damping / self.ds,
                    twist_damping=args.angular_damping / self.ds,
                    stretch_damping=0, shear_damping=0, label=f"rod_joint_{i}"))
        self.bodies = np.array(bodies)
        end = self.pivot + args.length * direction
        bob_inertia = 0.4 * args.bob_mass * args.bob_radius**2
        self.bob = builder.add_link(
            xform=wp.transform(wp.vec3(*end), q), mass=args.bob_mass,
            inertia=wp.mat33(bob_inertia, 0, 0, 0, bob_inertia, 0, 0, 0, bob_inertia),
            label="bob_1kg")
        builder.add_shape_sphere(self.bob, radius=args.bob_radius, cfg=cfg)
        builder.shape_color[-1] = (0.20, 0.70, 1.0)
        joints.append(builder.add_joint_fixed(
            bodies[-1], self.bob,
            parent_xform=wp.transform(wp.vec3(0, 0, self.ds / 2), wp.quat_identity()),
            child_xform=wp.transform_identity(), label="bob_attachment"))
        builder.add_articulation(joints)
        builder.color()
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverVBD(self.model, iterations=args.iterations,
                                                rigid_compliant_alm=True,
                                                rigid_avbd_gamma=getattr(args, 'lambda_retention', 0.999),
                                                rigid_joint_linear_ke=1e5,
                                                rigid_joint_angular_ke=1e5)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.energy_masses = self.model.body_mass.numpy().astype(np.float64)
        self.energy_inertias = self.model.body_inertia.numpy().astype(np.float64)
        self.energy_bodies = wp.array(self.bodies, dtype=int, device=self.model.device)
        self.energy_strains = wp.empty(args.segments - 1, dtype=wp.vec3, device=self.model.device)
        self.energies = []
        self._energy_plot_frame = None
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0, -26, -args.length / 2), pitch=0, yaw=90)
        if args.viewer == 'viser':
            self.configure_viser_scene()
        # Even substep count ensures captured state buffers return to their original roles.
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        self.rows, self.transforms, self.nodes, self.velocities = [], [], [], []

    def simulate(self):
        for _ in range(self.args.substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def sample(self):
        transforms = self.state_0.body_q.numpy().copy()
        velocities = self.state_0.body_qd.numpy().copy()
        if not np.isfinite(transforms).all() or not np.isfinite(velocities).all():
            raise RuntimeError(f"Nonfinite state at t={self.time}")
        cable = transforms[self.bodies]
        offset = rotate(cable[:, 3:], np.array([0, 0, self.ds / 2]))
        starts, ends = cable[:, :3] - offset, cable[:, :3] + offset
        gaps = np.linalg.norm(starts[1:] - ends[:-1], axis=1)
        pin_error = np.linalg.norm(starts[0] - self.pivot)
        attachment_error = np.linalg.norm(ends[-1] - transforms[self.bob, :3])
        max_gap = max(float(gaps.max()), float(attachment_error))
        bob = transforms[self.bob, :3]
        speed = np.linalg.norm(velocities[self.bob, :3])
        masses = self.energy_masses
        inertias = self.energy_inertias
        local_omega = rotate(np.concatenate((-transforms[:, 3:6], transforms[:, 6:7]), axis=1),
                             velocities[:, 3:])
        kinetic = (0.5 * masses * np.sum(velocities[:, :3]**2, axis=1)).sum()
        kinetic += 0.5 * np.einsum('bi,bij,bj->', local_omega, inertias, local_omega)
        potential = (masses * 9.81 * (transforms[:, 2] - self.pivot[2])).sum()
        wp.launch(rod_strains, dim=self.args.segments - 1,
                  inputs=[self.state_0.body_q, self.energy_bodies], outputs=[self.energy_strains])
        strain = self.energy_strains.numpy().astype(np.float64)
        twist = strain[:, 2]
        kinetic_body = 0.5 * masses * np.sum(velocities[:, :3].astype(np.float64)**2, axis=1)
        kinetic_body += 0.5 * np.einsum('bi,bij,bj->b', local_omega, inertias, local_omega)
        # Gravitational zero is the visual floor, 1 m below the nominal lowest endpoint.
        potential_body = masses * 9.81 * (transforms[:, 2].astype(np.float64) + self.args.length + 1)
        gap_vectors = starts[1:].astype(np.float64) - ends[:-1]
        tangent = rotate(cable[:-1, 3:], np.array([0, 0, 1]))
        axial_gap = np.sum(gap_vectors * tangent, axis=1)
        shear_gap = gap_vectors - axial_gap[:, None] * tangent
        stretch_energy = 0.5 * self.args.ea / self.ds * np.sum(axial_gap**2)
        shear_energy = 0.5 * self.args.ea / self.ds * np.sum(shear_gap**2)
        bend_energy = 0.5 * self.args.ei / self.ds * np.sum(strain[:, :2]**2)
        twist_energy = 0.5 * self.args.gj / self.ds * np.sum(twist**2)
        elastic = stretch_energy + shear_energy + bend_energy + twist_energy
        cable_k, cable_u = kinetic_body[self.bodies].sum(), potential_body[self.bodies].sum()
        bob_k, bob_u = kinetic_body[self.bob], potential_body[self.bob]
        # Ball/fixed attachments also have finite numerical stiffness (SolverVBD default 1e5).
        q_last, q_bob = transforms[self.bodies[-1], 3:], transforms[self.bob, 3:]
        relative = wp.mul(wp.quat_inverse(wp.quat(*q_last)), wp.quat(*q_bob))
        angle = 2 * math.atan2(np.linalg.norm(np.array(relative)[:3]), abs(float(relative[3])))
        attachments = 0.5e5 * (pin_error**2 + attachment_error**2 + angle**2)
        total = cable_k + cable_u + elastic + bob_k + bob_u + attachments
        initial = self.energies[0][-2] if self.energies else total
        self.energies.append([cable_k, cable_u, stretch_energy, shear_energy, bend_energy,
                              twist_energy, elastic, cable_k + cable_u + elastic,
                              bob_k, bob_u, bob_k + bob_u, attachments, total, total - initial])
        self.rows.append([self.time, *bob, speed, pin_error, max_gap,
                          float(np.max(np.abs(twist))), float(kinetic + potential)])
        self.transforms.append(transforms)
        self.velocities.append(velocities)
        self.nodes.append(np.vstack((starts, ends[-1])))

    def run(self):
        self.sample()
        frames = round(self.args.duration * self.args.fps)
        frame = 0
        self.render()
        while frame < frames and self.viewer.is_running():
            start = time.perf_counter()
            if not self.viewer.should_step():
                self.render()
                time.sleep(1 / self.args.fps)
                continue
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
            self.time = (frame + 1) / self.args.fps
            self.sample()
            self.render()
            if (frame + 1) % self.args.fps == 0:
                drift = 100 * self.energies[-1][-1] / self.energies[0][-2]
                print(f"t={self.time:.1f}s speed={self.rows[-1][4]:.3f} m/s energy drift={drift:+.3f}%", flush=True)
            frame += 1
            if self.args.viewer in ('viser', 'gl'):
                time.sleep(max(0, 1 / self.args.fps - (time.perf_counter() - start)))

    def render(self):
        self.viewer.begin_frame(self.time)
        self.viewer.log_state(self.state_0)
        if self.args.viewer == 'viser':
            self.update_viser_scene(self.state_0)
            self.update_energy_plots(len(self.energies) - 1)
        self.viewer.end_frame()

    def configure_viser_scene(self):
        server = self.viewer.server
        scene = server.scene
        server.gui.configure_theme(dark_mode=True,
                                   brand_color=(255, 182, 53), show_share_button=False)
        server.gui.main_panel.dock_right()
        # A small gradient fills the viewport without depending on CSS/theme defaults.
        top, bottom = np.array([12, 20, 35]), np.array([32, 48, 66])
        gradient = np.linspace(top, bottom, 128).astype(np.uint8)
        scene.set_background_image(np.repeat(gradient[:, None, :], 128, axis=1), format='png')
        scene.world_axes.visible = False
        floor_z = -self.args.length - 1.0
        floor = scene.add_grid('/display/floor', width=self.args.length * 3,
                       height=self.args.length * 2, plane='xy', position=(0, 0, floor_z),
                       cell_size=1, section_size=5, cell_color=(65, 85, 104),
                       section_color=(105, 130, 151), cell_thickness=0.7,
                       section_thickness=1.3, plane_color=(28, 42, 56), plane_opacity=1,
                       fade_distance=70, shadow_opacity=0.2)
        scene.add_icosphere('/display/top_pin', radius=0.10, color=(240, 246, 255),
                            position=(0, 0, 0))
        scene.add_box('/display/support', dimensions=(1.2, 0.5, 0.16),
                      position=(0, 0, 0.24), color=(125, 146, 166))
        scene.add_label('/display/pin_label', 'Fixed top', position=(0.2, 0, 0.7))
        floor_label = scene.add_label('/display/floor_label', 'Floor · 1 m grid',
                        position=(-self.args.length, 0, floor_z + 0.2))
        self.cable_highlight = scene.add_line_segments(
            '/display/cable', points=np.zeros((self.args.segments, 2, 3), dtype=np.float32),
            colors=(255, 188, 62), thickness=4, thickness_units='screen')
        self.bob_highlight = scene.add_icosphere(
            '/display/bob', radius=self.args.bob_radius * 1.15, color=(67, 188, 255),
            material='standard', receive_shadow=False)
        self.bob_label = scene.add_label('/display/bob_label', f'{self.args.bob_mass:g} kg',
                                         position=(0, 0, 0))

        def set_view(front=False):
            length = self.args.length
            position = np.array([0, -2.8 * length, -0.5 * length] if front
                                else [0.45 * length, -2.7 * length, 0.25 * length])
            # Newton caches this request for both new clients and its camera reset.
            self.viewer._set_camera_request(position, np.array([0, 0, -0.45 * length]),
                                            np.array([0, 0, 1]), fov=48)

        with server.gui.add_folder('Pendulum appearance'):
            server.gui.add_markdown(
                '**Amber:** cable · **Blue:** bob  \n'
                'Floor is a visual reference. Display thickness is enlarged for clarity.')
            thickness = server.gui.add_slider('Cable width (px)', min=2, max=10, step=1, initial_value=4)
            front_button = server.gui.add_button('Front view')
            perspective_button = server.gui.add_button('3D view')
            floor_toggle = server.gui.add_checkbox('Floor', initial_value=True)

        @thickness.on_update
        def _(_event):
            self.cable_highlight.thickness = thickness.value

        @front_button.on_click
        def _(_event):
            set_view(front=True)

        @perspective_button.on_click
        def _(_event):
            set_view()

        @floor_toggle.on_update
        def _(_event):
            floor.visible = floor_toggle.value
            floor_label.visible = floor_toggle.value

        set_view()
        self.energy_charts = {}
        self.energy_chart_specs = {
            'Cable energy (J)': [('Kinetic', 0, '#ffbd3e'), ('Gravity', 1, '#7ebcff'),
                                 ('Elastic', 6, '#e992ff'), ('Total', 7, '#eff5ff')],
            'Bob energy (J)': [('Kinetic', 8, '#ffbd3e'), ('Gravity', 9, '#7ebcff'),
                               ('Total', 10, '#eff5ff')],
            'System energy (J)': [('Total', 12, '#79e3ac'), ('Initial', None, '#b5c4d6')],
            'Cable elastic energy (J)': [('Stretch', 2, '#ffbd3e'), ('Shear', 3, '#7ebcff'),
                                        ('Bend', 4, '#e992ff'), ('Twist', 5, '#79e3ac')],
        }
        self.energy_folder = server.gui.add_folder('Energy plots', expand_by_default=True)
        with self.energy_folder:
            server.gui.add_markdown('Time in seconds; energy in joules. Gravity is zero at the visual floor. '
                                    'Kinetic includes translation and rotation. System total includes attachment compliance.')
            self.energy_readout = server.gui.add_markdown('Waiting for initial energy sample.')

    def update_energy_plots(self, frame):
        if self._energy_plot_frame == frame:
            return
        from viser import uplot
        values = np.asarray(self.energies[:frame + 1], dtype=np.float64)
        times = np.asarray([r[0] for r in self.rows[:frame + 1]], dtype=np.float64)
        # Keep a minimal x extent for the initial single sample.
        for title, specs in self.energy_chart_specs.items():
            arrays = (times,) + tuple(values[:, col] if col is not None
                                      else np.full(len(times), self.energies[0][-2])
                                      for _, col, _ in specs)
            if title not in self.energy_charts:
                with self.energy_folder:
                    self.energy_charts[title] = self.viewer.server.gui.add_uplot(
                        data=arrays, title=title,
                        series=(uplot.Series(label='Time (s)'),) + tuple(
                            uplot.Series(label=label, stroke=color, width=2) for label, _, color in specs),
                        scales={'x': uplot.Scale(time=False)},
                        axes=(uplot.Axis(label='Time (s)'), uplot.Axis(label='Energy (J)')),
                        height=205)
            else:
                self.energy_charts[title].data = arrays
        drift = values[-1, -1]
        initial = self.energies[0][-2]
        self.energy_readout.content = (f'**t = {times[-1]:.2f} s** · total **{values[-1, -2]:.3f} J**  \n'
                                       f'Change from initial: **{drift:+.3f} J ({100 * drift / initial:+.2f}%)**  \n'
                                       f'Maximum drift so far: **{100 * np.max(np.abs(values[:, -1])) / initial:.2f}%**')
        self._energy_plot_frame = frame

    def update_viser_scene(self, state):
        transforms = state.body_q.numpy()
        cable = transforms[self.bodies]
        offset = rotate(cable[:, 3:], np.array([0, 0, self.ds / 2]))
        self.cable_highlight.points = np.stack((cable[:, :3] - offset,
                                                cable[:, :3] + offset), axis=1).astype(np.float32)
        bob = transforms[self.bob, :3]
        self.bob_highlight.position = bob
        self.bob_label.position = bob + np.array([0.35, 0, 0.35])

    def keep_viser_open(self):
        """After the run, allow scrubbing recorded 3D states without rerunning physics."""
        viewer = self.viewer
        playback_state = self.model.state()
        with viewer.server.gui.add_folder('Recorded pendulum'):
            slider = viewer.server.gui.add_slider('Frame', min=0, max=len(self.transforms) - 1,
                                                  step=1, initial_value=len(self.transforms) - 1)
            play = viewer.server.gui.add_checkbox('Play recording', initial_value=False)
        print(f'Recorded 3D playback: {viewer.url} (Ctrl+C to stop)', flush=True)
        last_frame = None
        while viewer.is_running():
            frame = int(slider.value)
            if play.value:
                frame = (frame + 1) % len(self.transforms)
                slider.value = frame
            if frame != last_frame:
                playback_state.body_q.assign(self.transforms[frame])
                viewer.begin_frame(self.rows[frame][0])
                viewer.log_state(playback_state)
                self.update_viser_scene(playback_state)
                self.update_energy_plots(frame)
                viewer.end_frame()
                last_frame = frame
            time.sleep(1 / self.args.fps)

    def save(self):
        out = Path(self.args.output)
        out.mkdir(parents=True, exist_ok=True)
        rows = np.array(self.rows)
        headers = ['time_s', 'bob_x_m', 'bob_y_m', 'bob_z_m', 'bob_speed_m_s',
                   'pin_error_m', 'max_attachment_gap_m', 'max_joint_twist_rad',
                   'kinetic_plus_gravity_J']
        with (out / 'trajectory.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        np.savez_compressed(out / 'states.npz', time=rows[:, 0],
                            body_q=np.array(self.transforms), body_qd=np.array(self.velocities),
                            cable_nodes=np.array(self.nodes), energies=np.array(self.energies),
                            energy_columns=np.array(ENERGY_COLUMNS))
        with (out / 'energies.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s', *ENERGY_COLUMNS])
            writer.writerows([r[0], *e] for r, e in zip(self.rows, self.energies))
        summary = dict(parameters=vars(self.args), cable_mass_kg=float(self.model.body_mass.numpy()[self.bodies].sum()),
                       bob_mass_kg=float(self.model.body_mass.numpy()[self.bob]),
                       simulated_seconds=self.time, max_bob_speed_m_s=float(rows[:, 4].max()),
                       max_pin_error_m=float(rows[:, 5].max()),
                       max_attachment_gap_m=float(rows[:, 6].max()),
                       max_joint_twist_rad=float(rows[:, 7].max()),
                       kinetic_plus_gravity_min_J=float(rows[:, 8].min()),
                       kinetic_plus_gravity_max_J=float(rows[:, 8].max()),
                       initial_system_energy_J=float(self.energies[0][-2]),
                       final_system_energy_J=float(self.energies[-1][-2]),
                       max_abs_system_energy_drift_J=float(np.max(np.abs(np.asarray(self.energies)[:, -1]))),
                       max_abs_system_energy_drift_percent=float(100 * np.max(np.abs(np.asarray(self.energies)[:, -1])) / self.energies[0][-2]),
                       min_bob_z_m=float(rows[:, 3].min()))
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        template = Path(__file__).with_name('playback.html').read_text()
        data = dict(time=rows[:, 0].round(4).tolist(), nodes=np.array(self.nodes).round(4).tolist(),
                    bob=rows[:, 1:4].round(4).tolist(), speed=rows[:, 4].round(3).tolist(),
                    pivot=self.pivot.tolist(), length=self.args.length, parameters=vars(self.args))
        (out / 'pendulum.html').write_text(template.replace('/*SIMULATION_DATA*/null', json.dumps(data, separators=(',', ':'))))
        if summary['max_pin_error_m'] > 0.01 or summary['max_attachment_gap_m'] > 0.01:
            raise RuntimeError('Constraint error exceeds 1 cm; increase substeps/iterations.')
        print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length', type=float, default=10)
    parser.add_argument('--linear-density', type=float, default=0.03)
    parser.add_argument('--bob-mass', type=float, default=1)
    parser.add_argument('--radius', type=float, default=0.003, help='Illustrative cable radius, m')
    parser.add_argument('--bob-radius', type=float, default=0.15)
    parser.add_argument('--angle', type=float, default=90, help='Degrees from downward vertical')
    parser.add_argument('--ei', type=float, default=0.05, help='Bending rigidity, N m^2')
    parser.add_argument('--gj', type=float, default=0.02, help='Torsional rigidity, N m^2')
    parser.add_argument('--ea', type=float, default=1e5, help='Axial rigidity, N')
    parser.add_argument('--angular-damping', type=float, default=0, help='Angular viscosity, N m^2 s')
    parser.add_argument('--segments', type=int, default=40)
    parser.add_argument('--duration', type=float, default=10)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--substeps', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--lambda-retention', type=float, default=0.999,
                        help='VBD numerical multiplier warm-start retention, from 0 to 1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--viewer', choices=['null', 'gl', 'usd', 'viser'], default='viser')
    parser.add_argument('--port', type=int, default=8080, help='Viser HTTP port')
    parser.add_argument('--autoplay', action='store_true', help='Start Viser simulation without pausing')
    parser.add_argument('--exit-on-finish', action='store_true', help='Close Viser after simulation instead of showing recorded playback')
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    for name in ['length', 'linear_density', 'bob_mass', 'radius', 'bob_radius', 'ea', 'duration', 'fps', 'substeps', 'iterations']:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f'{name} must be finite and positive')
    if args.segments < 2 or args.substeps % 2:
        parser.error('Use at least 2 segments and an even number of substeps')
    if not math.isfinite(args.lambda_retention) or not 0 <= args.lambda_retention <= 1:
        parser.error('lambda-retention must be between 0 and 1')
    if any(not math.isfinite(v) or v < 0 for v in [args.ei, args.gj, args.angular_damping]):
        parser.error('Rigidities and damping must be finite and nonnegative')
    wp.init()
    wp.set_device(args.device)
    frames = round(args.duration * args.fps)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if args.viewer == 'gl':
        viewer = newton.viewer.ViewerGL(num_frames=frames + 1)
    elif args.viewer == 'usd':
        viewer = newton.viewer.ViewerUSD(output_path=str(Path(args.output) / 'pendulum.usd'), num_frames=frames + 1)
    elif args.viewer == 'viser':
        viewer = newton.viewer.ViewerViser(port=args.port, paused=not args.autoplay,
                                          label='10 m cable pendulum', share=False)
    else:
        viewer = newton.viewer.ViewerNull(num_frames=frames + 1)
    simulation = Pendulum(args, viewer)
    try:
        simulation.run()
        simulation.save()
        if args.viewer == 'viser' and not args.exit_on_finish:
            simulation.keep_viser_open()
    except KeyboardInterrupt:
        print('\nViewer stopped.')
    finally:
        viewer.close()


if __name__ == '__main__':
    main()

Newton cable pendulum

Setup from the repository root (tested with Python 3.13):
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt

Run the simulation (headless, GPU):
  source .venv/bin/activate
  python cable_pendulum.py --viewer null

Run with Newton's interactive 3D viewer:
  python cable_pendulum.py --viewer gl

Run with Viser's browser-based 3D viewer:
  python cable_pendulum.py --viewer viser --output results_viser
  Open http://localhost:8080 and uncheck Pause to release the pendulum.
  After the simulation, use Recorded pendulum / Frame or Play recording.
  The server stays open until Ctrl+C. Use --port 8081 for a different port.
  Failed attachment validation is shown in Viser and saved in summary.json;
  replay remains available for diagnosis. Use --strict-validation to require
  an unsuccessful exit for such runs (for example, in automated data collection).
  For unattended runs: --autoplay --exit-on-finish.
  The Viser scene uses a dark navy background, amber cable, blue bob, and a
  visual floor 1 m below the lowest nominal cable endpoint (no floor contact).
  Under Pendulum appearance, adjust cable width in pixels, toggle the floor,
  or switch between front and 3D camera views. Display highlights do not change
  the cable's physical mass, radius, or stiffness.
  Energy plots show cable and bob kinetic/gravitational/total energy, cable
  elastic modes, and system total versus its initial energy. The current energy
  drift is displayed in joules and percent. Gravity is zero at the visual floor.
  Kinetic energy includes rotation. Elastic energies use Newton's discrete rod
  strain definition and the authored joint stiffnesses, including shear.
  System total includes finite pin/bob attachment spring energy. These graphs
  update while simulating and while scrubbing recorded playback.
  Endpoint tension / loads plots pivot and bob attachment-force magnitudes in N.
  The elastic attachment law is F = k * displacement (k = 100000 N/m).
  These resultants include shear/compression; they are material-force estimates,
  not raw numerical ALM multipliers. Signed axial components are also exported
  (positive tension, negative compression). Every physics substep is recorded,
  including between display frames, so short force spikes remain in the history.
  Load histories update during live simulation and recorded playback; failed
  physics validation also applies to the reliability of these force estimates.

Open results/pendulum.html in a browser for recorded playback with Play/Pause
and a time slider. This playback displays recorded Newton states; it does not
recompute physics in the browser. Coordinates are in metres relative to the pin.

Model
  Cable length: 10 m; linear density: 0.03 kg/m; total cable mass: 0.30 kg.
  End bob: 1 kg, idealized sphere of radius 0.15 m.
  Cable: 40 rigid cylindrical mass elements, connected with Newton rod joints.
  Cable radius: 0.003 m (illustrative; affects rotational inertia and display).
  Bending rigidity EI: 0.05 N m^2 (illustrative).
  Torsional rigidity GJ: 0.02 N m^2 (illustrative).
  Axial rigidity EA: 100000 N (illustrative); shear uses the same rigidity.
  Joint stiffnesses: EA / element_length, EI / element_length, GJ / element_length.
  Segment masses/inertias are assigned explicitly; display capsules add no mass.
  The top endpoint has a frictionless spherical joint to the world: fixed
  position, free orientation. It is not an orientation-clamped cantilever.
  The bob center coincides with the cable endpoint and is rigidly attached.
  Initial configuration: straight cable at --angle, zero linear/angular velocity,
  no pretwist; all temporary supports removed at t=0. Gravity is -9.81 m/s^2.
  Angles are measured from DOWNWARD vertical: 0 hangs down, 90 is horizontal,
  and 160 points mostly upward. Use --angle 90 to reproduce the original run.
  Above-horizontal releases can buckle and become slack before a sharp tension
  transient; they require separate numerical validation. Domain randomization
  does not make large energy loss or failed attachment validation acceptable.
  No ground, air drag, self-collision, or prescribed physical damping is added.
  The symmetric motion is planar. Torsional stiffness is active but no torsional
  excitation is imposed, so zero twist is expected for this release.

Outputs
  results/pendulum.html: saved animation.
  results/trajectory.csv: bob coordinates, speed, connection errors, material
    twist diagnostic, and kinetic plus gravitational energy (excludes elastic
    energy; do not interpret this column as total mechanical energy).
  results/energies.csv: separate cable/bob energy terms, attachment elastic
    energy, total mechanical energy, and change from initial total.
  results/tensions.csv: physics-substep time, pivot/bob force magnitudes, signed
    axial loads, and world force vectors. Pivot vector is force ON the cable;
    bob vector is force ON the bob. The support reaction has the opposite sign.
  results/states.npz: time, body positions/quaternions (XYZ, quaternion XYZW),
    linear/angular velocities, cable centerline nodes, and energy components,
    including the initial state.
  results/summary.json: exact run parameters and diagnostics.
  Optional USD animation: python cable_pendulum.py --viewer usd

Adjust duration, stiffness, and resolution, for example:
  python cable_pendulum.py --duration 20 --ei 0.1 --gj 0.04 --output results_custom
  python cable_pendulum.py --segments 80 --iterations 80 --output results_80segments
  python cable_pendulum.py --help

Numerical limits
  Newton uses float32 body states. Tiny timesteps can lose gravity displacements
  to position rounding; increasing substeps does not always improve accuracy.
  The default origin is the top pin to reduce cancellation during release.
  VBD uses finite stiffness and iterative constraints, so small attachment errors
  and numerical energy drift remain. This is an illustrative simulation using
  assumed material properties, not a validated prediction for a specific cable.
  Compare timestep, iteration, and spatial resolution before using numerical
  values for engineering decisions. Nonfinite states stop the simulation;
  attachment errors above 1 cm fail validation and generate a warning (or an
  exception when --strict-validation is enabled).

Verification of saved run
  Original 90-degree run: 10 seconds on RTX 3090, 60 frames/s,
  20 substeps/frame, 40 VBD iterations.
  Maximum bob speed: 14.7496 m/s.
  Maximum top pin error: 2.86 mm; maximum interbody attachment gap: 2.52 mm.
  Doubling iterations to 80 over the first 2.5 seconds changes peak speed by
  0.897%; maximum corresponding bob position difference is 0.188 m.
  This iteration comparison is not a full timestep/spatial convergence study.
  The full 10-second baseline has approximately 11.2% maximum energy drift.
  The best tested 40-segment setting (80 iterations, 20 substeps, numerical
  lambda retention 0) reduces maximum drift to 4.9%; no tested setting
  established less than 1% full-run conservation. Defaults are unchanged.
  See diagnostics/energy_audit.txt and diagnostics/energy_comparison.json.
  Playback initial rendering, time slider, and Play control were checked.

Run analytical energy checks:
  python -m unittest discover -s tests -v

Generated results and local environments are excluded from version control.

"""
FrankaTrajectoryValidator — inference-time validator for Franka robot trajectories.

Computes the following metrics per sample (batch of trajectories for a single scene):
  - feasible_traj_rate : fraction of trajectories that are collision-free AND within limits
  - collision_free_rate: fraction of trajectories that are collision-free
  - limits_rate        : fraction of trajectories within pos/vel/acc/jerk limits
  - success            : 1 if at least one trajectory is both collision-free and within
                         all kinematic limits, else 0
  - mean_jerk          : mean absolute jerk across joints and timesteps
  - max_jerk           : max absolute jerk across the trajectory
  - collision_depth    : worst-case (most negative) penetration distance across the batch
  - motion_time        : minimum execution time (assuming equidistant waypoints scaled
                         so that joint-velocity limits are never exceeded)

Optionally launches a Meshcat browser viewer to animate the generated
trajectories together with the obstacle scene.
"""

from __future__ import annotations

import os
import sys
import webbrowser
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
import meshcat
import meshcat.geometry as mg
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from campd.data.trajectory_sample import TrajectorySample
from campd.data.context import TrajectoryContext
from campd.experiments.validators import Validator, VALIDATORS

# Ensure the example `src/` directory is on the path so we can import
# `robot_model` when this file is auto-loaded via the `dependencies` mechanism.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from robot_model import FrankaPinocchioModel  # noqa: E402
from browser_key_server import BrowserKeyServer  # noqa: E402


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


@VALIDATORS.register("FrankaTrajectoryValidator")
class FrankaTrajectoryValidator(Validator):
    """Validator that evaluates generated Franka trajectories using Pinocchio.

    Parameters
    ----------
    urdf_path : str, optional
        Path to the Franka URDF. Defaults to the bundled asset.
    use_spheres : bool
        Whether to use sphere-based collision checking.
    joint_vel_max : list[float], optional
        Per-joint velocity limits (rad/s). If *None*, they are read from the
        URDF.  The Franka Panda defaults are
        ``[2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]``.
    collision_margin : float
        Safety margin (metres) added around the robot spheres when computing
        collision depth.
    visualize : bool
        If *True*, launch a Meshcat browser viewer and animate the generated
        trajectories after validation. The viewer URL is printed to stdout.
    playback_speed : float
        Multiplier on real-time playback speed (1.0 = real-time, 2.0 = twice
        as fast, 0.5 = half speed).  The per-frame duration is derived from
        the minimum-time parameterisation of the selected trajectory.
        Only used when ``visualize=True``.
    show_collision_spheres : bool
        If *True*, draw the sphere collision model alongside the visual
        robot model in the Meshcat viewer.  Only used when
        ``visualize=True``.  Default is *False* (visual model only).
    """

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        mesh_dir: Optional[str] = None,
        use_spheres: bool = True,
        spheres_config_path: Optional[str] = None,
        joint_vel_max: Optional[List[float]] = None,
        joint_pos_min: Optional[List[float]] = None,
        joint_pos_max: Optional[List[float]] = None,
        joint_acc_max: Optional[List[float]] = None,
        joint_jerk_max: Optional[List[float]] = None,
        dt: float = 0.0625,
        collision_margin: float = 0.0,
        visualize: bool = False,
        playback_speed: float = 1.0,
        show_collision_spheres: bool = False,
    ):
        # --- Resolve default paths relative to the example root -----------
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if urdf_path is None:
            urdf_path = os.path.join(
                root_dir, "assets", "urdf",
                "franka_panda.urdf",
            )
        if mesh_dir is None:
            mesh_dir = os.path.dirname(urdf_path)

        self.visualize = visualize
        self.playback_speed = playback_speed
        self.show_collision_spheres = show_collision_spheres
        self._dt = dt

        self.robot = FrankaPinocchioModel(
            urdf_path=urdf_path,
            mesh_dir=mesh_dir,
            use_spheres=use_spheres,
            spheres_config_path=spheres_config_path,
        )
        self.robot.margin = collision_margin

        # --- Load robot config for default limits ---------------------------
        _franka_cfg_path = os.path.join(
            root_dir, "assets", "robots", "franka", "configs", "franka.yml")
        _franka_cfg: Dict[str, Any] = {}
        if os.path.isfile(_franka_cfg_path):
            with open(_franka_cfg_path) as f:
                _franka_cfg = yaml.safe_load(f)

        # Joint velocity limits (7-DOF arm only)
        if joint_vel_max is not None:
            self._vel_max = np.array(joint_vel_max, dtype=np.float64)
        elif "vel_limits" in _franka_cfg:
            self._vel_max = np.array(
                _franka_cfg["vel_limits"], dtype=np.float64)
        else:
            # Fallback: read from URDF
            self._vel_max = np.array(
                [
                    self.robot.model.velocityLimit[
                        self.robot.model.joints[j].idx_v
                    ]
                    for j in range(1, 8)
                ],
                dtype=np.float64,
            )

        # Joint position limits (7-DOF arm only)
        if joint_pos_min is not None:
            self._pos_min = np.array(joint_pos_min, dtype=np.float64)
        elif "pos_lower" in _franka_cfg:
            self._pos_min = np.array(
                _franka_cfg["pos_lower"], dtype=np.float64)
        else:
            self._pos_min = np.array(
                [self.robot.model.lowerPositionLimit[j] for j in range(7)],
                dtype=np.float64,
            )
        if joint_pos_max is not None:
            self._pos_max = np.array(joint_pos_max, dtype=np.float64)
        elif "pos_upper" in _franka_cfg:
            self._pos_max = np.array(
                _franka_cfg["pos_upper"], dtype=np.float64)
        else:
            self._pos_max = np.array(
                [self.robot.model.upperPositionLimit[j] for j in range(7)],
                dtype=np.float64,
            )

        # Joint acceleration limits (rad/s²)
        if joint_acc_max is not None:
            self._acc_max = np.array(joint_acc_max, dtype=np.float64)
        elif "acc_limits" in _franka_cfg:
            self._acc_max = np.array(
                _franka_cfg["acc_limits"], dtype=np.float64)
        else:
            self._acc_max = np.array(
                [8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5], dtype=np.float64)

        # Joint jerk limits (rad/s³)
        if joint_jerk_max is not None:
            self._jerk_max = np.array(joint_jerk_max, dtype=np.float64)
        elif "jerk_limits" in _franka_cfg:
            self._jerk_max = np.array(
                _franka_cfg["jerk_limits"], dtype=np.float64)
        else:
            self._jerk_max = np.array(
                [2500., 2500., 2500., 2500., 2500., 2500., 2500.],
                dtype=np.float64,
            )

        # Lazily initialised Meshcat viewer
        self._viz: Optional[MeshcatVisualizer] = None
        self._key_server: Optional[BrowserKeyServer] = None
        self._viz_obstacle_names: List[str] = []
        self._ee_trail_names: List[str] = []
        self._other_trail_names: List[str] = []   # non-animated trails
        self._anim_trail_names: List[str] = []    # animated trajectory trail
        self._show_all_trails: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, batch: TrajectorySample, output_dir: str) -> Dict[str, Any]:
        """Compute all metrics for a generated batch of trajectories."""
        stats: Dict[str, Any] = {}
        assert batch.shared_context == True

        traj = batch.trajectory  # (B, T, D) or (T, D)
        if traj.dim() == 2:
            traj = traj.unsqueeze(0)

        traj_np = traj.detach().cpu().numpy()
        B, T, D = traj_np.shape

        # === 1. Collision checking (per-trajectory boolean) =================
        collisions, collision_depths = self._check_collisions_and_depth(
            batch, traj_np)

        collision_free = ~collisions  # shape (B,)
        stats["collision_free_rate"] = float(collision_free.mean())
        stats["success_coll_free"] = float(collision_free.any())
        if collisions.any():
            stats["collision_depth"] = float(collision_depths.min())
        else:
            stats["collision_depth"] = np.nan

        # === 2. Kinematic limit checking ====================================
        q = traj_np[..., :7]  # (B, T, 7) — arm joints only
        dt = self._dt

        # Finite differences (physical units)
        vel = np.diff(q, axis=1) / dt              # (B, T-1, 7) rad/s
        acc = np.diff(vel, axis=1) / dt             # (B, T-2, 7) rad/s²
        jerk = np.diff(acc, axis=1) / dt            # (B, T-3, 7) rad/s³

        # Per-trajectory limit checks
        pos_ok = np.all(
            (q >= self._pos_min[None, None, :]) &
            (q <= self._pos_max[None, None, :]),
            axis=(1, 2),
        )  # (B,)
        vel_ok = np.all(
            np.abs(vel) <= self._vel_max[None, None, :],
            axis=(1, 2),
        )  # (B,)
        acc_ok = np.all(
            np.abs(acc) <= self._acc_max[None, None, :],
            axis=(1, 2),
        )  # (B,)
        jerk_ok = np.all(
            np.abs(jerk) <= self._jerk_max[None, None, :],
            axis=(1, 2),
        )  # (B,)

        within_limits = pos_ok & vel_ok & acc_ok & jerk_ok  # (B,)
        stats["limits_rate"] = float(within_limits.mean())
        stats["success_limits"] = float(within_limits.any())
        stats["pos_limit_rate"] = float(pos_ok.mean())
        stats["vel_limit_rate"] = float(vel_ok.mean())
        stats["acc_limit_rate"] = float(acc_ok.mean())
        stats["jerk_limit_rate"] = float(jerk_ok.mean())

        # === 3. Combined feasibility ========================================
        feasible = collision_free & within_limits  # (B,)
        stats["feasible_traj_rate"] = float(feasible.mean())
        stats["success"] = float(feasible.any())

        # Jerk summary stats (physical units)
        stats["mean_jerk"] = float(np.mean(np.abs(jerk)))
        stats["max_jerk"] = float(np.max(np.abs(jerk)))

        # === 4. Motion time (minimum time parameterisation) =================
        motion_times = self._compute_motion_times(q, feasible)
        stats["motion_time_mean"] = float(np.mean(motion_times))
        stats["motion_time_min"] = float(np.min(motion_times))

        # Report the best feasible motion time (if any trajectory is feasible)
        if feasible.any():
            feasible_times = motion_times[feasible]
            stats["motion_time_best_feasible"] = float(np.min(feasible_times))
            stats["motion_time_mean_feasible"] = float(np.mean(feasible_times))
        else:
            stats["motion_time_best_feasible"] = float("inf")
            stats["motion_time_mean_feasible"] = float("inf")

        # === 5. Visualization (optional) ====================================
        if self.visualize:
            self._visualize_trajectories(
                batch, traj_np, collisions, motion_times)

        return stats

    @classmethod
    def from_config(cls, cfg: Any) -> "FrankaTrajectoryValidator":
        if isinstance(cfg, dict):
            return cls(**cfg)
        return cls(**cfg.model_dump())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_collisions_and_depth(
        self,
        batch: TrajectorySample,
        traj_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Check collisions and compute penetration depth per trajectory.

        Returns
        -------
        collisions : np.ndarray, shape (B,), dtype bool
            True if any timestep has a collision.
        collision_depths : np.ndarray, shape (B,)
            Minimum signed distance (most negative = deepest penetration)
            across all timesteps and collision pairs.
        """
        # ---- Fast vectorised path (sphere collision model) ----------------
        if self.robot._batched_sphere_data_ready:
            # For shared / unbatched context the context is the same for all
            # trajectories — pass it straight through.
            should_update_global = batch.shared_context or not batch.is_batched
            if should_update_global:
                return self.robot.check_collisions_and_depth_batched(
                    traj_np, batch.context)
            else:
                # Per-trajectory context: run the batched method once per
                # trajectory (still much faster than the triple loop).
                B, T, D = traj_np.shape
                collisions = np.zeros(B, dtype=bool)
                depths = np.zeros(B, dtype=np.float64)
                for b in range(B):
                    ctx_b = batch.context.slice(b)
                    coll_b, dep_b = self.robot.check_collisions_and_depth_batched(
                        traj_np[b:b+1], ctx_b)
                    collisions[b] = coll_b[0]
                    depths[b] = dep_b[0]
                return collisions, depths

        # ---- Legacy per-config Pinocchio path -----------------------------
        B, T, D = traj_np.shape

        collisions = np.zeros(B, dtype=bool)
        depths = np.zeros(B, dtype=np.float64)

        # Update context obstacles once if context is shared
        should_update_global = batch.shared_context or not batch.is_batched
        if should_update_global:
            self.robot.update_context_obstacles(batch.context)
            self.robot._refresh_collision_data()

        for b in range(B):
            if not should_update_global:
                ctx_b = batch.context.slice(b)
                self.robot.update_context_obstacles(ctx_b)
                self.robot._refresh_collision_data()

            traj_min_dist = np.inf
            traj_collides = False

            for t in range(T):
                q = traj_np[b, t]
                if q.shape[0] == 7:
                    q = np.concatenate([q, np.zeros(2)])

                pin.forwardKinematics(self.robot.model, self.robot.data, q)
                pin.updateGeometryPlacements(
                    self.robot.model, self.robot.data,
                    self.robot.collision_model, self.robot.collision_data, q,
                )
                pin.computeCollisions(
                    self.robot.collision_model, self.robot.collision_data, False)
                pin.computeDistances(
                    self.robot.collision_model, self.robot.collision_data)

                # Scan all collision pairs for min distance
                for k in range(len(self.robot.collision_model.collisionPairs)):
                    if self.robot.collision_data.collisionResults[k].isCollision():
                        traj_collides = True

                    dist = self.robot.collision_data.distanceResults[k].min_distance
                    if dist < traj_min_dist:
                        traj_min_dist = dist

            collisions[b] = traj_collides
            depths[b] = traj_min_dist

        return collisions, depths

    def _compute_motion_times(
        self,
        q: np.ndarray,
        feasible: np.ndarray,
    ) -> np.ndarray:
        """Compute minimum-time parameterisation for each trajectory.

        Given ``T`` equidistant waypoints, the time spacing ``dt`` must satisfy
        ``|q[t+1,j] - q[t,j]| / dt <= v_max[j]`` for every step ``t`` and
        joint ``j``.  Rearranging:

        .. math::

            dt >= max_{t,j} |Δq_{t,j}| / v_{max,j}

        and the total motion time is ``dt * (T - 1)``.

        Parameters
        ----------
        q : np.ndarray, shape (B, T, 7)
            Joint-space trajectories (arm joints only).
        feasible : np.ndarray, shape (B,), dtype bool
            Feasibility flags (unused here, but kept for potential filtering).

        Returns
        -------
        motion_times : np.ndarray, shape (B,)
            Minimum motion time for each trajectory.
        """
        B, T, D = q.shape
        dq = np.abs(np.diff(q, axis=1))  # (B, T-1, 7)

        # Required dt per step per joint: dq / v_max  (broadcast v_max over B, T)
        # v_max shape: (7,) → broadcast to (1, 1, 7)
        v_max = self._vel_max[np.newaxis, np.newaxis, :]
        dt_required = dq / v_max  # (B, T-1, 7)

        # The binding constraint is the maximum required dt across all steps & joints
        dt_min = dt_required.max(axis=(1, 2))  # (B,)

        motion_times = dt_min * (T - 1)
        return motion_times

    # ------------------------------------------------------------------
    # Meshcat visualisation
    # ------------------------------------------------------------------

    def _ensure_viz(self) -> MeshcatVisualizer:
        """Lazily create or return the Meshcat viewer."""
        if self._viz is not None:
            return self._viz

        viz = MeshcatVisualizer(
            self.robot.model,
            self.robot.collision_model,
            self.robot.visual_model,
        )
        import io as _io
        _real_stdout = sys.stdout
        sys.stdout = _io.StringIO()
        try:
            viz.initViewer(open=False)
        finally:
            sys.stdout = _real_stdout

        viz.loadViewerModel()
        viz.displayVisuals(True)
        viz.displayCollisions(self.show_collision_spheres)
        viz.display(pin.neutral(self.robot.model))

        # Start the browser key-relay server and open the wrapper page.
        meshcat_url = viz.viewer.url()
        self._key_server = BrowserKeyServer(meshcat_url)
        webbrowser.open(self._key_server.url)

        print(f"\n{'=' * 60}")
        print(f"  Viewer:  {self._key_server.url}")
        print(f"  Backend: {meshcat_url}")
        print(f"{'=' * 60}\n")
        sys.stdout.flush()

        self._key_server.start_terminal_reader()

        self._viz = viz
        return viz

    def _clear_viz_extras(self, viewer: meshcat.Visualizer) -> None:
        """Remove previously drawn obstacles and EE trails."""
        for name in self._viz_obstacle_names:
            viewer[name].delete()
        self._viz_obstacle_names.clear()

        for name in self._ee_trail_names:
            viewer[name].delete()
        self._ee_trail_names.clear()
        self._other_trail_names.clear()
        self._anim_trail_names.clear()

    def _draw_context_obstacles(
        self, viewer: meshcat.Visualizer, context: TrajectoryContext
    ) -> None:
        """Draw context obstacles (spheres, cuboids, cylinders) in Meshcat."""

        def _get(key: str):
            try:
                data = context.get_item(key)
                mask = context.get_mask(key)
            except KeyError:
                return None
            if torch.is_tensor(data):
                data = data.detach().cpu().numpy()
            if torch.is_tensor(mask):
                mask = mask.detach().cpu().numpy()
            return data[mask.astype(bool)]

        obstacle_color = np.array([0.8, 0.2, 0.2, 0.5])  # semi-transparent red

        # --- Sphere obstacles ------------------------------------------------
        centers = _get("sphere_centers")
        radii = _get("sphere_radii")
        if centers is not None and radii is not None:
            for i in range(len(centers)):
                name = f"obstacles/sphere_{i}"
                r = float(radii[i]) if radii.ndim == 1 else float(radii[i, 0])
                viewer[name].set_object(
                    mg.Sphere(r),
                    mg.MeshLambertMaterial(
                        color=self._rgba_to_hex(obstacle_color),
                        opacity=float(obstacle_color[3]),
                    ),
                )
                T_sphere = np.eye(4)
                T_sphere[:3, 3] = centers[i][:3]
                viewer[name].set_transform(T_sphere)
                self._viz_obstacle_names.append(name)

        # --- Cuboid obstacles ------------------------------------------------
        box_centers = _get("cuboid_centers")
        box_dims = _get("cuboid_dims")
        box_quats = _get("cuboid_quaternions")
        if box_centers is not None and box_dims is not None:
            for i in range(len(box_centers)):
                name = f"obstacles/cuboid_{i}"
                d = [abs(float(x)) for x in box_dims[i]]
                viewer[name].set_object(
                    mg.Box(d),
                    mg.MeshLambertMaterial(
                        color=self._rgba_to_hex(obstacle_color),
                        opacity=float(obstacle_color[3]),
                    ),
                )
                T_box = np.eye(4)
                T_box[:3, 3] = box_centers[i][:3]
                if box_quats is not None:
                    T_box[:3, :3] = self._quat_wxyz_to_R(box_quats[i])
                viewer[name].set_transform(T_box)
                self._viz_obstacle_names.append(name)

        # --- Cylinder obstacles ----------------------------------------------
        cyl_centers = _get("cylinder_centers")
        cyl_radii = _get("cylinder_radii")
        cyl_heights = _get("cylinder_heights")
        cyl_quats = _get("cylinder_quaternions")
        if cyl_centers is not None and cyl_radii is not None and cyl_heights is not None:
            for i in range(len(cyl_centers)):
                name = f"obstacles/cylinder_{i}"
                r = float(cyl_radii[i]) if cyl_radii.ndim == 1 else float(
                    cyl_radii[i, 0])
                h = float(cyl_heights[i]) if cyl_heights.ndim == 1 else float(
                    cyl_heights[i, 0])
                viewer[name].set_object(
                    mg.Cylinder(h, r),
                    mg.MeshLambertMaterial(
                        color=self._rgba_to_hex(obstacle_color),
                        opacity=float(obstacle_color[3]),
                    ),
                )
                T_cyl = np.eye(4)
                T_cyl[:3, 3] = cyl_centers[i][:3]
                if cyl_quats is not None:
                    T_cyl[:3, :3] = self._quat_wxyz_to_R(cyl_quats[i])
                # Meshcat cylinder is Y-up; Pinocchio convention is Z-up.
                R_fix = np.eye(4)
                R_fix[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                viewer[name].set_transform(T_cyl @ R_fix)
                self._viz_obstacle_names.append(name)

    def _draw_ee_trail(
        self,
        viewer: meshcat.Visualizer,
        traj_q: np.ndarray,
        traj_idx: int,
        color: np.ndarray,
        visible: bool = True,
    ) -> None:
        """Draw a thin line-strip of EE positions for one trajectory.

        """
        T = traj_q.shape[0]
        ee_positions = np.zeros((T, 3))

        model, data = self.robot.model, self.robot.data
        lf_id = model.getFrameId("panda_leftfinger")
        rf_id = model.getFrameId("panda_rightfinger")
        # Finger mesh extends ~0.054 m along the finger frame's local z-axis
        finger_tip_local = np.array([0.0, 0.0, 0.054])

        for t in range(T):
            q = traj_q[t]
            if q.shape[0] == 7:
                q = np.concatenate([q, np.zeros(2)])
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            lf = data.oMf[lf_id]
            rf = data.oMf[rf_id]
            lf_tip = lf.translation + lf.rotation @ finger_tip_local
            rf_tip = rf.translation + rf.rotation @ finger_tip_local
            ee_positions[t] = 0.5 * (lf_tip + rf_tip)

        # Build line segments from consecutive EE points
        starts = ee_positions[:-1]  # (T-1, 3)
        ends = ee_positions[1:]      # (T-1, 3)
        # Interleave: [s0, e0, s1, e1, ...] → shape (2*(T-1), 3)
        vertices = np.empty((2 * (T - 1), 3), dtype=np.float32)
        vertices[0::2] = starts
        vertices[1::2] = ends

        name = f"ee_trails/traj_{traj_idx}"
        viewer[name].set_object(
            mg.LineSegments(
                mg.PointsGeometry(vertices.T),
                mg.LineBasicMaterial(
                    color=self._rgba_to_hex(color), linewidth=2),
            )
        )
        if not visible:
            viewer[name].set_property("visible", False)
        self._ee_trail_names.append(name)

        # Start / goal markers
        for suffix, pos, marker_color in [
            ("start", ee_positions[0], [0.0, 0.8, 0.0, 1.0]),
            ("goal", ee_positions[-1], [0.8, 0.0, 0.0, 1.0]),
        ]:
            mname = f"ee_trails/traj_{traj_idx}_{suffix}"
            viewer[mname].set_object(
                mg.Sphere(0.015),
                mg.MeshLambertMaterial(
                    color=self._rgba_to_hex(np.array(marker_color)),
                ),
            )
            T_m = np.eye(4)
            T_m[:3, 3] = pos
            viewer[mname].set_transform(T_m)
            if not visible:
                viewer[mname].set_property("visible", False)
            self._ee_trail_names.append(mname)

    def _visualize_trajectories(
        self,
        batch: TrajectorySample,
        traj_np: np.ndarray,
        collisions: np.ndarray,
        motion_times: np.ndarray,
    ) -> None:
        """Render the scene and animate trajectories in Meshcat.

        1. All trajectories' EE trails are drawn as persistent lines
           (orange / yellow for feasible, black for colliding).
        2. The trajectory with the shortest motion time is animated in
           real-time (robot moves along waypoints).  The per-frame sleep
           is ``motion_time / (T-1) / playback_speed`` so the animation
           matches the true execution duration.
        """
        B, T, D = traj_np.shape
        viz = self._ensure_viz()
        viewer = viz.viewer

        # 1. Clear old drawings
        self._clear_viz_extras(viewer)

        # 2. Draw obstacles from context
        ctx = batch.context
        if ctx is not None:
            ctx_to_draw = ctx.slice(0) if ctx.is_batched else ctx
            self._draw_context_obstacles(viewer, ctx_to_draw)

        # 3. Determine which trajectory to animate.
        feasible_mask = ~collisions
        if feasible_mask.any():
            feasible_times = np.where(feasible_mask, motion_times, np.inf)
            best_idx = int(np.argmin(feasible_times))
        else:
            best_idx = int(np.argmin(motion_times))

        # 4. Draw EE trails for all trajectories.
        feasible_color = np.array([1.0, 0.55, 0.0, 0.9])    # orange
        colliding_color = np.array([0.0, 0.0, 0.0, 0.7])     # black
        anim_color = np.array([0.1, 0.4, 1.0, 1.0])          # blue
        for b in range(B):
            if b == best_idx:
                color = anim_color
            else:
                color = colliding_color if collisions[b] else feasible_color
            vis = True if b == best_idx else self._show_all_trails
            self._draw_ee_trail(viewer, traj_np[b], b, color, visible=vis)

        # Partition trail names into animated vs other.
        self._anim_trail_names = [
            n for n in self._ee_trail_names
            if n.startswith(f"ee_trails/traj_{best_idx}")
        ]
        self._other_trail_names = [
            n for n in self._ee_trail_names
            if n not in self._anim_trail_names
        ]

        best_mt = float(motion_times[best_idx])
        frame_dt = best_mt / max(T - 1, 1)  # real-time per-frame duration
        sleep_dt = frame_dt / max(self.playback_speed, 1e-6)

        # Precompute padded configs for the animation trajectory
        anim_qs = []
        for t in range(T):
            q = traj_np[best_idx, t]
            if q.shape[0] == 7:
                q = np.concatenate([q, np.zeros(2)])
            anim_qs.append(q)

        # If the trajectory is colliding, stop at the first collision frame
        anim_end = T
        if collisions[best_idx]:
            # Run a lightweight per-timestep collision check on just this traj
            traj_single = traj_np[best_idx:best_idx + 1]  # (1, T, D)
            _, per_t_depths = self.robot.check_collisions_and_depth_batched(
                traj_single.reshape(T, 1, D),   # treat each timestep as B=1
                batch.context if (batch.shared_context or not batch.is_batched)
                else batch.context.slice(best_idx),
            )
            # per_t_depths shape: (T,) — signed distance per timestep
            collision_mask = per_t_depths < 0
            if collision_mask.any():
                anim_end = int(np.argmax(collision_mask)) + \
                    1  # stop just after
                anim_qs = anim_qs[:anim_end]

        status = "feasible" if not collisions[best_idx] else "COLLIDING"
        stop_info = f", stops at t={anim_end}/{T}" if collisions[best_idx] else ""
        print(
            f"  ▶ Animating trajectory {best_idx} ({status}), "
            f"motion time {best_mt:.3f}s, "
            f"frame dt {frame_dt:.4f}s × {self.playback_speed:.1f}x"
            f"{stop_info}"
        )

        anim_frame, interrupt_key = self._play_animation(
            viz, anim_qs, sleep_dt)

        if interrupt_key in ('\r', '\n', 'q'):
            return

        self._interactive_scrub(viz, anim_qs, sleep_dt,
                                start_frame=anim_frame,
                                pending_key=interrupt_key)

    # ------------------------------------------------------------------
    # Animation playback helpers
    # ------------------------------------------------------------------

    def _getch(self, timeout: float | None = None) -> str | None:
        """Read a single keypress from the browser control bar.

        Returns the key string, or *None* if *timeout* seconds elapse
        without input.  Arrow keys are returned as ``'LEFT'``,
        ``'RIGHT'``, ``'UP'``, ``'DOWN'``.
        """
        return self._key_server.get_key(timeout)

    def _play_animation(
        self,
        viz: MeshcatVisualizer,
        qs: List[np.ndarray],
        sleep_dt: float,
        start: int = 0,
    ) -> Tuple[int, str | None]:
        """Play the animation from *start*.

        Returns ``(frame_index, key)`` where *key* is the keypress that
        interrupted playback, or *None* if the animation played to the end.
        """
        for i in range(start, len(qs)):
            viz.display(qs[i])
            key = self._getch(timeout=sleep_dt)
            if key is not None:
                if key == 't':
                    self._toggle_all_trails(viz.viewer)
                    continue
                return i, key
        return len(qs) - 1, None

    def _interactive_scrub(
        self,
        viz: MeshcatVisualizer,
        qs: List[np.ndarray],
        sleep_dt: float,
        start_frame: int | None = None,
        pending_key: str | None = None,
    ) -> None:
        """Interactive post-animation controller.

        Keys
        ----
        ← / →   Step one frame backward / forward.
        Space   Resume playback from the current frame.
        r       Replay from the beginning.
        Enter   Advance to the next environment.
        q       Advance to the next environment (alias).
        """
        frame = start_frame if start_frame is not None else len(qs) - 1
        print(
            "  Controls: ←/→ step | Space resume | "
            "r replay | t trails | Enter/q next env"
        )

        key = pending_key
        while True:
            if key is None:
                key = self._getch()
            if key is None:
                continue

            if key == 'LEFT':
                frame = max(0, frame - 1)
                viz.display(qs[frame])
                print(
                    f"\033[2K\r  Frame {frame}/{len(qs)-1}", end="", flush=True)

            elif key == 'RIGHT':
                frame = min(len(qs) - 1, frame + 1)
                viz.display(qs[frame])
                print(
                    f"\033[2K\r  Frame {frame}/{len(qs)-1}", end="", flush=True)

            elif key == ' ':  # space — resume playback
                print(f"\033[2K\r  ▶ Resuming from frame {frame}...")
                frame, sub_key = self._play_animation(
                    viz, qs, sleep_dt, start=frame,
                )
                if sub_key in ('\r', '\n', 'q'):
                    print()
                    break
                key = sub_key
                print(
                    f"\033[2K\r  Paused at frame {frame}/{len(qs)-1}. "
                    "←/→ step | Space resume | r replay | Enter/q next",
                    end="", flush=True,
                )
                continue

            elif key == 'r':  # replay from start
                print("\033[2K\r  ▶ Replaying...")
                frame, sub_key = self._play_animation(
                    viz, qs, sleep_dt, start=0,
                )
                if sub_key in ('\r', '\n', 'q'):
                    print()
                    break
                key = sub_key
                print(
                    f"\033[2K\r  Paused at frame {frame}/{len(qs)-1}. "
                    "←/→ step | Space resume | r replay | Enter/q next",
                    end="", flush=True,
                )
                continue

            elif key == 't':  # toggle other trajectory trails
                self._toggle_all_trails(viz.viewer)
                label = "all" if self._show_all_trails else "animated only"
                print(
                    f"\033[2K\r  Trails: {label}",
                    end="", flush=True,
                )

            elif key in ('\r', '\n', 'q'):  # Enter or q — next env
                print()
                break

            # Clear so next iteration reads a fresh keypress.
            key = None

    def _toggle_all_trails(self, viewer: meshcat.Visualizer) -> None:
        """Toggle visibility of non-animated EE trails."""
        self._show_all_trails = not self._show_all_trails
        for name in self._other_trail_names:
            viewer[name].set_property("visible", self._show_all_trails)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rgba_to_hex(color: np.ndarray) -> int:
        """Convert [R, G, B, (A)] float array to a 0xRRGGBB integer for Meshcat."""
        r, g, b = (np.clip(color[:3], 0, 1) * 255).astype(int)
        return (int(r) << 16) + (int(g) << 8) + int(b)

    @staticmethod
    def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
        """Quaternion (w,x,y,z) → 3×3 rotation matrix."""
        return pin.Quaternion(
            np.array([q[1], q[2], q[3], q[0]], dtype=float)
        ).toRotationMatrix()

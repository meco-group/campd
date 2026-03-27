from collections import defaultdict
from matplotlib.figure import Figure
import time
import os
import yaml
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, List, Tuple
from campd.data.trajectory_sample import TrajectorySample
from campd.data.context import TrajectoryContext
import torch
import pinocchio as pin
import hppfcl
import meshcat
from pinocchio.visualize import MeshcatVisualizer


def quat_wxyz_to_R(quat_wxyz):
    return pin.Quaternion(
        np.array(quat_wxyz[[1, 2, 3, 0]])).toRotationMatrix()


def _batch_pR_revolute(pR: np.ndarray, theta: np.ndarray, axis: str) -> np.ndarray:
    """Compute ``pR @ R_axis(theta)`` for *N* joint angles at once.

    Parameters
    ----------
    pR : (3, 3)
        Fixed placement rotation of the joint.
    theta : (N,)
        Joint angle for each configuration.
    axis : ``'x'``, ``'y'``, or ``'z'``

    Returns
    -------
    result : (N, 3, 3)
    """
    N = len(theta)
    c = np.cos(theta)[:, None]          # (N, 1)
    s = np.sin(theta)[:, None]
    p0, p1, p2 = pR[:, 0], pR[:, 1], pR[:, 2]  # each (3,)

    if axis == 'z':
        col0 = c * p0 + s * p1
        col1 = -s * p0 + c * p1
        col2 = np.broadcast_to(p2, (N, 3))
    elif axis == 'x':
        col0 = np.broadcast_to(p0, (N, 3))
        col1 = c * p1 + s * p2
        col2 = -s * p1 + c * p2
    elif axis == 'y':
        col0 = c * p0 - s * p2
        col1 = np.broadcast_to(p1, (N, 3))
        col2 = s * p0 + c * p2
    else:
        raise ValueError(f"Unknown revolute axis: {axis}")

    return np.stack([col0, col1, col2], axis=-1)   # (N, 3, 3)


def _batch_pR_revolute_general(pR: np.ndarray, theta: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Compute ``pR @ R_rodrigues(axis, theta)`` for *N* joint angles.

    Parameters
    ----------
    pR : (3, 3)
        Fixed placement rotation of the joint.
    theta : (N,)
        Joint angle for each configuration.
    axis : (3,)
        Unit rotation axis in placement-local frame.

    Returns
    -------
    result : (N, 3, 3)
    """
    N = len(theta)
    c = np.cos(theta)                    # (N,)
    s = np.sin(theta)                    # (N,)
    t = 1.0 - c                          # (N,)
    x, y, z = axis[0], axis[1], axis[2]

    # Rodrigues rotation matrix R(axis, theta) - column-major assembly
    R = np.empty((N, 3, 3))
    R[:, 0, 0] = t * x * x + c
    R[:, 0, 1] = t * x * y - s * z
    R[:, 0, 2] = t * x * z + s * y
    R[:, 1, 0] = t * x * y + s * z
    R[:, 1, 1] = t * y * y + c
    R[:, 1, 2] = t * y * z - s * x
    R[:, 2, 0] = t * x * z - s * y
    R[:, 2, 1] = t * y * z + s * x
    R[:, 2, 2] = t * z * z + c

    # pR @ R  →  (N, 3, 3)
    return np.matmul(pR[None, :, :], R)


class FrankaPinocchioModel:
    def __init__(self, urdf_path: str = None, mesh_dir: str = None, ee_frame: str = "panda_hand",
                 debug: bool = False, use_spheres: bool = True,
                 spheres_config_path: str = None,
                 obstacle_collision_margin: float = 0.0,
                 self_collision_margin: float = 0.0):
        """
        Initialize the FrankaPinocchioModel with a Pinocchio model.

        Args:
            urdf_path: Path to the URDF file. If None, tries to find it relative to this file.
            mesh_dir: Path to the mesh directory.
            ee_frame: Name of the end-effector frame.
            debug: If True, launch a MeshcatVisualizer for interactive debugging.
            use_spheres: If True, replace URDF collision geometries with spheres
                         from `spheres_config_path`.
            spheres_config_path: Path to the sphere collision YAML config.
                                 If None and use_spheres is True, defaults to
                                 `assets/robots/franka/configs/franka_spheres.yml`
                                 relative to the example root directory.
            obstacle_collision_margin: Margin for obstacle collision checking.
            self_collision_margin: Margin for self collision checking.
        """
        if urdf_path is None:
            # Default logic from the existing visualizer.py
            root_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            urdf_path = os.path.join(
                root_dir, "assets", "urdf", "franka_panda.urdf")

        if mesh_dir is None:
            mesh_dir = os.path.dirname(urdf_path)

        self.use_spheres = use_spheres

        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, mesh_dir)

        srdf_model_path = urdf_path.replace(".urdf", ".srdf")
        if use_spheres:
            if spheres_config_path is None:
                root_dir = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__)))
                spheres_config_path = os.path.join(
                    root_dir, "assets", "franka_spheres.yml")
            self._load_sphere_collision_model(
                spheres_config_path, srdf_model_path)
        else:
            self.collision_model.addAllCollisionPairs()
            # Remove collision pairs listed in the SRDF file

            pin.removeCollisionPairs(
                self.model, self.collision_model, srdf_model_path)

        self.data, self.collision_data, self.visual_data = pin.createDatas(
            self.model, self.collision_model, self.visual_model
        )

        if self.model.existFrame(ee_frame):
            self.ee_fid = self.model.getFrameId(ee_frame)
        else:
            print(f"Warning: Frame '{ee_frame}' not found. Using last frame.")
            self.ee_fid = self.model.nframes - 1

        self.registered_obstacles = set()
        self.registered_obstacles_geoms = set()

        if not use_spheres:
            self.robot_geoms = set(range(self.collision_model.ngeoms))
            self.robot_geoms.remove(
                self.collision_model.getGeometryId("panda_link0_0"))
            self.robot_geoms.remove(
                self.collision_model.getGeometryId("panda_link1_0"))
            # link0/link1 already excluded from robot_geoms
            self.obstacle_check_geoms = set(self.robot_geoms)
        self.obstacle_collision_margin = obstacle_collision_margin
        self.self_collision_margin = self_collision_margin

        # Precompute sphere data for batched collision checking
        self._batched_sphere_data_ready = False
        self._vectorized_fk_ok = False
        if use_spheres:
            self._precompute_batched_sphere_data()
            self._precompute_fk_chain()

        self.debug = debug
        self.viz: MeshcatVisualizer | None = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpicklable Data objects
        state.pop('data', None)
        state.pop('collision_data', None)
        state.pop('visual_data', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate Data objects after pickling (e.g., in multiprocessing workers)
        if hasattr(self, 'model') and hasattr(self, 'collision_model') and hasattr(self, 'visual_model'):
            self.data, self.collision_data, self.visual_data = pin.createDatas(
                self.model, self.collision_model, self.visual_model
            )

    def _load_sphere_collision_model(self, config_path: str, srdf_path: str):
        """
        Replace the URDF mesh collision model with spheres defined in a YAML config.

        The YAML file maps link names to lists of {center, radius} dicts.
        Each sphere is attached to the corresponding link's parent joint
        with the appropriate placement offset. Collision pairs listed in the
        SRDF file are excluded.

        Args:
            config_path: Path to the sphere collision YAML config file.
            srdf_path: Path to the SRDF file for collision pair exclusions.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Sphere collision config not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        collision_spheres = config.get('collision_spheres', {})

        # Parse SRDF to get excluded collision pairs (as frozensets of link names)
        excluded_link_pairs = set()
        if os.path.exists(srdf_path):
            tree = ET.parse(srdf_path)
            for elem in tree.iter('disable_collisions'):
                l1 = elem.get('link1')
                l2 = elem.get('link2')
                if l1 and l2:
                    excluded_link_pairs.add(frozenset((l1, l2)))
        else:
            print(f"Warning: SRDF not found at {srdf_path}, "
                  "no collision pairs will be excluded.")

        # Clear existing URDF collision model
        # Build a fresh empty collision model attached to the same kinematic model
        self.collision_model = pin.GeometryModel()

        self.robot_geoms = set()
        # Track parent link name per sphere (for SRDF exclusion filtering)
        sphere_link_map = {}  # geom_id -> link_name

        for link_name, spheres in collision_spheres.items():
            # Skip links with negative radius (disabled) like attached_object
            if not self.model.existFrame(link_name):
                continue

            frame_id = self.model.getFrameId(link_name)
            frame = self.model.frames[frame_id]
            parent_joint = frame.parentJoint
            # The frame placement relative to the parent joint
            frame_placement = frame.placement

            for i, sphere_def in enumerate(spheres):
                radius = float(sphere_def['radius'])
                if radius <= 0:
                    continue  # Skip disabled spheres

                center = np.array(sphere_def['center'], dtype=float)
                shape = hppfcl.Sphere(radius)

                # Sphere placement: frame placement composed with the sphere center offset
                sphere_placement = frame_placement * pin.SE3(
                    np.eye(3), center)

                name = f"{link_name}_sphere_{i}"
                geom = pin.GeometryObject(
                    name, parent_joint, frame_id, sphere_placement, shape)
                geom.color = np.array([0.0, 0.8, 0.2, 0.4])  # Green-ish

                geom_id = self.collision_model.addGeometryObject(geom)
                self.robot_geoms.add(geom_id)
                sphere_link_map[geom_id] = link_name

        # Exclude link0/link1 from obstacle collision checks (they don't move)
        STATIC_LINKS = {"panda_link0", "panda_link1"}
        self.obstacle_check_geoms = {
            gid for gid in self.robot_geoms
            if sphere_link_map[gid] not in STATIC_LINKS
        }

        # Add collision pairs, excluding:
        # - pairs on the same link
        # - pairs listed in the SRDF disable_collisions
        geom_ids = sorted(self.robot_geoms)
        for idx_a in range(len(geom_ids)):
            for idx_b in range(idx_a + 1, len(geom_ids)):
                ga, gb = geom_ids[idx_a], geom_ids[idx_b]
                la, lb = sphere_link_map[ga], sphere_link_map[gb]
                # Skip same-link pairs
                if la == lb:
                    continue
                # Skip pairs excluded by SRDF
                if frozenset((la, lb)) in excluded_link_pairs:
                    continue
                self.collision_model.addCollisionPair(
                    pin.CollisionPair(ga, gb))

    def _precompute_batched_sphere_data(self):
        """Precompute sphere metadata for vectorised collision computation.

        Extracts per-sphere parent joint IDs, local translations, and radii
        from the Pinocchio collision model into contiguous NumPy arrays.
        Also stores self-collision pair indices and obstacle-check sphere
        indices so that :meth:`check_collisions_and_depth_batched` can run
        without touching Pinocchio's collision pipeline.
        """
        geom_ids = sorted(self.robot_geoms)
        n = len(geom_ids)

        self._batched_sphere_parent_joints = np.zeros(n, dtype=np.int32)
        self._batched_sphere_local_translations = np.zeros((n, 3))
        self._batched_sphere_radii = np.zeros(n)
        self._batched_geom_to_sphere_idx: dict[int, int] = {}

        # Group spheres by parent joint for efficient FK application
        spheres_by_joint: dict[int, list[int]] = defaultdict(list)

        for s, gid in enumerate(geom_ids):
            geom = self.collision_model.geometryObjects[gid]
            self._batched_geom_to_sphere_idx[gid] = s
            self._batched_sphere_parent_joints[s] = geom.parentJoint
            self._batched_sphere_local_translations[s] = geom.placement.translation.copy(
            )
            self._batched_sphere_radii[s] = geom.geometry.radius
            spheres_by_joint[geom.parentJoint].append(s)

        # Store per-joint groups as NumPy arrays for fast indexing
        self._batched_spheres_by_joint: dict[int, np.ndarray] = {
            jid: np.array(idxs) for jid, idxs in spheres_by_joint.items()
        }

        # Self-collision pairs (translated from geom-IDs to sphere indices)
        # Split into same-joint (constant distance) and cross-joint pairs.
        same_joint_pairs = []
        cross_joint_pairs = []
        for cp in self.collision_model.collisionPairs:
            ga, gb = cp.first, cp.second
            if ga in self._batched_geom_to_sphere_idx and gb in self._batched_geom_to_sphere_idx:
                sa = self._batched_geom_to_sphere_idx[ga]
                sb = self._batched_geom_to_sphere_idx[gb]
                if self._batched_sphere_parent_joints[sa] == self._batched_sphere_parent_joints[sb]:
                    same_joint_pairs.append((sa, sb))
                else:
                    cross_joint_pairs.append((sa, sb))

        # Same-joint spheres have constant distance – precompute once
        if same_joint_pairs:
            sj = np.array(same_joint_pairs, dtype=np.int32)
            d = (self._batched_sphere_local_translations[sj[:, 0]]
                 - self._batched_sphere_local_translations[sj[:, 1]])
            dists = np.sqrt((d * d).sum(axis=-1))
            signed = dists - (self._batched_sphere_radii[sj[:, 0]]
                              + self._batched_sphere_radii[sj[:, 1]]) - self.self_collision_margin
            self._same_joint_min_signed = float(signed.min())
        else:
            self._same_joint_min_signed = np.inf

        # Only cross-joint pairs need per-config computation.
        # Group by (joint_A, joint_B) for better cache locality.
        from collections import defaultdict as _ddict
        _groups: dict[tuple[int, int], list[tuple[int, int]]] = _ddict(list)
        for sa, sb in cross_joint_pairs:
            ja = int(self._batched_sphere_parent_joints[sa])
            jb = int(self._batched_sphere_parent_joints[sb])
            _groups[(ja, jb)].append((sa, sb))

        # Also keep flat pairs array for fallback / debugging
        self._batched_self_collision_pairs = (
            np.array(cross_joint_pairs, dtype=np.int32) if cross_joint_pairs
            else np.empty((0, 2), dtype=np.int32)
        )

        # Build per-group data: for each group, store unique sphere indices
        # and the pair mask so we can compute all-pairs and select the
        # relevant ones.
        self._self_collision_groups = []
        for (ja, jb), pair_list in _groups.items():
            a_set = sorted(set(p[0] for p in pair_list))
            b_set = sorted(set(p[1] for p in pair_list))
            a_arr = np.array(a_set, dtype=np.int32)
            b_arr = np.array(b_set, dtype=np.int32)
            a_map = {v: i for i, v in enumerate(a_set)}
            b_map = {v: i for i, v in enumerate(b_set)}
            # Radius sum for each (local_a, local_b) in the all-pairs grid
            ra = self._batched_sphere_radii[a_arr]                   # (nA,)
            rb = self._batched_sphere_radii[b_arr]                   # (nB,)
            sum_r_grid = ra[:, None] + rb[None, :]                   # (nA, nB)
            # Mask: True for grid cells that are actual collision pairs
            mask = np.zeros((len(a_set), len(b_set)), dtype=bool)
            for sa, sb in pair_list:
                mask[a_map[sa], b_map[sb]] = True
            self._self_collision_groups.append(
                (a_arr, b_arr, sum_r_grid, mask))

        # Indices of robot spheres that should be checked against obstacles
        self._batched_obstacle_check_indices = np.array(
            sorted(self._batched_geom_to_sphere_idx[gid]
                   for gid in self.obstacle_check_geoms),
            dtype=np.int32,
        )

        self._batched_sphere_data_ready = True

    # ------------------------------------------------------------------
    # Batched forward kinematics & collision checking
    # ------------------------------------------------------------------

    def _precompute_fk_chain(self):
        """Extract kinematic chain data for fully vectorised FK.

        Stores per-joint placement matrices, parent indices, and joint
        type/axis information so that
        :meth:`compute_sphere_centers_batched` can compute all
        world-frame sphere centres with a short loop over *joints*
        (typically ~10) instead of a loop over *configurations*
        (potentially thousands).
        """
        nj = self.model.njoints
        self._fk_njoints = nj
        self._fk_parent = np.array(
            [self.model.parents[j] for j in range(nj)], dtype=np.int32
        )
        self._fk_pR = np.stack(
            [self.model.jointPlacements[j].rotation.copy() for j in range(nj)]
        )  # (nj, 3, 3)
        self._fk_pt = np.stack(
            [self.model.jointPlacements[j].translation.copy()
             for j in range(nj)]
        )  # (nj, 3)

        # Classify each joint
        _REVOLUTE_TAGS = {'RZ': 'rz', 'RX': 'rx', 'RY': 'ry'}
        _PRISMATIC_TAGS = {'PX': 'px', 'PY': 'py', 'PZ': 'pz'}

        self._fk_jtype: list[str] = []
        self._fk_idx_q: list[int] = []
        # Per-joint local prismatic axis (only used for 'p_unaligned')
        self._fk_prismatic_axis: dict[int, np.ndarray] = {}
        ok = True

        for j in range(nj):
            joint = self.model.joints[j]
            if joint.nq == 0:
                self._fk_jtype.append('fixed')
                self._fk_idx_q.append(-1)
                continue

            sn = joint.shortname()
            self._fk_idx_q.append(joint.idx_q)

            matched = False
            for tag, jtype in {**_REVOLUTE_TAGS, **_PRISMATIC_TAGS}.items():
                if tag in sn:
                    self._fk_jtype.append(jtype)
                    matched = True
                    break
            if not matched:
                # Handle PrismaticUnaligned (or any 1-DOF prismatic variant)
                if 'Prismatic' in sn and joint.nq == 1:
                    axis = self._extract_prismatic_axis(j)
                    self._fk_prismatic_axis[j] = axis
                    self._fk_jtype.append('p_unaligned')
                    matched = True
                # Handle RevoluteUnaligned (arbitrary-axis revolute, 1-DOF)
                elif 'Revolute' in sn and joint.nq == 1:
                    axis = self._extract_revolute_axis(j)
                    # reuse dict for axis storage
                    self._fk_prismatic_axis[j] = axis
                    self._fk_jtype.append('r_unaligned')
                    matched = True

            if not matched:
                self._fk_jtype.append('unknown')
                ok = False

        self._vectorized_fk_ok = ok
        if not ok:
            unsupported = [(j, self._fk_jtype[j], self.model.joints[j].shortname())
                           for j in range(nj) if self._fk_jtype[j] == 'unknown']
            print(f"Warning: unsupported joint type(s) {unsupported}; "
                  "falling back to Pinocchio-based FK loop.")

    def _extract_prismatic_axis(self, j: int) -> np.ndarray:
        """Get the local prismatic axis for joint *j* via finite-difference FK."""
        data = pin.Data(self.model)
        q0 = pin.neutral(self.model)
        q1 = q0.copy()
        q1[self.model.joints[j].idx_q] = 1.0

        pin.forwardKinematics(self.model, data, q0)
        t0 = data.oMi[j].translation.copy()
        pin.forwardKinematics(self.model, data, q1)
        t1 = data.oMi[j].translation.copy()

        # World delta → parent-local delta → placement-local delta
        parent = self.model.parents[j]
        pR_world = data.oMi[parent].rotation  # at q1, but parent is unaffected
        delta_parent = pR_world.T @ (t1 - t0)
        pR = self.model.jointPlacements[j].rotation
        axis_local = pR.T @ delta_parent
        return axis_local

    def _extract_revolute_axis(self, j: int) -> np.ndarray:
        """Get the local revolute axis for joint *j* via finite-difference FK."""
        data = pin.Data(self.model)
        q0 = pin.neutral(self.model)
        q1 = q0.copy()
        q1[self.model.joints[j].idx_q] = 0.01  # small angle

        pin.forwardKinematics(self.model, data, q0)
        R0 = self.model.jointPlacements[j].rotation.T @ (
            data.oMi[self.model.parents[j]].rotation.T @ data.oMi[j].rotation)
        pin.forwardKinematics(self.model, data, q1)
        R1 = self.model.jointPlacements[j].rotation.T @ (
            data.oMi[self.model.parents[j]].rotation.T @ data.oMi[j].rotation)

        # R1 ≈ R_axis(0.01) → log to find axis
        dR = R0.T @ R1
        angle_axis = pin.log3(dR)
        axis = angle_axis / np.linalg.norm(angle_axis)
        return axis

    def compute_sphere_centers_batched(self, q_batch: np.ndarray) -> np.ndarray:
        """Compute world-frame sphere centres for a batch of configurations.

        When vectorised FK is available (all joints are standard revolute
        or prismatic), this method loops over the ~10 kinematic-chain
        joints — **not** over the *N* configurations — making it orders
        of magnitude faster for large *N*.

        Parameters
        ----------
        q_batch : np.ndarray, shape (N, nq)
            Joint configurations (``nq >= 7``; pad with zeros for grippers).

        Returns
        -------
        centers : np.ndarray, shape (N, n_spheres, 3)
        """
        assert self._batched_sphere_data_ready, (
            "Call _precompute_batched_sphere_data() first (requires use_spheres=True)")

        if not self._vectorized_fk_ok:
            return self._compute_sphere_centers_pinocchio(q_batch)

        N = q_batch.shape[0]
        nj = self._fk_njoints

        # Allocate joint transforms: rotation (N, nj, 3, 3), translation (N, nj, 3)
        oR = np.empty((N, nj, 3, 3))
        ot = np.empty((N, nj, 3))

        # Joint 0 (universe) — constant placement
        oR[:, 0] = self._fk_pR[0]
        ot[:, 0] = self._fk_pt[0]

        for j in range(1, nj):
            p = self._fk_parent[j]
            pR = self._fk_pR[j]            # (3, 3)
            pt = self._fk_pt[j]            # (3,)
            parent_R = oR[:, p]            # (N, 3, 3)
            parent_t = ot[:, p]            # (N, 3)
            jtype = self._fk_jtype[j]

            if jtype == 'fixed':
                # oR = parent_R @ pR, ot = parent_R @ pt + parent_t
                oR[:, j] = parent_R @ pR
                ot[:, j] = (parent_R @ pt[:, None]).squeeze(-1) + parent_t

            elif jtype in ('rz', 'rx', 'ry'):
                # Revolute: oR = parent_R @ (pR @ R_axis(θ))
                theta = q_batch[:, self._fk_idx_q[j]]  # (N,)
                compound = _batch_pR_revolute(
                    pR, theta, jtype[-1])  # (N, 3, 3)
                oR[:, j] = np.matmul(parent_R, compound)
                ot[:, j] = (parent_R @ pt[:, None]).squeeze(-1) + parent_t

            elif jtype in ('px', 'py', 'pz'):
                # Prismatic: oR = parent_R @ pR (no joint rotation)
                # ot = parent_R @ (pt + q * pR[:, ax]) + parent_t
                ax = {'px': 0, 'py': 1, 'pz': 2}[jtype]
                q_val = q_batch[:, self._fk_idx_q[j]]   # (N,)
                oR[:, j] = parent_R @ pR
                local_t = pt + q_val[:, None] * pR[:, ax]  # (N, 3)
                ot[:, j] = np.einsum(
                    'nij,nj->ni', parent_R, local_t) + parent_t

            elif jtype == 'p_unaligned':
                # Prismatic along arbitrary local axis
                axis = self._fk_prismatic_axis[j]        # (3,) local axis
                q_val = q_batch[:, self._fk_idx_q[j]]    # (N,)
                oR[:, j] = parent_R @ pR
                local_t = pt + q_val[:, None] * (pR @ axis)  # (N, 3)
                ot[:, j] = np.einsum(
                    'nij,nj->ni', parent_R, local_t) + parent_t

            elif jtype == 'r_unaligned':
                # Revolute about arbitrary local axis
                axis = self._fk_prismatic_axis[j]         # (3,) local axis
                theta = q_batch[:, self._fk_idx_q[j]]     # (N,)
                compound = _batch_pR_revolute_general(
                    pR, theta, axis)  # (N, 3, 3)
                oR[:, j] = np.matmul(parent_R, compound)
                ot[:, j] = (parent_R @ pt[:, None]).squeeze(-1) + parent_t

        # --- Extract sphere centres from joint transforms -----------------
        n_spheres = len(self._batched_sphere_radii)
        centers = np.empty((N, n_spheres, 3))
        for jid, s_idx in self._batched_spheres_by_joint.items():
            R_j = oR[:, jid]            # (N, 3, 3)
            t_j = ot[:, jid]            # (N, 3)
            local = self._batched_sphere_local_translations[s_idx]  # (k, 3)
            # world = R_j @ local^T + t_j  →  (N, k, 3)
            centers[:, s_idx] = np.einsum(
                'nij,kj->nki', R_j, local) + t_j[:, None, :]

        return centers

    def _compute_sphere_centers_pinocchio(self, q_batch: np.ndarray) -> np.ndarray:
        """Fallback: compute sphere centres via Pinocchio FK loop."""
        N = q_batch.shape[0]
        n_spheres = len(self._batched_sphere_radii)
        centers = np.empty((N, n_spheres, 3))

        for i in range(N):
            pin.forwardKinematics(self.model, self.data, q_batch[i])
            for jid, s_idx in self._batched_spheres_by_joint.items():
                oMj = self.data.oMi[jid]
                R = oMj.rotation
                t = oMj.translation
                local = self._batched_sphere_local_translations[s_idx]
                centers[i, s_idx] = local @ R.T + t

        return centers

    def check_collisions_and_depth_batched(
        self,
        traj_np: np.ndarray,
        context: TrajectoryContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised collision checking for a batch of trajectories.

        Instead of calling Pinocchio's per-pair collision / distance
        routines at every ``(b, t)`` pair, this method:

        1. Flattens all configurations to ``(B*T, nq)``.
        2. Computes world-frame robot-sphere centres via FK (loop over
           configs, but sphere extraction is vectorised per joint).
        3. Evaluates **all** self-collision and obstacle distances with
           pure NumPy broadcasting — no Python loop over collision pairs.

        Supported obstacle primitives (read directly from *context*):
        spheres, cuboids (OBB), and cylinders.

        Parameters
        ----------
        traj_np : np.ndarray, shape (B, T, D)
            Joint-space trajectories (``D >= 7``).
        context : TrajectoryContext
            Obstacle context (assumed shared / unbatched for the whole batch).

        Returns
        -------
        collisions : np.ndarray, shape (B,), dtype bool
            ``True`` when *any* timestep of the trajectory has a collision.
        depths : np.ndarray, shape (B,)
            Minimum signed distance per trajectory (most-negative =
            deepest penetration).  Positive means collision-free.
        """
        assert self._batched_sphere_data_ready

        B, T, D = traj_np.shape
        N = B * T

        # --- Flatten & pad to model nq ------------------------------------
        q_flat = traj_np.reshape(N, D)
        if D < self.model.nq:
            q_flat = np.concatenate(
                [q_flat, np.zeros((N, self.model.nq - D))], axis=1)

        # --- Robot sphere centres -----------------------------------------
        robot_centers = self.compute_sphere_centers_batched(q_flat)
        # robot_centers shape: (N, n_spheres, 3)

        # === Self-collision (per joint-pair group) =========================
        min_self = np.full(N, self._same_joint_min_signed)
        for a_arr, b_arr, sum_r_grid, mask in self._self_collision_groups:
            # a_arr: (nA,), b_arr: (nB,)  – small, typically 2-14 spheres
            ca = robot_centers[:, a_arr]          # (N, nA, 3)
            cb = robot_centers[:, b_arr]          # (N, nB, 3)
            # All-pairs distance: (N, nA, nB)
            d = ca[:, :, None, :] - cb[:, None, :, :]   # (N, nA, nB, 3)
            dist_sq = np.einsum('ijkl,ijkl->ijk', d, d)  # (N, nA, nB)
            del d
            np.maximum(dist_sq, 0.0, out=dist_sq)
            np.sqrt(dist_sq, out=dist_sq)
            dist_sq -= sum_r_grid[None, :, :]             # signed distance
            dist_sq -= self.self_collision_margin

            # Mask out non-pair entries with +inf, then reduce
            dist_sq[:, ~mask] = np.inf
            grp_min = dist_sq.reshape(N, -1).min(axis=1)  # (N,)
            np.minimum(min_self, grp_min, out=min_self)
            del dist_sq

        # === Obstacle collision ============================================
        chk_idx = self._batched_obstacle_check_indices
        # (N, n_chk, 3)
        chk_centers = robot_centers[:, chk_idx]
        chk_radii = self._batched_sphere_radii[chk_idx]              # (n_chk,)
        margin = self.obstacle_collision_margin

        min_obs = np.full(N, np.inf)

        # Helper to pull masked obstacle arrays from context
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

        # --- Sphere obstacles ---------------------------------------------
        obs_c = _get("sphere_centers")
        obs_r = _get("sphere_radii")
        if obs_c is not None and obs_r is not None:
            obs_r = obs_r.flatten()                                  # (n_obs,)
            n_obs = len(obs_r)
            # Loop over obstacles to avoid (N, n_chk, n_obs, 3) allocation
            for oi in range(n_obs):
                # (N, n_chk)
                # (N, n_chk, 3)
                d = chk_centers - obs_c[oi]
                dist_so = np.sqrt(np.einsum('ijk,ijk->ij', d, d))
                signed_so = dist_so - chk_radii[None, :] - obs_r[oi] - margin
                min_obs = np.minimum(min_obs, signed_so.min(axis=1))

        # --- Cuboid (OBB) obstacles ---------------------------------------
        box_c = _get("cuboid_centers")
        box_d = _get("cuboid_dims")
        box_q = _get("cuboid_quaternions")
        if box_c is not None and box_d is not None and box_q is not None:
            # (n_box, 3)
            half = np.abs(box_d) / 2.0
            for bi in range(len(box_c)):
                R = quat_wxyz_to_R(box_q[bi])                        # (3, 3)
                h = half[bi]                                         # (3,)
                # Transform to box-local frame
                # (N, n_chk, 3)
                local = (chk_centers - box_c[bi]) @ R
                clamped = np.clip(local, -h, h)
                diff_b = local - clamped
                dist_b_sq = np.einsum('ijk,ijk->ij', diff_b, diff_b)
                dist_b = np.sqrt(dist_b_sq)                  # (N, n_chk)

                # Outside: signed = dist_b - r_robot - margin
                signed_b = dist_b - chk_radii[None, :] - margin
                # Inside box (dist_b ≈ 0): penetration = r + min_face_dist
                inside = dist_b_sq < 1e-20
                if inside.any():
                    # (N, n_chk, 3)
                    face = np.minimum(h - local, local + h)
                    # (N, n_chk)
                    min_face = face.min(axis=-1)
                    signed_b = np.where(
                        inside,
                        -(chk_radii[None, :] + min_face + margin),
                        signed_b,
                    )
                min_obs = np.minimum(min_obs, signed_b.min(axis=1))

        # --- Cylinder obstacles -------------------------------------------
        cyl_c = _get("cylinder_centers")
        cyl_r = _get("cylinder_radii")
        cyl_h = _get("cylinder_heights")
        cyl_q = _get("cylinder_quaternions")
        if (cyl_c is not None and cyl_r is not None
                and cyl_h is not None and cyl_q is not None):
            cyl_r = cyl_r.flatten()
            cyl_h = cyl_h.flatten()
            for ci in range(len(cyl_c)):
                R = quat_wxyz_to_R(cyl_q[ci])
                # (N, n_chk, 3)
                local = (chk_centers - cyl_c[ci]) @ R
                rho = np.sqrt(local[..., 0]**2 +
                              local[..., 1]**2)   # (N, n_chk)
                hz = cyl_h[ci] / 2.0
                d_rad = rho - cyl_r[ci]
                d_ax = np.abs(local[..., 2]) - hz
                # Signed distance to cylinder surface
                out_r = np.maximum(d_rad, 0.0)
                out_z = np.maximum(d_ax, 0.0)
                dist_out = np.sqrt(out_r**2 + out_z**2)
                dist_in = np.maximum(d_rad, d_ax)  # both <= 0 inside
                sd_cyl = np.where(
                    (d_rad <= 0) & (d_ax <= 0), dist_in, dist_out)
                signed_cy = sd_cyl - chk_radii[None, :] - margin
                min_obs = np.minimum(min_obs, signed_cy.min(axis=1))

        # === Aggregate per trajectory =====================================
        all_min = np.minimum(min_self, min_obs).reshape(B, T)
        collisions = (all_min < 0).any(axis=1)                       # (B,)
        depths = all_min.min(axis=1)                                 # (B,)

        return collisions, depths

    def _init_or_refresh_visualizer(self):
        if self.viz is None:
            self.viz = MeshcatVisualizer(
                self.model, self.collision_model, self.visual_model)
            self.viz.initViewer()
            self.viz.loadViewerModel()
            self.viz.display(pin.neutral(self.model))
        else:
            self.viz.loadViewerModel()

        self.viz.displayVisuals(True)
        # self.viz.displayCollisions(True)

    def _refresh_collision_data(self):
        self.collision_data = pin.GeometryData(self.collision_model)
        for req in self.collision_data.collisionRequests:
            req.security_margin = self.obstacle_collision_margin

    def _set_configuration(self, q: np.ndarray):
        """
        Update the robot's configuration in the visualizer.
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.updateGeometryPlacements(
            self.model, self.data,
            self.visual_model, self.visual_data,
            q
        )
        pin.updateGeometryPlacements(
            self.model, self.data,
            self.collision_model, self.collision_data,
            q
        )
        if self.viz is not None:
            self.viz.display(q)

    def _get_ee_pos(self, q: np.ndarray, update_kinematics=True) -> np.ndarray:
        """Compute end-effector position for a given configuration q."""
        if update_kinematics:
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.ee_fid].translation.copy()

    def update_context_obstacles(self, context: TrajectoryContext):
        """
        Updates the collision model with obstacles from the context.
        """
        if self.registered_obstacles:
            for name in list(self.registered_obstacles):
                if self.collision_model.existGeometryName(name):
                    self.collision_model.removeGeometryObject(name)
                # Remove from visualizer if it exists
                if self.viz is not None:
                    try:
                        self.viz.viewer[name].delete()
                    except KeyError:
                        pass
            self.registered_obstacles.clear()
            self.registered_obstacles_geoms.clear()

        # Helper
        def get_items(key):
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

        # Add Cuboids
        centers = get_items("cuboid_centers")
        dims = get_items("cuboid_dims")
        quats = get_items("cuboid_quaternions")

        if centers is not None:
            for i in range(len(centers)):
                name = f"context_cuboid_{i}"
                # Cast to float to ensure compatibility with hppfcl/coal
                d = [abs(x.item()) if hasattr(x, 'item') else abs(float(x))
                     for x in dims[i]]
                shape = hppfcl.Box(*d)
                rot = quat_wxyz_to_R(quats[i])
                placement = pin.SE3(rot, centers[i])
                # name, parent_joint, parent_frame, placement, geometry
                geom = pin.GeometryObject(name, 0, 0, placement, shape)
                geom.color = np.array([1.0, 0.0, 0.0, 0.5])  # Red-ish
                gem_idx = self.collision_model.addGeometryObject(geom)
                self.registered_obstacles.add(name)
                self.registered_obstacles_geoms.add(gem_idx)

                # Add collision pairs between this obstacle and the robot
                # (excluding static links like link0/link1)
                for i in self.obstacle_check_geoms:
                    self.collision_model.addCollisionPair(
                        pin.CollisionPair(i, gem_idx))

                # Add to visualizer
                if self.viz is not None:
                    self.viz.viewer[name].set_object(
                        meshcat.geometry.Box(d))
                    self.viz.viewer[name].set_transform(
                        placement.homogeneous)
                    self.viz.viewer[name].set_property(
                        "color", [1.0, 0.0, 0.0, 0.5])

        # Add Cylinders
        centers = get_items("cylinder_centers")
        radii = get_items("cylinder_radii")
        heights = get_items("cylinder_heights")
        quats = get_items("cylinder_quaternions")

        if centers is not None:
            for i in range(len(centers)):
                name = f"context_cylinder_{i}"
                # Cast to float
                r = radii[i].item() if hasattr(
                    radii[i], 'item') else float(radii[i])
                h = heights[i].item() if hasattr(
                    heights[i], 'item') else float(heights[i])
                shape = hppfcl.Cylinder(r, h)
                rot = quat_wxyz_to_R(quats[i])
                placement = pin.SE3(rot, centers[i])
                # name, parent_joint, parent_frame, placement, geometry
                geom = pin.GeometryObject(name, 0, 0, placement, shape)
                geom.color = np.array([1.0, 0.0, 0.0, 0.5])
                gem_idx = self.collision_model.addGeometryObject(geom)
                self.registered_obstacles.add(name)
                self.registered_obstacles_geoms.add(gem_idx)

                # Add collision pairs (excluding static links like link0/link1)
                for i in self.obstacle_check_geoms:
                    self.collision_model.addCollisionPair(
                        pin.CollisionPair(i, gem_idx))

                # Add to visualizer
                if self.viz is not None:
                    self.viz.viewer[name].set_object(
                        meshcat.geometry.Cylinder(height=h, radius=r))

                    # Meshcat cylinder is Y-up, Pinocchio is Z-up.
                    # We need to apply a rotation to align Meshcat cylinder with Z-axis.
                    # Rotate -90 deg around X axis.
                    R_fix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [
                                     0, -1, 0, 0], [0, 0, 0, 1]])
                    self.viz.viewer[name].set_transform(
                        placement.homogeneous @ R_fix)

                    self.viz.viewer[name].set_property(
                        "color", [1.0, 0.0, 0.0, 0.5])

        # Add Spheres
        centers = get_items("sphere_centers")
        radii = get_items("sphere_radii")
        if centers is not None and radii is not None:
            radii = radii.flatten()
            for i in range(len(centers)):
                name = f"context_sphere_{i}"
                r = radii[i].item() if hasattr(
                    radii[i], 'item') else float(radii[i])
                shape = hppfcl.Sphere(r)
                placement = pin.SE3(
                    np.eye(3), np.array(centers[i], dtype=float))
                geom = pin.GeometryObject(name, 0, 0, placement, shape)
                geom.color = np.array([1.0, 0.0, 0.0, 0.5])
                gem_idx = self.collision_model.addGeometryObject(geom)
                self.registered_obstacles.add(name)
                self.registered_obstacles_geoms.add(gem_idx)

                for j in self.obstacle_check_geoms:
                    self.collision_model.addCollisionPair(
                        pin.CollisionPair(j, gem_idx))

                if self.viz is not None:
                    self.viz.viewer[name].set_object(
                        meshcat.geometry.Sphere(r))
                    self.viz.viewer[name].set_transform(
                        placement.homogeneous)
                    self.viz.viewer[name].set_property(
                        "color", [1.0, 0.0, 0.0, 0.5])

        # Re-create datas
        self.collision_data = pin.GeometryData(self.collision_model)

        self._refresh_collision_data()

        if self.debug:
            self._init_or_refresh_visualizer()

    def _preprocess_traj(self, sample: TrajectorySample) -> np.ndarray:
        """
        Add 2 zeros for the gripper to the trajectory.
        """
        traj = sample.trajectory
        if traj.shape[-1] == 7:
            traj = torch.cat(
                [traj, torch.zeros(*traj.shape[:-1], 2, device=traj.device, dtype=traj.dtype)], dim=-1)
        return traj.detach().cpu().numpy()

    def check_collision(self, sample: TrajectorySample, visualize: bool = False, per_q: bool = False) -> torch.Tensor:
        """
        Check collision for the given sample.
        Returns a boolean tensor of shape [Batch] indicating if each trajectory verifies collision.
        If per_q is True, returns shape [Batch, Timesteps].
        """
        traj = self._preprocess_traj(sample)

        if not sample.is_batched:
            traj = traj[np.newaxis, ...]

        batch_size = traj.shape[0]
        timesteps = traj.shape[1]

        if per_q:
            colliding = np.zeros((batch_size, timesteps), dtype=bool)
        else:
            colliding = np.zeros(batch_size, dtype=bool)

        # If context is NOT batched (shared), update once here
        should_update_global = sample.shared_context or not sample.is_batched
        if should_update_global:
            self.update_context_obstacles(sample.context)

        continue_till_end_of_batch = False
        for b in range(batch_size):
            if not should_update_global:
                # Update obstacles for this specific trajectory's context, as it is not shared
                ctx_b = sample.context.slice(b)
                self.update_context_obstacles(ctx_b)

            for t in range(timesteps):
                q = traj[b, t]
                if len(q) == 7:
                    q = np.concatenate([q, np.zeros(2)])
                pin.forwardKinematics(self.model, self.data, q)
                # Ensure collision data is synced with kinematics
                pin.updateGeometryPlacements(
                    self.model, self.data, self.collision_model, self.collision_data, q)

                if not self.debug:
                    if pin.computeCollisions(self.collision_model, self.collision_data, True):
                        if per_q:
                            colliding[b, t] = True
                        else:
                            colliding[b] = True
                            break  # One collision is enough to mark trajectory as bad
                else:
                    pin.computeCollisions(
                        self.collision_model, self.collision_data, False)
                    pin.computeDistances(
                        self.collision_model, self.collision_data)

                    self_coll = False
                    obs_coll = False
                    coll_list = []

                    for k, cp in enumerate(self.collision_model.collisionPairs):
                        if not self.collision_data.collisionResults[k].isCollision():
                            continue

                        # Get distance
                        dist = self.collision_data.distanceResults[k].min_distance
                        if dist < 0.0:
                            if per_q:
                                colliding[b, t] = True
                            else:
                                colliding[b] = True

                        _a, _b = cp.first, cp.second

                        a_robot = _a in self.robot_geoms
                        b_robot = _b in self.robot_geoms
                        a_obs = _a in self.registered_obstacles_geoms
                        b_obs = _b in self.registered_obstacles_geoms

                        coll_pair_name = f"{self.collision_model.geometryObjects[_a].name} <-> {self.collision_model.geometryObjects[_b].name} (dist: {dist:.4f})"

                        if a_robot and b_robot:
                            self_coll = True
                            coll_list.append(coll_pair_name)
                        elif (a_robot and b_obs) or (b_robot and a_obs):
                            obs_coll = True
                            coll_list.append(coll_pair_name)

                        # if you just need a boolean:
                        if self_coll or obs_coll:
                            colliding[b] = True

                    if self_coll or obs_coll:
                        self._set_configuration(q)
                        print("Collision detected in time step",
                              t, "in trajectory", b, ":", coll_list)
                        if continue_till_end_of_batch:
                            continue
                        cmd = input(
                            "Next step (enter) /  next trajectory (t) / skip to end of batch (b)? ")
                        if cmd == "t":
                            break
                        elif cmd == "b":
                            continue_till_end_of_batch = True
                            break
                        continue
        return torch.tensor(colliding, device=sample.trajectory.device)

    def plot(self,
             sample: TrajectorySample,
             save_path: Optional[str] = None,
             show: bool = False,
             draw_context: bool = True,
             figsize: tuple = (10, 8),
             collisions: Optional[np.ndarray] = None) -> Figure:
        """
        Generate a 3D plot of the end-effector trajectory and context.

        Args:
            sample: The TrajectorySample to plot. Can be batched or unbatched.
            save_path: If provided, save the plot to this path.
            show: Whether to call plt.show().
            draw_context: Whether to draw the context objects.
            figsize: Figure size.
            collisions: Optional boolean array indicating which trajectories are in collision.
                        If provided, colliding trajectories are plotted in black.
                        Shape should match the batch size of the sample.

        Returns:
            fig: The matplotlib figure object.  
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        traj = self._preprocess_traj(sample)

        if not sample.is_batched:
            # Add batch dim for uniform handling
            traj = traj[np.newaxis, ...]

        # Handle collisions argument
        if collisions is not None:
            if torch.is_tensor(collisions):
                collisions = collisions.detach().cpu().numpy()
            collisions = np.array(collisions, dtype=bool)
            if collisions.ndim == 0:
                pass  # Scalar boolean? Unlikely for batch but possible if single sample
            elif len(collisions) != traj.shape[0]:
                print(
                    f"Warning: collisions array shape {collisions.shape} does not match trajectory batch size {traj.shape[0]}. Ignoring collisions.")
                collisions = None

        # Plot trajectories
        for i in range(traj.shape[0]):
            q_traj = traj[i]  # Shape: [T, D]
            ee_positions = []
            for t in range(q_traj.shape[0]):
                q = q_traj[t]
                ee_pos = self._get_ee_pos(q)
                ee_positions.append(ee_pos)

            ee_positions = np.array(ee_positions)

            # Determine color and style
            if collisions is not None and collisions[i]:
                color = 'k'     # Black for collision
                alpha = 0.9     # Slightly more opaque
                linewidth = 1.5  # Slightly thicker
                # Avoid duplicate labels? Maybe just 'Collision'
                label = 'Collision' if i == 0 else None
            else:
                color = "orange"
                alpha = 0.7
                linewidth = 1
                label = f'Traj {i}' if i < 5 else None

            # Plot start and end points
            ax.scatter(ee_positions[0, 0], ee_positions[0, 1],
                       ee_positions[0, 2], c='g', marker='o', s=50)
            ax.scatter(ee_positions[-1, 0], ee_positions[-1, 1],
                       ee_positions[-1, 2], c='r', marker='x', s=50)

            ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], "o-",
                    label=label, alpha=alpha, linewidth=linewidth, markersize=2, color=color)

        # Plot Context
        if draw_context and sample.context is not None:
            self._plot_context(ax, sample.context)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("End Effector Trajectory")

        # Set aspect ratio to equal
        self._set_axes_equal(ax)

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

        if show:
            plt.show()

        plt.close(fig)
        return fig

    def _plot_sphere(self, ax, center, radius):
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)

        radius = radius.item() if hasattr(radius, 'item') else float(radius)

        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

        ax.plot_surface(x, y, z, color='grey', alpha=0.3)

    def plot_joints(self,
                    sample: TrajectorySample,
                    save_path: Optional[str] = None,
                    show: bool = False,
                    figsize: tuple = (15, 15),
                    collisions: Optional[np.ndarray] = None,
                    dt: float = 1.0,
                    pos_limits: Optional[torch.Tensor] = None,
                    vel_limits: Optional[torch.Tensor] = None,
                    acc_limits: Optional[torch.Tensor] = None,
                    jerk_limits: Optional[torch.Tensor] = None) -> Figure:
        """
        Generate a 2D plot of the joint states over time.

        Args:
            sample: The TrajectorySample to plot. Can be batched or unbatched.
            save_path: If provided, save the plot to this path.
            show: Whether to call plt.show().
            figsize: Figure size.
            collisions: Optional boolean array indicating which trajectories are in collision.
                        If provided, colliding trajectories are plotted in black.
                        Shape should match the batch size of the sample.
            dt: Time step for the trajectory.
            pos_limits: Optional tensor of shape (D,) containing the position limits.
            vel_limits: Optional tensor of shape (D,) containing the velocity limits.
            acc_limits: Optional tensor of shape (D,) containing the acceleration limits.
            jerk_limits: Optional tensor of shape (D,) containing the jerk limits.

        Returns:
            fig: The matplotlib figure object.
        """
        # 4 columns: pos, vel, acc, jerk
        fig, axs = plt.subplots(7, 4, figsize=figsize, sharex=True)

        traj = self._preprocess_traj(sample)

        if not sample.is_batched:
            traj = traj[np.newaxis, ...]

        if collisions is not None:
            if torch.is_tensor(collisions):
                collisions = collisions.detach().cpu().numpy()
            collisions = np.array(collisions, dtype=bool)
            if collisions.ndim == 0:
                pass
            elif len(collisions) != traj.shape[0]:
                collisions = None

        for i in range(traj.shape[0]):
            q_traj = traj[i]

            # Compute derivatives
            vel = (q_traj[1:] - q_traj[:-1]) / dt
            acc = (vel[1:] - vel[:-1]) / dt
            jerk = (acc[1:] - acc[:-1]) / dt

            if collisions is not None and collisions[i]:
                color = 'k'
                alpha = 0.9
                linewidth = 1.5
                label = 'Collision' if i == 0 else None
            else:
                color = "orange"
                alpha = 0.7
                linewidth = 1
                label = f'Traj {i}' if i < 5 else None

            for j in range(7):
                # Position
                axs[j, 0].plot(q_traj[:, j], color=color, alpha=alpha,
                               linewidth=linewidth, marker='.', markersize=3, label=label if j == 0 else None)
                # Velocity
                axs[j, 1].plot(vel[:, j], color=color,
                               alpha=alpha, linewidth=linewidth, marker='.', markersize=3)
                # Acceleration
                axs[j, 2].plot(acc[:, j], color=color,
                               alpha=alpha, linewidth=linewidth, marker='.', markersize=3)
                # Jerk
                axs[j, 3].plot(jerk[:, j], color=color,
                               alpha=alpha, linewidth=linewidth, marker='.', markersize=3)

        # Draw limits if provided
        for j in range(7):
            if pos_limits is not None and not torch.isinf(pos_limits[j]):
                axs[j, 0].axhline(y=pos_limits[j].item(),
                                  color='r', linestyle='--', alpha=0.5)
                axs[j, 0].axhline(y=-pos_limits[j].item(),
                                  color='r', linestyle='--', alpha=0.5)
            if vel_limits is not None and not torch.isinf(vel_limits[j]):
                axs[j, 1].axhline(y=vel_limits[j].item(),
                                  color='r', linestyle='--', alpha=0.5)
                axs[j, 1].axhline(y=-vel_limits[j].item(),
                                  color='r', linestyle='--', alpha=0.5)
            if acc_limits is not None and not torch.isinf(acc_limits[j]):
                axs[j, 2].axhline(y=acc_limits[j].item(),
                                  color='r', linestyle='--', alpha=0.5)
                axs[j, 2].axhline(y=-acc_limits[j].item(),
                                  color='r', linestyle='--', alpha=0.5)
            if jerk_limits is not None and not torch.isinf(jerk_limits[j]):
                axs[j, 3].axhline(y=jerk_limits[j].item(),
                                  color='r', linestyle='--', alpha=0.5)
                axs[j, 3].axhline(y=-jerk_limits[j].item(),
                                  color='r', linestyle='--', alpha=0.5)

        # Draw start & goal joint positions
        if sample.context is not None:
            hard_conds = sample.get_hard_conditions()
            start_q = hard_conds["start"]
            goal_q = hard_conds["goal"]
            if torch.is_tensor(start_q):
                start_q = start_q.detach().cpu().numpy()
            if torch.is_tensor(goal_q):
                goal_q = goal_q.detach().cpu().numpy()
            if start_q.ndim > 1:
                start_q = start_q[0]
            if goal_q.ndim > 1:
                goal_q = goal_q[0]
            for j in range(7):
                axs[j, 0].axhline(y=start_q[j], color='green',
                                  linestyle=':', alpha=0.8,
                                  label='Start' if j == 0 else None)
                axs[j, 0].axhline(y=goal_q[j], color='blue',
                                  linestyle=':', alpha=0.8,
                                  label='Goal' if j == 0 else None)

        for j in range(7):
            axs[j, 0].set_ylabel(f'Joint {j+1}')
            for col in range(4):
                axs[j, col].grid(True)

        axs[0, 0].set_title('Position')
        axs[0, 1].set_title('Velocity')
        axs[0, 2].set_title('Acceleration')
        axs[0, 3].set_title('Jerk')

        for col in range(4):
            axs[-1, col].set_xlabel('Time step')

        fig.tight_layout()

        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()

        plt.close(fig)
        return fig

    def plot_joints_chain(self,
                          chain_samples: List[TrajectorySample],
                          save_path: Optional[str] = None,
                          show: bool = False,
                          figsize: tuple = (25, 15),
                          collisions: Optional[np.ndarray] = None,
                          dt: float = 1.0,
                          pos_limits: Optional[torch.Tensor] = None,
                          vel_limits: Optional[torch.Tensor] = None,
                          acc_limits: Optional[torch.Tensor] = None,
                          jerk_limits: Optional[torch.Tensor] = None,
                          idx_list: Optional[List[int]] = None) -> Figure:
        """
        Plot the diffusion chain side-by-side. 
        Positions for each step, and derivatives only for the final step.
        """
        num_steps = len(chain_samples)
        num_cols = num_steps + 3
        fig, axs = plt.subplots(7, num_cols, figsize=figsize, sharex=True)

        if idx_list is None:
            idx_list = list(range(num_steps))

        if collisions is not None:
            if torch.is_tensor(collisions):
                collisions = collisions.detach().cpu().numpy()
            collisions = np.array(collisions, dtype=bool)
            if collisions.ndim == 0:
                pass
            elif len(collisions) != chain_samples[0].trajectory.shape[0]:
                collisions = None

        for step_idx, sample in enumerate(chain_samples):
            traj = self._preprocess_traj(sample)
            if not sample.is_batched:
                traj = traj[np.newaxis, ...]

            for i in range(traj.shape[0]):
                q_traj = traj[i]

                if collisions is not None and collisions[i]:
                    color = 'k'
                    alpha = 0.9
                    linewidth = 1.5
                    label = 'Collision' if i == 0 else None
                else:
                    color = "orange"
                    alpha = 0.7
                    linewidth = 1
                    label = f'Traj {idx_list[step_idx]}' if i < 5 else None

                for j in range(7):
                    axs[j, step_idx].plot(q_traj[:, j], color=color, alpha=alpha,
                                          linewidth=linewidth, marker='.', markersize=3, label=label if j == 0 else None)

                # If this is the last step, also plot derivatives
                if step_idx == num_steps - 1:
                    vel = (q_traj[1:] - q_traj[:-1]) / dt
                    acc = (vel[1:] - vel[:-1]) / dt
                    jerk = (acc[1:] - acc[:-1]) / dt

                    for j in range(7):
                        axs[j, num_steps].plot(
                            vel[:, j], color=color, alpha=alpha, linewidth=linewidth, marker='.', markersize=3)
                        axs[j, num_steps + 1].plot(acc[:, j], color=color, alpha=alpha,
                                                   linewidth=linewidth, marker='.', markersize=3)
                        axs[j, num_steps + 2].plot(jerk[:, j], color=color, alpha=alpha,
                                                   linewidth=linewidth, marker='.', markersize=3)

        # Draw limits
        for j in range(7):
            for step_idx in range(num_steps):
                if pos_limits is not None and not torch.isinf(pos_limits[j]):
                    axs[j, step_idx].axhline(
                        y=pos_limits[j].item(), color='r', linestyle='--', alpha=0.5)
                    axs[j, step_idx].axhline(
                        y=-pos_limits[j].item(), color='r', linestyle='--', alpha=0.5)

            if vel_limits is not None and not torch.isinf(vel_limits[j]):
                axs[j, num_steps].axhline(
                    y=vel_limits[j].item(), color='r', linestyle='--', alpha=0.5)
                axs[j, num_steps].axhline(
                    y=-vel_limits[j].item(), color='r', linestyle='--', alpha=0.5)
            if acc_limits is not None and not torch.isinf(acc_limits[j]):
                axs[j, num_steps+1].axhline(y=acc_limits[j].item(),
                                            color='r', linestyle='--', alpha=0.5)
                axs[j, num_steps+1].axhline(y=-acc_limits[j].item(),
                                            color='r', linestyle='--', alpha=0.5)
            if jerk_limits is not None and not torch.isinf(jerk_limits[j]):
                axs[j, num_steps+2].axhline(y=jerk_limits[j].item(),
                                            color='r', linestyle='--', alpha=0.5)
                axs[j, num_steps+2].axhline(y=-jerk_limits[j].item(),
                                            color='r', linestyle='--', alpha=0.5)

        # Draw context
        sample = chain_samples[0]
        if sample.context is not None:
            hard_conds = sample.get_hard_conditions()
            start_q = hard_conds["start"]
            goal_q = hard_conds["goal"]
            if torch.is_tensor(start_q):
                start_q = start_q.detach().cpu().numpy()
            if torch.is_tensor(goal_q):
                goal_q = goal_q.detach().cpu().numpy()
            if start_q.ndim > 1:
                start_q = start_q[0]
            if goal_q.ndim > 1:
                goal_q = goal_q[0]
            for j in range(7):
                for step_idx in range(num_steps):
                    axs[j, step_idx].axhline(
                        y=start_q[j], color='green', linestyle=':', alpha=0.8, label='Start' if j == 0 else None)
                    axs[j, step_idx].axhline(
                        y=goal_q[j], color='blue', linestyle=':', alpha=0.8, label='Goal' if j == 0 else None)

        for j in range(7):
            axs[j, 0].set_ylabel(f'Joint {j+1}')
            for col in range(num_cols):
                axs[j, col].grid(True)

        for step_idx in range(num_steps):
            axs[0, step_idx].set_title(f'Pos (Step {step_idx})')
        axs[0, num_steps].set_title('Velocity')
        axs[0, num_steps+1].set_title('Acceleration')
        axs[0, num_steps+2].set_title('Jerk')

        for col in range(num_cols):
            axs[-1, col].set_xlabel('Time step')

        fig.tight_layout()

        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()

        plt.close(fig)
        return fig

    def _plot_context(self, ax, context: TrajectoryContext):
        """Helper to plot context objects."""
        # Helper to get tensor data
        def get_data(key):
            try:
                # Use slice(0) to get first batch element if batched
                ctx_to_use = context
                if context.is_batched:
                    ctx_to_use = context.slice(0)

                # Check for mask
                mask = ctx_to_use.get_mask(key)  # Shape [N_objects]
                # Shape [N_objects, Features]
                data = ctx_to_use.get_item(key)

                if torch.is_tensor(data):
                    data = data.detach().cpu().numpy()
                if torch.is_tensor(mask):
                    mask = mask.detach().cpu().numpy()

                return data[mask.astype(bool)]
            except KeyError:
                return None

        cuboids_centers = get_data('cuboid_centers')
        cuboids_dims = get_data('cuboid_dims')
        cuboids_quaternions = get_data('cuboid_quaternions')
        if cuboids_centers is not None:
            for i in range(len(cuboids_centers)):
                self._plot_cuboid(
                    ax, cuboids_centers[i], cuboids_dims[i], cuboids_quaternions[i])

        cylinders_centers = get_data('cylinder_centers')
        cylinders_radii = get_data('cylinder_radii')
        cylinders_heights = get_data('cylinder_heights')
        cylinders_quaternions = get_data('cylinder_quaternions')
        if cylinders_centers is not None:
            for i in range(len(cylinders_centers)):
                self._plot_cylinder(
                    ax, cylinders_centers[i], cylinders_radii[i], cylinders_heights[i], cylinders_quaternions[i])

        spheres_centers = get_data('sphere_centers')
        spheres_radii = get_data('sphere_radii')
        if spheres_centers is not None:
            for i in range(len(spheres_centers)):
                self._plot_sphere(ax, spheres_centers[i], spheres_radii[i])

    def _plot_cuboid(self, ax, center, dims, quat):
        rot = pin.Quaternion(
            float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])).toRotationMatrix()

        # Corners
        corners = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5]
        ]) * dims

        corners = corners @ rot.T + center

        # Faces
        faces = [
            [corners[0], corners[1], corners[2], corners[3]],
            [corners[4], corners[5], corners[6], corners[7]],
            [corners[0], corners[1], corners[5], corners[4]],
            [corners[2], corners[3], corners[7], corners[6]],
            [corners[1], corners[2], corners[6], corners[5]],
            [corners[4], corners[7], corners[3], corners[0]]
        ]

        # Plot faces using Poly3DCollection
        poly3d = Poly3DCollection(
            faces, alpha=0.25, linewidths=1, edgecolors=(0, 0, 0, 0.3))
        poly3d.set_facecolor('grey')
        ax.add_collection3d(poly3d)

    def _plot_cylinder(self, ax, center, radius, height, quat):
        radius = radius.item() if hasattr(radius, 'item') else float(radius)
        height = height.item() if hasattr(height, 'item') else float(height)
        rot = pin.Quaternion(
            float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])).toRotationMatrix()

        z = np.linspace(-height/2, height/2, 10)
        theta = np.linspace(0, 2*np.pi, 20)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)

        N = x_grid.size
        points = np.stack([x_grid.flatten(), y_grid.flatten(),
                           z_grid.flatten()], axis=1)  # [N, 3]
        points = points @ rot.T + center

        X = points[:, 0].reshape(x_grid.shape)
        Y = points[:, 1].reshape(y_grid.shape)
        Z = points[:, 2].reshape(z_grid.shape)

        ax.plot_surface(X, Y, Z, alpha=0.3, color='grey')

        # Caps
        # Top cap (z = height/2)
        r = np.array([0, radius])

        # Create a grid for the cap
        R, Theta = np.meshgrid(r, theta)
        X_cap = R * np.cos(Theta)
        Y_cap = R * np.sin(Theta)

        # Top cap
        Z_cap_top = np.full_like(X_cap, height/2)
        points_top = np.stack(
            [X_cap.flatten(), Y_cap.flatten(), Z_cap_top.flatten()], axis=1)
        points_top = points_top @ rot.T + center
        X_top = points_top[:, 0].reshape(X_cap.shape)
        Y_top = points_top[:, 1].reshape(Y_cap.shape)
        Z_top = points_top[:, 2].reshape(X_cap.shape)
        ax.plot_surface(X_top, Y_top, Z_top, alpha=0.3, color='grey')

        # Bottom cap
        Z_cap_bot = np.full_like(X_cap, -height/2)
        points_bot = np.stack(
            [X_cap.flatten(), Y_cap.flatten(), Z_cap_bot.flatten()], axis=1)
        points_bot = points_bot @ rot.T + center
        X_bot = points_bot[:, 0].reshape(X_cap.shape)
        Y_bot = points_bot[:, 1].reshape(Y_cap.shape)
        Z_bot = points_bot[:, 2].reshape(X_cap.shape)
        ax.plot_surface(X_bot, Y_bot, Z_bot, alpha=0.3, color='grey')

        # Edges of the cylinder
        theta_edge = np.linspace(0, 2 * np.pi, 100)
        for z_val in [height / 2, -height / 2]:
            pts_edge = np.stack([radius * np.cos(theta_edge), radius * np.sin(theta_edge),
                                np.full_like(theta_edge, z_val)], axis=1)
            pts_edge = pts_edge @ rot.T + center
            ax.plot(pts_edge[:, 0], pts_edge[:, 1],
                    pts_edge[:, 2], color='black', alpha=0.3)
        for t in [0, np.pi/2, np.pi, 3*np.pi/2]:
            pts_v = np.array([[radius * np.cos(t), radius * np.sin(t), -height/2],
                              [radius * np.cos(t), radius * np.sin(t), height/2]])
            pts_v = pts_v @ rot.T + center
            ax.plot(pts_v[:, 0], pts_v[:, 1], pts_v[:, 2],
                    color='black', alpha=0.3)

    def _set_axes_equal(self, ax):
        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


# ----------------------------
# Example trajectory and Verification
# ----------------------------
if __name__ == "__main__":
    import torch

    # Setup paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(root_dir, "assets", "robots",
                             "franka", "urdf", "franka_panda.urdf")

    # Directory that contains the mesh files referenced by the URDF
    mesh_dir = os.path.dirname(urdf_path)

    print(f"Loading visualizer with URDF: {urdf_path}")
    if not os.path.exists(urdf_path):
        print(f"ERROR: URDF not found at {urdf_path}")
        exit(1)

    viz = FrankaPinocchioModel(
        urdf_path, mesh_dir, debug=True, use_spheres=True)

    # random trajectory
    q_neutral = [
        0.0,                 # q1
        -0.78539816339,      # q2  (-π/4)
        0.0,                # q3
        -2.35619449019,      # q4  (-3π/4)
        0.0,                # q5
        1.57079632679,      # q6  (π/2)
        0.78539816339,      # q7  (π/4)
        0.0,                # gripper
        0.0                 # gripper
    ]
    q0 = torch.tensor(q_neutral)
    q_goal = q0 + 2 * torch.randn(viz.model.nq)
    steps = 64
    t = torch.linspace(0, 1, steps).unsqueeze(1)
    q_traj_1 = q0 * (1 - t) + q_goal * t
    # sine
    q_traj_2 = q_traj_1 + 0.5 * \
        torch.sin(torch.linspace(0, 4 * np.pi, steps)).unsqueeze(1)

    q_traj = torch.stack([q_traj_1, q_traj_2], dim=0)
    q_traj[..., -2:] = 0.0

    # Force collision for trajectory 1:
    # Get midpoint of traj 1
    coll_idx = int(steps // (3/2))
    q_coll = q_traj_2[coll_idx].numpy()
    ee_coll = viz._get_ee_pos(q_coll)
    print(f"Placing obstacle at {ee_coll} to force collision on Traj 1")

    # Define validation collisions (ground truth for this verification)
    expected_coll = torch.tensor([False, True])

    context = TrajectoryContext({
        "cuboid_centers": torch.tensor(np.array([ee_coll]), dtype=torch.float32),
        "cuboid_dims": torch.tensor([[0.1, 0.1, 0.1]]),
        "cuboid_quaternions": torch.tensor([[1, 0, 0, 0]]),
        "cylinder_centers": torch.tensor([[0.5, 0.5, 0.5]]),
        "cylinder_radii": torch.tensor([[0.1]]),
        "cylinder_heights": torch.tensor([[0.1]]),
        "cylinder_quaternions": torch.tensor([[1, 0, 0, 0]]),
    }, start=q0, goal=q_traj[-1], is_batched=False)

    sample = TrajectorySample(
        trajectory=q_traj, context=context, is_batched=True)

    # 1. Verify Collision Checking
    print("Verifying Collision Checking...")
    start_time = time.time()
    computed_coll = viz.check_collision(sample)
    end_time = time.time()
    print(f"Computed Collisions: {computed_coll}")
    print(f"Collision Checking Time: {end_time - start_time:.6f} seconds")

    # 2. Visualize
    output_path = "trajectory_verification.png"
    print(f"Plotting to {output_path}...")
    start_time = time.time()
    viz.plot(sample, show=False, save_path=output_path,
             collisions=computed_coll)
    end_time = time.time()
    print(f"Plotting Time: {end_time - start_time:.6f} seconds")

    if os.path.exists(output_path):
        print("Success: Plot generated.")
    else:
        print("Error: Plot not found.")

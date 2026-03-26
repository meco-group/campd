import os
import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
import einops
from flask import Flask, request, jsonify

from campd.models.diffusion.model import ContextTrajectoryDiffusionModel, SamplingCfg
from campd.data.context import TrajectoryContext

from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.geom.types import WorldConfig, Cuboid, Cylinder, Sphere
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig


# -----------------------------------------------------------------------------
# Curobo Collision Checker and Utilities
# -----------------------------------------------------------------------------

def load_world_cfg(obj_list_dict_numpy, pad=False, nb_cuboids_cache=40, nb_cylinders_cache=12, nb_spheres_cache=0, add_floor=False):
    cuboids = []
    cylinders = []
    spheres = []

    if 'spheres' in obj_list_dict_numpy:
        for idx, (center, radius) in enumerate(zip(obj_list_dict_numpy['spheres']['centers'], obj_list_dict_numpy['spheres']['radii'])):
            cuboids.append(Cuboid(pose=[center[0], center[1], center[2], 1, 0, 0, 0], dims=[
                           2*radius/np.sqrt(3), 2*radius/np.sqrt(3), 2*radius/np.sqrt(3)], name=f'sphere_{idx}'))
    if 'boxes' in obj_list_dict_numpy:
        for idx, (center, size, quat) in enumerate(zip(obj_list_dict_numpy['boxes']['centers'], obj_list_dict_numpy['boxes']['sizes'], obj_list_dict_numpy.get('boxes', {}).get('quats', [[1, 0, 0, 0]]*len(obj_list_dict_numpy['boxes']['centers'])))):
            if 'quats' not in obj_list_dict_numpy['boxes']:
                quat = [1, 0, 0, 0]
            cuboids.append(Cuboid(pose=[center[0], center[1], center[2], quat[0],
                                        quat[1], quat[2], quat[3]], dims=size, name=f'box_{idx}'))
    if 'cylinders' in obj_list_dict_numpy:
        for idx, (center, height, radius, quat) in enumerate(zip(obj_list_dict_numpy['cylinders']['centers'], obj_list_dict_numpy['cylinders']['heights'], obj_list_dict_numpy['cylinders']['radii'], obj_list_dict_numpy['cylinders']['quats'])):
            cylinders.append(Cylinder(pose=[center[0], center[1], center[2], quat[0], quat[1],
                             quat[2], quat[3]], radius=radius, height=height, name=f'cylinder_{idx}'))
    if add_floor:
        cuboids.append(Cuboid(pose=[0, 0, -0.1, 1, 0, 0, 0], dims=[
                       1, 1, 0.1], name='floor'))

    if pad:
        for i in range(len(cuboids), nb_cuboids_cache):
            cuboids.append(Cuboid(
                name=f'cuboid_{i}',
                pose=[0, 0, 0, 1, 0, 0, 0],
                dims=[0, 0, 0]
            ))
        for i in range(len(cylinders), nb_cylinders_cache):
            cylinders.append(Cylinder(
                name=f'cylinder_{i}',
                pose=[0, 0, 0, 1, 0, 0, 0],
                radius=0,
                height=0
            ))
        for i in range(len(spheres), nb_spheres_cache):
            spheres.append(Sphere(
                name=f'sphere_{i}',
                pose=[0, 0, 0, 1, 0, 0, 0],
                radius=0
            ))

    world_cfg = WorldConfig(
        cuboid=cuboids,
        cylinder=cylinders,
        sphere=spheres,
    )
    return world_cfg


class CuroboFrankaCC:
    def __init__(self, nb_cuboids_cache=40, nb_cylinders_cache=12, nb_spheres_cache=0, tensor_args={}):
        self.tensor_args = TensorDeviceType(**tensor_args)

        # Retrieve paths from campd or directly (fallback)
        robot_yaml_path = os.path.join(os.path.dirname(
            __file__), '..', '..', 'examples', 'franka_curobo', 'assets', 'robots', 'franka', 'configs', 'franka.yml')
        if not os.path.exists(robot_yaml_path):
            # Hardcode a generic relative fallback if needed
            robot_yaml_path = 'src/campd/examples/franka_curobo/assets/robots/franka/configs/franka.yml'
            if not os.path.exists(robot_yaml_path):
                robot_yaml_path = os.path.abspath(os.path.join(os.path.dirname(
                    __file__), '..', '..', '..', 'examples', 'franka_curobo', 'assets', 'robots', 'franka', 'configs', 'franka.yml'))

        try:
            self.robot_data = load_yaml(robot_yaml_path)["robot_cfg"]
        except Exception as e:
            # Default generic if missing
            self.robot_data = load_yaml(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(__file__)))), "examples", "franka_curobo", "assets", "robots", "franka", "configs", "franka.yml"))["robot_cfg"]

        self.robot_cfg = RobotConfig.from_dict(
            self.robot_data, self.tensor_args)

        self.world_cfg = load_world_cfg({}, pad=True, nb_cuboids_cache=nb_cuboids_cache,
                                        nb_cylinders_cache=nb_cylinders_cache, nb_spheres_cache=nb_spheres_cache)

        config = RobotWorldConfig.load_from_config(
            robot_yaml_path, self.world_cfg, collision_activation_distance=0.0
        )
        self.curobo_fn = RobotWorld(config)

    def check_collision(self, trajs: torch.Tensor, obj_list_dict=None):
        if obj_list_dict is not None:
            world_cfg = load_world_cfg(
                obj_list_dict, pad=True, nb_cuboids_cache=40, nb_cylinders_cache=12, nb_spheres_cache=0)
        else:
            world_cfg = self.world_cfg

        self.curobo_fn.update_world(world_cfg)

        nb_trajs = trajs.shape[0]
        q_s = einops.rearrange(trajs, 'b s j -> (b s) j')
        d_world, d_self = self.curobo_fn.get_world_self_collision_distance_from_joints(
            q_s)
        d_world = einops.rearrange(d_world, '(b s) -> b s', b=nb_trajs)
        d_self = einops.rearrange(d_self, '(b s) -> b s', b=nb_trajs)
        return d_world, d_self


# -----------------------------------------------------------------------------
# Metric Functions
# -----------------------------------------------------------------------------

def compute_smoothness(trajs, dt=1.0):
    # trajs: (B, T, D)
    if isinstance(trajs, list):
        trajs = torch.stack(trajs)
    vel = torch.diff(trajs, dim=1) / dt
    acc = torch.diff(vel, dim=1) / dt
    jerk = torch.diff(acc, dim=1) / dt
    return torch.mean(torch.abs(jerk), dim=(1, 2))


def compute_path_length(trajs):
    # trajs: (B, T, D)
    if isinstance(trajs, list):
        trajs = torch.stack(trajs)
    diff = torch.diff(trajs, dim=1)
    return torch.sum(torch.norm(diff, dim=-1), dim=1)


def compute_shortest_duration(traj, robot_vel_max=None):
    # traj: (T, D)
    if robot_vel_max is None:
        # Default Franka approx velocity limits
        robot_vel_max = torch.tensor(
            [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], device=traj.device)
    diff = torch.abs(torch.diff(traj, dim=0))
    dt_required = diff / robot_vel_max.unsqueeze(0)
    dt_max = torch.max(dt_required)
    return dt_max * (traj.shape[0] - 1)


# -----------------------------------------------------------------------------
# Inference Server
# -----------------------------------------------------------------------------

class InferenceServer:
    def __init__(self, model_dir, device='cuda', debug=False):
        self.model_dir = model_dir
        self.device = device
        self.debug = debug

        self.model = ContextTrajectoryDiffusionModel.from_pretrained(
            model_dir=self.model_dir,
            device=self.device,
            freeze_params=True
        )

        self.collision_checker = None
        self.is_buffer_prepped = False

        self._debug_save_lock = threading.Lock()
        self.debug_executor = ThreadPoolExecutor(max_workers=1)

        self.last_trajectory = None
        self.last_goal = None
        self.last_request_time = 0

        # Will be populated with proper Context dict conversions
        self.tensor_args = {'dtype': torch.float32, 'device': self.device}

        self._previous_sampling_cfg = None

        self.init_server()

    def _check_sampling_cfg_changed(self, sampling_cfg: SamplingCfg):
        if self._previous_sampling_cfg is None:
            return True
        for key, value in sampling_cfg.dict().items():
            if hasattr(self._previous_sampling_cfg, key):
                if getattr(self._previous_sampling_cfg, key) != value:
                    return True
            else:
                return True
        return False

    def init_server(self):
        self.app = Flask(__name__)

        @self.app.route('/inference_release2', methods=['POST'])
        def inference_release():
            data = request.json
            mode = data.get('mode', 'position')
            n_samples = data['n_samples']
            n_support_points = data['n_support_points']
            selection_method = data.get('selection_method', 'best')
            obj_list_dict = data['environment_data']['obj_list_dict']

            start_state_pos = torch.tensor(
                data['start_state_pos'], **self.tensor_args)
            goal_state_pos = torch.tensor(
                data['goal_state_pos'], **self.tensor_args)

            cond_scale = data.get('cond_scale', 1.0)

            # Use defaults from config for sampling if needed
            sampling_cfg = SamplingCfg(
                n_support_points=n_support_points,
                batch_size=n_samples,
                return_chain=False,
                cond_scale=cond_scale,
                use_cuda_graph=True,
                context_buffer_sizes={
                    "spheres": 40, "boxes": 40, "cylinders": 12}  # Defaults approx
            )

            # check if sampling config has changed
            if self._check_sampling_cfg_changed(sampling_cfg):
                self.model.prepare_for_sampling(sampling_cfg)
                self._previous_sampling_cfg = sampling_cfg

            try:
                result = self.inference_server_release(
                    n_samples, n_support_points, cond_scale, selectionmethod=selection_method,
                    obj_list_dict=obj_list_dict, start_state_pos=start_state_pos, goal_state_pos=goal_state_pos, mode=mode)
            except Exception as e:
                import traceback
                print(e)
                traceback.print_exc()
                return jsonify({"error": "Could not find collision free trajectory", "details": str(e)}), 200
            return jsonify(result), 200

    def check_collisions(self, trajs, obj_list_dict):
        """
        Check collisions for a batch of trajectories.
        Returns:
            collision_free_trajs: Tensor of collision-free trajectories
            d_world: Distance to world obstacles
            d_self: Distance to self-collision
        """
        if trajs is None or len(trajs) == 0:
            return [], None, None

        try:
            if self.collision_checker is None:
                self.collision_checker = CuroboFrankaCC(
                    tensor_args={'device': self.device})

            if trajs.ndim == 2:
                trajs = trajs.unsqueeze(0)

            d_world, d_self = self.collision_checker.check_collision(
                trajs, obj_list_dict=obj_list_dict)

            collision_free_mask = (d_world.max(dim=1)[0] <= 0) & (
                d_self.max(dim=1)[0] <= 0)
            collision_free_indices = torch.where(collision_free_mask)[0]

            if len(collision_free_indices) > 0:
                collision_free_trajs = trajs[collision_free_indices]
            else:
                collision_free_trajs = []

            return collision_free_trajs, d_world, d_self

        except Exception as e:
            print(f"Collision checking failed: {e}")
            import traceback
            traceback.print_exc()
            return [], None, None

    def _obj_list_to_context(self, obj_list_dict, start, goal, n_samples):
        # Build batched Context dictionary
        # We need to map dict arrays to batched tensors
        context_data = {}
        if 'spheres' in obj_list_dict:
            centers = torch.tensor(
                obj_list_dict['spheres']['centers'], **self.tensor_args)
            radii = torch.tensor(
                obj_list_dict['spheres']['radii'], **self.tensor_args).reshape(-1, 1)

            tensor_spheres = torch.cat([centers, radii], dim=1).unsqueeze(
                0).repeat(n_samples, 1, 1)
            context_data['spheres'] = tensor_spheres

            # Also append to multivariates in standard models? Usually campd separates keys directly.

        # Add batches
        batch_start = start.unsqueeze(0).repeat(n_samples, 1)
        batch_goal = goal.unsqueeze(0).repeat(n_samples, 1)

        ctx = TrajectoryContext(
            data=context_data,
            start=batch_start,
            goal=batch_goal,
            is_batched=True,
            is_normalized=False
        )
        return ctx

    def inference_server_release(self, n_samples, n_support_points, cond_scale=1.0, selectionmethod='best', obj_list_dict=None, start_state_pos=None, goal_state_pos=None, mode='position'):
        s = time.time()

        ctx = self._obj_list_to_context(
            obj_list_dict, start_state_pos, goal_state_pos, n_samples)

        ctx_normalized = ctx.normalize(self.model.normalizer)
        # Sample

        assert ctx_normalized.is_normalized
        samples = self.model.sample(ctx_normalized)
        assert not samples.is_normalized
        trajs = samples.trajectory  # Should be shape (B, T, D)
        t_total = time.time() - s

        # Check collisions
        collision_free_trajs, d_world, d_self = self.check_collisions(
            trajs, obj_list_dict)

        if len(collision_free_trajs) > 0:
            if isinstance(collision_free_trajs, list):
                collision_free_trajs = torch.stack(collision_free_trajs)
            smoothness = compute_smoothness(collision_free_trajs)
            path_length = compute_path_length(collision_free_trajs)

            if selectionmethod == 'random':
                idx = torch.randint(0, len(collision_free_trajs), (1,))
                best_traj = collision_free_trajs[idx]
            elif selectionmethod == 'smoothest':
                best_idx = torch.argmin(smoothness)
                best_traj = collision_free_trajs[best_idx]
            elif selectionmethod == 'shortest':
                best_idx = torch.argmin(path_length)
                best_traj = collision_free_trajs[best_idx]
            else:
                best_traj = collision_free_trajs[0]
        else:
            best_traj = None

        if best_traj is not None:
            best_motion_time = compute_shortest_duration(best_traj).tolist()
            if mode == 'position':
                best_traj_val = best_traj
            else:
                import flask
                return flask.abort(400, "Velocity mode not supported yet")

            if best_traj.ndim == 3:
                best_traj_val = best_traj_val.squeeze(0)

            out = dict(
                traj_val=best_traj_val.tolist(),
                t_total=t_total,
                collision_free_count=len(collision_free_trajs),
                best_motion_time=best_motion_time
            )
            self.last_trajectory = best_traj
            self.last_goal = goal_state_pos
            self.last_request_time = time.time()
        else:
            out = dict(
                traj_val=None,
                t_total=t_total,
                collision_free_count=0,
                best_motion_time=None,
                error="No valid trajectory found"
            )

        return out

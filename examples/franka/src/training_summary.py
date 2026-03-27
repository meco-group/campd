from campd.data import TrajectoryContext
from campd.models import SamplingCfg
from campd.training.summary import Summary
from campd.models.diffusion import ContextTrajectoryDiffusionModel
from torch.utils.data import DataLoader
from campd.data import TrajectorySample
from campd.training.registry import SUMMARIES
from typing import List

from robot_model import FrankaPinocchioModel
from inference_validator import FrankaTrajectoryValidator
import os
import numpy as np


@SUMMARIES.register("ValidationSummary")
class ValidationSummary(Summary):
    """
    An example custom summary that prints a message during training.
    """

    def __init__(self, every_n_steps: int = 1, run_first: bool = False, train_indices: List[int] = None, val_indices: List[int] = None, sample_cfg: SamplingCfg = None, dt: float = 1.0,
                 plot_chain: bool = False, plot_chain_spacing: int = 1,
                 n_samples_per_context: int = 10,
                 validator_kwargs: dict = None):
        super().__init__(every_n_steps=every_n_steps, run_first=run_first)
        self.train_indices = train_indices if train_indices is not None else [
            0, 1, 2]
        self.val_indices = val_indices if val_indices is not None else [
            0, 1, 2]

        if sample_cfg is None:
            sample_cfg = {
                "batch_size": 1,
                "n_support_points": 64,
                "return_chain": plot_chain
            }
        else:
            if isinstance(sample_cfg, dict):
                sample_cfg["return_chain"] = True
        self.sample_cfg = SamplingCfg.model_validate(sample_cfg)
        self.dt = dt
        self.plot_chain = plot_chain
        self.plot_chain_spacing = plot_chain_spacing
        self.n_samples_per_context = n_samples_per_context

        root_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        urdf_path = os.path.join(
            root_dir, "assets", "urdf", "franka_panda.urdf")

        # Directory that contains the mesh files referenced by the URDF
        mesh_dir = os.path.dirname(urdf_path)
        self.robot_model = FrankaPinocchioModel(urdf_path, mesh_dir)

        # Validator for per-context evaluation
        _vkw = validator_kwargs if validator_kwargs is not None else {}
        _vkw.setdefault("dt", dt)
        self.validator = FrankaTrajectoryValidator(**_vkw)

    def _run(
        self,
        model: ContextTrajectoryDiffusionModel,
        train_dataloader: DataLoader[TrajectorySample],
        val_dataloader: DataLoader[TrajectorySample],
        step: int,
    ):
        out = {}
        # set timesteps for inference in model
        model.prepare_for_sampling(self.sample_cfg)

        # inference for train_indices
        for idx in self.train_indices:
            sample = train_dataloader.dataset[idx]
            sample.to(model.device, non_blocking=True)

            gen_ret = model.sample(sample.context)
            if self.sample_cfg.return_chain:
                gen, chain = gen_ret
            else:
                gen, chain = gen_ret, None

            coll = self.robot_model.check_collision(gen)
            feasible_traj_rate = (coll == 0).cpu().float().mean()

            out[f"train_{idx}"] = gen
            out[f"train_{idx}_feasible_traj_rate"] = feasible_traj_rate

            fig = self.robot_model.plot(gen, collisions=coll)
            out[f"train_{idx}_trajectory"] = fig

            fig_joints = self.robot_model.plot_joints(
                gen, collisions=coll, dt=self.dt)
            out[f"train_{idx}_joints"] = fig_joints

            if chain is not None:
                chain_samples = []
                idx_list = []
                for step_idx in range(chain.shape[0]):
                    if step_idx % self.plot_chain_spacing != 0 and step_idx != 0 and step_idx != chain.shape[0] - 1:
                        continue
                    step_traj = chain[step_idx]
                    step_sample = TrajectorySample(
                        trajectory=step_traj,
                        context=gen.context,
                        is_normalized=gen.is_normalized,
                        is_batched=gen.is_batched,
                        shared_context=gen.shared_context
                    )
                    chain_samples.append(step_sample)
                    idx_list.append(step_idx)

                fig_chain = self.robot_model.plot_joints_chain(
                    chain_samples, collisions=coll, dt=self.dt, idx_list=idx_list)
                out[f"train_{idx}_joints_chain"] = fig_chain
            else:
                fig = self.robot_model.plot_joints(
                    gen, collisions=coll, dt=self.dt)
                out[f"train_{idx}_joints"] = fig

        # inference for val_indices
        for idx in self.val_indices:
            sample = val_dataloader.dataset[idx]
            sample.to(model.device, non_blocking=True)
            gen_ret = model.sample(sample.context)
            if self.sample_cfg.return_chain:
                gen, chain = gen_ret
            else:
                gen, chain = gen_ret, None

            coll = self.robot_model.check_collision(gen)
            feasible_traj_rate = (coll == 0).cpu().float().mean()
            out[f"val_{idx}"] = gen
            out[f"val_{idx}_feasible_traj_rate"] = feasible_traj_rate

            fig = self.robot_model.plot(gen, collisions=coll)
            out[f"val_{idx}_trajectory"] = fig

            fig_joints = self.robot_model.plot_joints(
                gen, collisions=coll, dt=self.dt)
            out[f"val_{idx}_joints"] = fig_joints

            if chain is not None:
                chain_samples = []
                idx_list = []
                for step_idx in range(chain.shape[0]):
                    if step_idx % self.plot_chain_spacing != 0 and step_idx != 0 and step_idx != chain.shape[0] - 1:
                        continue
                    step_traj = chain[step_idx]
                    step_sample = TrajectorySample(
                        trajectory=step_traj,
                        context=gen.context,
                        is_normalized=gen.is_normalized,
                        is_batched=gen.is_batched,
                        shared_context=gen.shared_context
                    )
                    chain_samples.append(step_sample)
                    idx_list.append(step_idx)

                fig_chain = self.robot_model.plot_joints_chain(
                    chain_samples, collisions=coll, dt=self.dt, idx_list=idx_list)
                out[f"val_{idx}_joints_chain"] = fig_chain
            else:
                fig = self.robot_model.plot_joints(
                    gen, collisions=coll, dt=self.dt)
                out[f"val_{idx}_joints"] = fig

        # ===================================================================
        # Validation over entire val set — N samples per context, validated
        # with FrankaTrajectoryValidator (shared_context per batch).
        #
        # We sample the full dataloader batch at once (batched contexts) for
        # GPU efficiency, then split the result into per-context groups.
        # ===================================================================
        import torch as _torch

        sample_cfg = self.sample_cfg.model_copy()
        sample_cfg.return_chain = False

        per_ctx_success = []
        per_ctx_success_coll = []
        per_ctx_success_limits = []
        per_ctx_feasible_rate = []
        per_ctx_coll_free_rate = []
        per_ctx_acc_limit_rate = []
        per_ctx_vel_limit_rate = []
        per_ctx_jerk_limit_rate = []
        per_ctx_joint_limit_rate = []

        for sample in val_dataloader:
            sample.to(model.device, non_blocking=True)

            ctx: TrajectoryContext = sample.context
            n_contexts = ctx.batch_size if ctx.is_batched else 1

            # --- Batched sampling: N trajectories per context ---------------
            # sample_cfg.batch_size = total trajectories = n_contexts * N
            sample_cfg.batch_size = n_contexts * self.n_samples_per_context
            ctx_interleaved = ctx.repeat_interleave(
                self.n_samples_per_context, dim=0)
            model.prepare_for_sampling(sample_cfg)

            # trajectory shape: (n_contexts * N, T, D)
            gen = model.sample(ctx_interleaved)

            # --- Split into per-context groups and validate -----------------
            all_trajs = gen.trajectory  # (n_contexts * N, T, D)
            N = self.n_samples_per_context
            traj_chunks = all_trajs.reshape(
                n_contexts, N, *all_trajs.shape[1:])

            for c in range(n_contexts):
                single_ctx = ctx.slice(c) if ctx.is_batched else ctx
                per_ctx_gen = TrajectorySample(
                    trajectory=traj_chunks[c],  # (N, T, D)
                    context=single_ctx,
                    is_normalized=gen.is_normalized,
                    is_batched=True,
                    shared_context=True,
                )

                stats = self.validator.validate(per_ctx_gen, "")

                per_ctx_success.append(stats["success"])
                per_ctx_success_coll.append(
                    1.0 if stats["collision_free_rate"] > 0 else 0.0)
                per_ctx_success_limits.append(
                    1.0 if stats["limits_rate"] > 0 else 0.0)
                per_ctx_feasible_rate.append(stats["feasible_traj_rate"])
                per_ctx_coll_free_rate.append(stats["collision_free_rate"])
                per_ctx_acc_limit_rate.append(stats["acc_limit_rate"])
                per_ctx_vel_limit_rate.append(stats["vel_limit_rate"])
                per_ctx_jerk_limit_rate.append(stats["jerk_limit_rate"])
                per_ctx_joint_limit_rate.append(stats["pos_limit_rate"])

        out["val_success_rate"] = float(np.mean(per_ctx_success))
        out["val_success_coll_rate"] = float(np.mean(per_ctx_success_coll))
        out["val_success_limits_rate"] = float(np.mean(per_ctx_success_limits))
        out["val_feasible_traj_rate"] = float(np.mean(per_ctx_feasible_rate))
        out["val_collision_free_rate"] = float(np.mean(per_ctx_coll_free_rate))
        out["val_acc_limit_rate"] = float(np.mean(per_ctx_acc_limit_rate))
        out["val_vel_limit_rate"] = float(np.mean(per_ctx_vel_limit_rate))
        out["val_jerk_limit_rate"] = float(np.mean(per_ctx_jerk_limit_rate))
        out["val_joint_limit_rate"] = float(np.mean(per_ctx_joint_limit_rate))

        return out

from __future__ import annotations
import torch
import torch.nn as nn
from abc import ABC
from functools import partial

import diffusers as dm
import os
from datetime import datetime, timezone
from typing import Tuple, Optional
from pydantic import BaseModel

from campd.architectures.context.encoder import ContextEncoder
from campd.architectures.diffusion.base import ReverseDiffusionNetwork
from campd.models.helpers import analytical_first_step, apply_hard_conditioning, expand_time
from campd.data.context import TrajectoryContext
from campd.data.embedded_context import EmbeddedContext
from campd.data import TrajectorySample
from campd.utils.registry import import_string, Spec, impl_path
from campd.utils.file import save_params_to_yaml, load_params_from_yaml, to_yamlable
from campd.utils.torch import get_torch_device, freeze_torch_model_params, TensorArgs
from campd.data.normalization import DatasetNormalizer, NormalizationCfg
from campd.architectures.registry import REVERSE_NETS, CONTEXT_NETS


class ContextTrajectoryDiffusionModelCfg(BaseModel):
    """
    Configuration for the diffusion model.

    Args:
        state_dim (int): Dimensionality of the state space.
        network (Spec[ReverseDiffusionNetwork]): Specification for the reverse diffusion network.
        context_network (Spec[ContextEncoder] | None): Specification for the context encoder network. Defaults to None.
        normalizer (NormalizationCfg): Configuration for the dataset normalizer.
        model_type (str, optional): Type of model output. Options: 'epsilon', 'sample', or 'v_prediction'. Defaults to 'epsilon'.
        n_diffusion_steps (int, optional): Number of diffusion steps during training. Defaults to 100.
        ddpm_scheduler_extra_kwargs (dict, optional): Additional arguments passed to the DDPM scheduler configuration. Defaults to {}. 
            Note that `num_train_timesteps` is set to `n_diffusion_steps` and `prediction_type` is set to `model_type`.
    """
    state_dim: int
    model_type: str = 'epsilon'
    ddpm_scheduler_extra_kwargs: dict = {}
    n_diffusion_steps: int = 100
    network: Spec[ReverseDiffusionNetwork]
    context_network: Spec[ContextEncoder] | None = None

    # None for type checking higher up the chain
    normalizer: NormalizationCfg | None = None

    def validate_for_run(self):
        if self.normalizer is None:
            raise ValueError(
                "Normalizer configuration is required for ContextTrajectoryDiffusionModel.")


class SamplingCfg(BaseModel):
    """
    Configuration for sampling from the diffusion model.

    Args:
        n_support_points (int): Number of support points in the generated trajectory.
        batch_size (int, optional): Number of samples to generate per batch. Defaults to 1.
        return_chain (bool, optional): Whether to return the full diffusion chain (all intermediate steps). Defaults to False.
        cond_scale (float, optional): Conditioning scale for classifier-free guidance. Defaults to 1.0.
        rescaled_phi (float, optional): Rescaling factor for noise prediction in CFG. Defaults to 0.0.
        inference_scheduler_cls (str): Inference scheduler to use for sampling.
        inference_scheduler_kwargs (dict, optional): Additional arguments passed to the inference scheduler configuration. Defaults to {}.
        num_inference_steps (int | None, optional): Number of inference steps to use. Mutually exclusive with `timesteps`. Defaults to None.
        timesteps (list[float] | None, optional): Explicit list of timesteps to use for sampling. Mutually exclusive with `num_inference_steps`. Defaults to None.
        analytical_first_step (bool, optional): Whether to use an analytical first step in the sampling process. Defaults to False.
        denormalize (bool, optional): Whether to denormalize the generated trajectory (and context if available). Defaults to True.
        use_cuda_graph (bool, optional): Whether to use CUDA graph for inference. Defaults to False.
    """
    n_support_points: int
    batch_size: int = 1
    return_chain: bool = False
    cond_scale: float = 1.0
    rescaled_phi: float = 0.0
    inference_scheduler_cls: str = 'diffusers.DDPMScheduler'
    inference_scheduler_kwargs: dict = {}
    num_inference_steps: int | None = None
    timesteps: list[float] | None = None
    analytical_first_step: bool = False
    denormalize: bool = True
    use_cuda_graph: bool = False
    context_buffer_sizes: dict[str, int] = {}


class ContextTrajectoryDiffusionModel(nn.Module, ABC):
    def __init__(self, config: ContextTrajectoryDiffusionModelCfg):
        """
        Wrapper around Diffusers library diffusion models for context-conditional trajectory generation.

        Args:
            config (ContextTrajectoryDiffusionModelCfg): Configuration object for diffusion model.
        """
        super().__init__()
        config.validate_for_run()

        self.config: ContextTrajectoryDiffusionModelCfg = config

        self.network = self.config.network.build_from(REVERSE_NETS)
        self.context_network = self.config.context_network.build_from(
            CONTEXT_NETS) if self.config.context_network else None

        self.model_type = config.model_type
        self.state_dim = config.state_dim

        ddpm_scheduler_kwargs = config.ddpm_scheduler_extra_kwargs.copy()
        ddpm_scheduler_kwargs['num_train_timesteps'] = config.n_diffusion_steps
        ddpm_scheduler_kwargs['prediction_type'] = config.model_type

        self.noise_scheduler: dm.DDPMScheduler = dm.DDPMScheduler(
            **ddpm_scheduler_kwargs)

        self.inference_scheduler = None
        self._sample_cfg: SamplingCfg = SamplingCfg(n_support_points=64)
        self.normalizer = DatasetNormalizer.from_config(self.config.normalizer)

    @classmethod
    def from_config(cls, config: ContextTrajectoryDiffusionModelCfg | dict):
        """
        Factory method to create ContextTrajectoryDiffusionModel from config.
        """
        config = ContextTrajectoryDiffusionModelCfg.model_validate(config)
        return cls(config)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Try to determine device from arguments or parameters
        device = None
        # Check first argument if it's device-like
        if args:
            arg = args[0]
            if isinstance(arg, torch.device):
                device = arg
            elif isinstance(arg, (str, int)):
                device = get_torch_device(arg)
            elif isinstance(arg, torch.Tensor):
                device = arg.device

        # Fallback to parameters
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                pass

        if device is not None:
            if self.normalizer is not None:
                self.normalizer.to(device)

        self.device = device

        return self

    def save_config(self, save_dir: str) -> None:
        """
        Saves the model configuration snapshot to the specified directory.
        """
        os.makedirs(save_dir, exist_ok=True)

        model_payload = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "torch_version": str(getattr(torch, "__version__", "")) or None,
            "model_impl": impl_path(self),
            "cfg_impl": impl_path(self.config),
            "config": to_yamlable(self.config),
        }

        config_dir = os.path.join(save_dir, "model_config")
        os.makedirs(config_dir, exist_ok=True)

        save_params_to_yaml(os.path.join(
            config_dir, "model.yaml"), model_payload)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str | torch.device = 'cpu',
        model_iter: Optional[str] = None,
        freeze_params: bool = True,
    ) -> ContextTrajectoryDiffusionModel:
        """
        Loads a trained diffusion model checkpoint and normalizer from a directory.
        """
        if isinstance(device, str):
            device = get_torch_device(device)

        config_path = os.path.join(model_dir, "model_config", 'model.yaml')
        data = load_params_from_yaml(config_path)
        cfg_cls = import_string(data['cfg_impl'])
        cfg = cfg_cls.model_validate(data['config'])

        model_impl = data.get('model_impl')
        model_cls = import_string(model_impl) if model_impl else cls

        # Instantiate model
        model = model_cls.from_config(cfg)
        model.to(device)
        model.tensor_args = TensorArgs(device=device)

        # Load Checkpoint
        checkpoint_path = cls._resolve_checkpoint_path(model_dir, model_iter)
        print(
            f'Loading model {os.path.basename(checkpoint_path)} from {model_dir}')

        state_dict = torch.load(checkpoint_path, map_location=device)

        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            weights = state_dict.get('ema_state_dict') if getattr(cfg,
                                                                  'use_ema', False) else state_dict.get('state_dict')
        else:
            weights = state_dict

        # Strip _orig_mod prefix if present (from torch.compile)
        new_weights = {}
        for k, v in weights.items():
            if k.startswith('_orig_mod.'):
                new_weights[k[10:]] = v
            else:
                new_weights[k] = v
        weights = new_weights

        model.load_state_dict(weights)
        model.eval()
        if freeze_params:
            freeze_torch_model_params(model)

        try:
            model = torch.compile(model)
        except Exception as e:
            print(f'Warning: torch.compile failed: {e}')

        return model

    @staticmethod
    def _resolve_checkpoint_path(model_dir: str, model_iter: Optional[str]) -> str:
        checkpoints_dir = os.path.join(model_dir, 'checkpoints')

        if not os.path.isdir(checkpoints_dir):
            raise FileNotFoundError(
                f"checkpoints dir not found: {checkpoints_dir}")

        files = sorted([f for f in os.listdir(
            checkpoints_dir) if f.endswith('.pth')])
        if not files:
            raise FileNotFoundError(
                f"No .pth checkpoints found in: {checkpoints_dir}")

        # Preferred naming from CheckpointCallback
        if model_iter is not None:
            match = []
            for f in files:
                if model_iter in f[-(len(model_iter)+4):]:
                    match.append(os.path.join(checkpoints_dir, f))

            ema = [f for f in match if 'ema' in f]
            if len(ema) == 1:
                return ema[0]

            non_ema = [f for f in match if 'ema' not in f]
            if len(non_ema) == 1:
                return non_ema[0]

            # raise ValueError(
            #     f"Multiple checkpoints found for iter {model_iter}: {match}")

        for preferred in ("best.pth", "last.pth"):
            p = os.path.join(checkpoints_dir, preferred)
            if os.path.isfile(p):
                return p

        # Otherwise: pick newest by mtime
        newest = max((os.path.join(checkpoints_dir, f)
                      for f in files), key=os.path.getmtime)
        return newest

    def _set_inference_scheduler(self):
        """
        Sets the inference scheduler for sampling.
        """
        self.inference_scheduler_cls = import_string(
            self._sample_cfg.inference_scheduler_cls)
        self.inference_scheduler = self.inference_scheduler_cls.from_config(
            self.noise_scheduler.config, **self._sample_cfg.inference_scheduler_kwargs)

        self._set_inference_timesteps()

    def _set_inference_timesteps(self) -> None:
        """
        (Re)initializes the inference scheduler timesteps.
        Used at the start of sampling to reset internal state.
        """
        assert self.inference_scheduler is not None, "Inference scheduler not set"
        assert self._sample_cfg is not None, "Sample config not set"

        if self._sample_cfg.timesteps or self._sample_cfg.num_inference_steps:
            self.inference_scheduler.set_timesteps(
                timesteps=self._sample_cfg.timesteps, num_inference_steps=self._sample_cfg.num_inference_steps)

    def _set_trajectory_buffer(self):
        """
        Sets the trajectory buffer for CUDA graph.
        """
        if not self._sample_cfg.use_cuda_graph:
            return

        if not self._sample_cfg.return_chain:
            shape = (self._sample_cfg.batch_size,
                     self._sample_cfg.n_support_points,
                     self.state_dim)
        else:
            shape = (self.config.n_diffusion_steps+1,
                     self._sample_cfg.batch_size,
                     self._sample_cfg.n_support_points,
                     self.state_dim + self._sample_cfg.n_support_points)

        self.trajectory_buffer = torch.zeros(
            shape,
            device=self.tensor_args.device)

    def _set_context_buffer(self):
        """
        Sets the context buffers for CUDA graph.
        """
        if not self._sample_cfg.use_cuda_graph or self.context_network is None:
            return

        context_buffer_items = {}
        for key, dim in self.context_network.context_dims.items():
            if key in ['start', 'goal']:
                continue
            try:
                self._sample_cfg.context_buffer_sizes[key]
            except KeyError:
                raise ValueError(
                    f"Context buffer size for key {key} not specified in sample config")
            context_buffer_items[key] = torch.zeros(
                (self._sample_cfg.batch_size,
                 self._sample_cfg.context_buffer_sizes[key], dim),
                device=self.tensor_args.device)

        self.context_buffer = TrajectoryContext(
            data=context_buffer_items,
            start=torch.zeros((self._sample_cfg.batch_size,
                               self.state_dim), device=self.tensor_args.device),
            goal=torch.zeros((self._sample_cfg.batch_size,
                              self.state_dim), device=self.tensor_args.device),
            is_batched=True,
            is_normalized=False
        )

    def _reset_cuda_graph(self):
        if not self._sample_cfg.use_cuda_graph:
            return

        self.graph: torch.cuda.graph | None = None

    def add_noise(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Adds noise to the input data x_start according to the forward diffusion process for timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)
        return self.noise_scheduler.add_noise(x_start, noise, t)

    def prepare_for_sampling(self, config: SamplingCfg):
        """
        Prepares the model for sampling.
        """
        self._sample_cfg = config
        self._set_inference_scheduler()
        self._set_context_buffer()
        self._set_trajectory_buffer()
        self._reset_cuda_graph()

    @torch.no_grad()
    def sample(
        self,
        context: TrajectoryContext = None,
    ) -> TrajectorySample | Tuple[TrajectorySample, torch.Tensor]:
        """
        Generates samples from the diffusion model conditioned on the provided context.
        Args:
           context (TrajectoryContext, optional): Context for conditioning the diffusion model. Defaults to None.
        Returns:
            TrajectorySample: Generated samples.
            Optional[torch.Tensor]: Full diffusion chain if return_chain is True of shape [num_timesteps, batch_size, n_support_points, state_dim].
        """
        if self.inference_scheduler is None:
            raise RuntimeError(
                "Inference scheduler not set. Call prepare_for_sampling(config) before sampling.")

        context_normalized = context.normalize(
            self.normalizer) if context is not None else None
        context_normalized_batched = None
        if context_normalized is not None:
            if context_normalized.is_batched:
                context_normalized.validate_batch_size(
                    self._sample_cfg.batch_size)
                context_normalized_batched = context_normalized
            else:
                context_normalized_batched = context_normalized.expand(
                    self._sample_cfg.batch_size)

        # Reset scheduler internal state for a fresh sampling run.
        self._set_inference_timesteps()

        def _sample(context: TrajectoryContext = None) -> torch.Tensor:
            shape = (self._sample_cfg.batch_size,
                     self._sample_cfg.n_support_points, self.state_dim)
            x = torch.randn(shape, device=self.device)
            hard_conds = context.get_hard_conditions(
            ) if context is not None else {}
            x = apply_hard_conditioning(x, hard_conds)

            chain = [x] if self._sample_cfg.return_chain else None
            # embed context
            embedded_context = self.context_network(
                context) if context is not None else None

            # optionally perform an analytical first step before the diffusion loop
            timesteps = self.inference_scheduler.timesteps
            if hasattr(self.inference_scheduler, '_step_index'):
                self.inference_scheduler._step_index = 0
            if self._sample_cfg.analytical_first_step:
                x = self._analytical_first_step(
                    x, timesteps[0], hard_conds)
                if self._sample_cfg.return_chain:
                    chain.append(x)
                timesteps = timesteps[1:]

            # diffusion sampling loop
            for t in timesteps:
                t_tensor = expand_time(
                    self._sample_cfg.batch_size, t, self.device)
                model_output = self.forward_with_cond_scale(
                    x, t_tensor, embedded_context, self._sample_cfg.cond_scale, self._sample_cfg.rescaled_phi)
                out = self.inference_scheduler.step(
                    model_output, t, x)
                out.prev_sample = apply_hard_conditioning(
                    out.prev_sample, hard_conds)
                x = out.prev_sample
                if self._sample_cfg.return_chain:
                    chain.append(out.prev_sample)

            return x, chain

        if self._sample_cfg.use_cuda_graph:
            # copy context to buffers
            self.context_buffer.copy_from(context_normalized_batched)

            if self.graph is None:
                print("Warming up CUDA graph...")
                for _ in range(5):
                    x, chain = _sample(self.context_buffer)
                    if self._sample_cfg.return_chain:
                        self.trajectory_buffer.copy_(chain)
                    else:
                        self.trajectory_buffer.copy_(x)

                print("Recording CUDA graph...")
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    x, chain = _sample(self.context_buffer)

                    if self._sample_cfg.return_chain:
                        self.trajectory_buffer.copy_(chain)
                    else:
                        self.trajectory_buffer.copy_(x)
                print("Executing...")
            else:
                self.graph.replay()

            x = self.trajectory_buffer[-1] if self._sample_cfg.return_chain else self.trajectory_buffer
            chain = self.trajectory_buffer if self._sample_cfg.return_chain else None

        else:
            x, chain = _sample(context_normalized_batched)
        samples = TrajectorySample(
            trajectory=x, context=context if self._sample_cfg.denormalize else context_normalized,
            is_normalized=True, is_batched=True,
            shared_context=not context.is_batched if context is not None else False)

        if self._sample_cfg.denormalize:
            # context is already unnormalized but will be skipped during denormalization of samples
            samples = samples.denormalize(self.normalizer)

        if self._sample_cfg.return_chain:
            chain = torch.stack(chain, dim=0)
            if self._sample_cfg.denormalize:
                chain = self.normalizer.unnormalize(chain, 'traj')
            return samples, chain
        return samples

    def _analytical_first_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        hard_conds: dict,
    ) -> torch.Tensor:
        """
        Perform an analytical first step before the diffusion loop.
        At the first (noisiest) timestep, the noisy input is a good
        approximation of epsilon, so we skip the network prediction and
        directly compute x_{t-1} using the scheduler formula.
        """
        return analytical_first_step(self.inference_scheduler, x, t, hard_conds)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: TrajectoryContext | None) -> torch.Tensor:
        """
        Forward pass for training.
        """
        embedded_context = None
        if self.context_network is not None and context is not None:
            embedded_context = self.context_network(context)

        return self.network(x, t, embedded_context)

    def forward_with_cond_scale(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        context: EmbeddedContext | None,
        cond_scale: float = 1.0,
        rescaled_phi: float = 0.0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Parallel classifier-free guidance (CFG) by doubling the batch:
          - conditional half uses `context`
          - unconditional half uses `context.null()`
        """
        b = x.shape[0]

        if context is None or cond_scale == 1.0:
            return self.network(x, time, context)

        uncond_context = context.null()
        context_total = context.concat_batch(uncond_context)

        x_total = torch.cat((x, x), dim=0)
        time_total = torch.cat((time, time), dim=0)

        pred_total = self.network(x_total, time_total, context_total)
        pred_cond, pred_uncond = pred_total[:b], pred_total[b:]

        guided = pred_uncond + (pred_cond - pred_uncond) * cond_scale

        if rescaled_phi == 0.0:
            return guided

        std_fn = partial(torch.std, dim=tuple(
            range(1, guided.ndim)), keepdim=True)
        guided_rescaled = guided * (std_fn(pred_cond) / (std_fn(guided) + eps))

        return guided_rescaled * rescaled_phi + guided * (1.0 - rescaled_phi)

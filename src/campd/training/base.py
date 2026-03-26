from __future__ import annotations
from torchjd.aggregation import Aggregator
from campd.training.summary import Summary
from campd.training.callbacks import Callback
from campd.training.objectives.base import TrainingObjective
from campd.data import TrajectorySample
from campd.data.context import TrajectoryContext
from campd.utils.torch import TensorArgs
from campd.utils.registry import Spec
from campd.training.registry import CALLBACKS, SUMMARIES, OBJECTIVES
from campd.models.diffusion import ContextTrajectoryDiffusionModel

import gc
from contextlib import nullcontext
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator

import torch
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchjd.autojac import mtl_backward
from accelerate import Accelerator


class CUDAGraphCfg(BaseModel):
    """Configuration for CUDA graph acceleration during training.

    CUDA graphs capture a fixed sequence of GPU operations and replay them
    without CPU-side overhead.  All tensor shapes must be constant across
    iterations, so buffer sizes must be declared up-front.

    Limitations:
      - Incompatible with TorchJD multi-objective (use_torchjd must be False).
      - Incompatible with Accelerator / DDP.
      - Batch size, n_support_points, and state_dim must remain constant.
    """
    enabled: bool = False
    # Number of warm-up iterations before recording the graph
    warmup_iters: int = 3
    # Trajectory buffer dimensions (must match dataset / dataloader)
    n_support_points: int = 64
    state_dim: int = 7
    batch_size: int = 128
    # Context buffer sizes: mapping from context key to max number of objects
    # e.g. {"spheres": 10, "cuboids": 5}
    context_buffer_sizes: Dict[str, int] = Field(default_factory=dict)
    # How often to read losses from GPU (causes a CUDA sync).
    # Lower values give more frequent logging but reduce throughput.
    log_every_n_steps: int = 10


class TrainerCfg(BaseModel):
    """Configuration for the DiffusionTrainer."""
    results_dir: str

    max_epochs: int = 100

    # Optimization
    optimizer: Spec[torch.optim.Optimizer] = Field(default_factory=lambda: Spec(
        cls="torch.optim.Adam", init={"lr": 1e-4}))
    use_amp: bool = False
    clip_grad: Union[bool, float] = False  # False | True | float(max_norm)
    scheduler: Optional[Spec[Any]] = None
    scheduler_interval: str = "step"  # "epoch" or "step"

    # Validation
    val_check_interval: int = 1  # epochs

    # Registered component specs
    callbacks: List[Spec[Callback]] = Field(default_factory=list)
    summaries: List[Spec[Summary]] = Field(default_factory=list)

    # TorchJD multi-objective optimization
    use_torchjd: bool = True  # Enable TorchJD for multi-loss optimization
    torchjd_aggregator: Spec[Aggregator] = Field(
        default_factory=lambda: Spec(cls="torchjd.aggregation.UPGrad", init={})
    )
    # Device configuration
    tensor_args: TensorArgs

    # Training Objective
    objective: Spec[TrainingObjective]

    # CUDA graph acceleration
    cuda_graph: CUDAGraphCfg = Field(default_factory=CUDAGraphCfg)


class Trainer:
    """
    Trainer for context-aware trajectory diffusion models.

    Handles the training loop, validation, and logging for trajectory diffusion models.
    Supports multi-objective optimization via TorchJD and provides hooks for callbacks
    and summaries.

    Args:
        config: Configuration for the trainer.
        model: The context-aware trajectory diffusion model to train.
    """

    def __init__(self, config: TrainerCfg, model: ContextTrajectoryDiffusionModel, accelerator: Optional[Accelerator] = None):
        self.config = config
        self.tensor_args = config.tensor_args
        self.accelerator = accelerator

        if self.accelerator is not None:
            self.model: ContextTrajectoryDiffusionModel = model.to(
                self.accelerator.device)
        else:
            self.model: ContextTrajectoryDiffusionModel = model.to(
                self.tensor_args.device)
        # set by EMACallback if used
        self.ema_model: Optional[ContextTrajectoryDiffusionModel] = None

        # State
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.last_batch: TrajectorySample = None
        self.current_data: TrajectorySample = None
        self.current_val_data: TrajectorySample = None
        self.stop_training: bool = False

        # Build components
        self.callbacks = [spec.build_from(CALLBACKS)
                          for spec in config.callbacks]
        self.summaries = [spec.build_from(
            SUMMARIES) for spec in config.summaries]

        # Build Objective
        self.objective = config.objective.build_from(OBJECTIVES)

        # Optimizer (Spec is import-string capable; no registry needed)
        self.optimizer = config.optimizer.build(params=self.model.parameters())

        self.scheduler = None
        if config.scheduler is not None:
            self.scheduler = config.scheduler.build(optimizer=self.optimizer)

        # TorchJD aggregator
        self.use_torchjd = config.use_torchjd
        self.aggregator = None
        if self.use_torchjd:
            self.aggregator = config.torchjd_aggregator.build()

        # AMP config
        self._amp_enabled = bool(config.use_amp)
        self._amp_device_type = "cuda" if self.tensor_args.device.type == "cuda" else "cpu"
        self.scaler = torch.amp.GradScaler(
            enabled=self._amp_enabled and self._amp_device_type == "cuda")

        # CUDA graph training acceleration
        self._cuda_graph_mgr: Optional[_TrainCUDAGraphManager] = None
        if config.cuda_graph.enabled:
            if self.use_torchjd:
                raise ValueError(
                    "CUDA graph training is incompatible with TorchJD. "
                    "Disable TorchJD or CUDA graphs.")
            if self.accelerator is not None:
                raise ValueError(
                    "CUDA graph training is incompatible with Accelerator / DDP. "
                    "Disable accelerator or CUDA graphs.")

        self._run_callbacks("on_train_start", trainer=self)

        self.train_dataloader: DataLoader[TrajectorySample] = None
        self.val_dataloader: DataLoader[TrajectorySample] = None

    @property
    def results_dir(self) -> str:
        return self.config.results_dir

    @property
    def _is_main_process(self) -> bool:
        """True on rank-0 (or when not using distributed)."""
        if self.accelerator is None:
            return True
        return self.accelerator.is_main_process

    # -------------------------
    # Public API
    # -------------------------
    def fit(self, train_dataloader: DataLoader[TrajectorySample], val_dataloader: Optional[DataLoader[TrajectorySample]] = None) -> None:
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Prepare model, optimizer, and dataloaders with accelerator if available
        if self.accelerator is not None:
            self.model, self.optimizer, self.train_dataloader = \
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_dataloader)
            if self.val_dataloader is not None:
                self.val_dataloader = self.accelerator.prepare(
                    self.val_dataloader)

        self._run_callbacks("on_fit_start", trainer=self)

        epoch_pbar = tqdm(range(self.config.max_epochs), desc="Epochs",
                          disable=not self._is_main_process)

        use_manual_gc = self.config.cuda_graph.enabled
        if use_manual_gc:
            gc.disable()

        try:
            for _ in epoch_pbar:
                if self.stop_training:
                    break

                self._run_callbacks("on_epoch_start", trainer=self)

                train_metrics = self._train_epoch()

                if (
                    self.val_dataloader is not None
                    and self.config.val_check_interval > 0
                    and (self.current_epoch + 1) % self.config.val_check_interval == 0
                ):
                    self._run_callbacks("on_validation_start", trainer=self)
                    val_metrics = self._validate(self.val_dataloader)
                    self._run_callbacks("on_validation_end",
                                        trainer=self, val_losses=val_metrics)

                self._run_callbacks("on_epoch_end", trainer=self,
                                    train_losses=train_metrics)

                if self.scheduler is not None and self.config.scheduler_interval == "epoch":
                    self.scheduler.step()

                self.current_epoch += 1

                # Run GC in the gap between epochs (no CUDA graph in flight)
                if use_manual_gc:
                    gc.collect()
        finally:
            if use_manual_gc:
                gc.enable()

        self._run_callbacks("on_train_end", trainer=self)

    # -------------------------
    # Train / Val
    # -------------------------
    def _reduce_losses(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Reduce possibly-unreduced losses to per-key scalars.
        """
        reduced: Dict[str, float] = {}
        for k, v in (losses or {}).items():
            if isinstance(v, float):
                reduced[k] = v
                continue
            if not torch.is_tensor(v):
                continue
            v_det = v.detach()
            v_scalar = v_det.mean() if v_det.dim() > 0 else v_det
            reduced[k] = float(v_scalar.item())
        return reduced

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()

        # Lazily initialise CUDA graph manager on first training epoch
        if self.config.cuda_graph.enabled and self._cuda_graph_mgr is None:
            self._cuda_graph_mgr = _TrainCUDAGraphManager(
                cfg=self.config.cuda_graph,
                model=self.model,
                optimizer=self.optimizer,
                objective=self.objective,
                device=self.tensor_args.device,
                clip_grad=self.config.clip_grad,
                amp_enabled=self._amp_enabled,
                amp_device_type=self._amp_device_type,
                scaler=self.scaler,
            )

        use_cuda_graph = self._cuda_graph_mgr is not None

        running = defaultdict(float)
        n = 0
        self._last_losses: Dict[str, float] = {}
        self._last_info: Dict[str, float] = {}

        batch_pbar: Iterator[TrajectorySample] = tqdm(
            self.train_dataloader, desc=f"Epoch {self.current_epoch}", leave=False,
            disable=not self._is_main_process)
        for batch in batch_pbar:
            if self.stop_training:
                break
            batch.to(self.tensor_args.device, non_blocking=True)
            self.current_data = batch
            self.last_batch = batch

            self._run_callbacks("on_batch_start", trainer=self, batch=batch)

            grad_norm_val = None
            if use_cuda_graph:
                if batch.trajectory.shape[0] != self.config.cuda_graph.batch_size:
                    continue
                self._cuda_graph_mgr.step(batch)

                log_interval = self.config.cuda_graph.log_every_n_steps
                should_log = (
                    (self.global_step + 1) % log_interval == 0
                    or self.global_step == 0
                )
                if should_log:
                    losses = self._cuda_graph_mgr.read_losses()
                    info = self._cuda_graph_mgr.read_info()
                else:
                    losses = self._last_losses
                    info = self._last_info

                # Store for steps where we skip logging
                self._last_losses = losses
                self._last_info = info

            else:
                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=self._amp_device_type, enabled=self._amp_enabled):
                    losses, features, info = self._compute_losses(
                        self.model, batch)

                grad_norm_val = self._backward_and_step(losses, features)

            reduced_losses = self._reduce_losses(losses)
            log_dict = dict(reduced_losses, **info)
            log_dict["total"] = sum(reduced_losses.values())
            self._run_callbacks("on_batch_end", trainer=self,
                                batch=batch, loss_dict=log_dict)

            # summaries
            if self._is_main_process:
                for summary in self.summaries:
                    if summary.should_run(self.global_step):
                        summary_results = summary.run(
                            trainer=self, step=self.global_step)
                        self._run_callbacks(
                            "on_summary_end",
                            trainer=self,
                            summary=summary,
                            step=self.global_step,
                            payload=summary_results,
                        )

            self.global_step += 1

            # accumulate epoch metrics
            total_loss = 0.0
            for k, v in losses.items():
                v_val = v.detach() if torch.is_tensor(v) else v
                running[k] += float(v_val)
                total_loss += float(v_val)
            running["total"] += float(total_loss)
            n += 1

            if self.scheduler is not None and self.config.scheduler_interval == "step":
                self.scheduler.step()

            if not use_cuda_graph or should_log:
                postfix_dict = {"loss": float(total_loss)}
                if grad_norm_val is not None:
                    postfix_dict["grad_norm"] = float(grad_norm_val)
                batch_pbar.set_postfix(**postfix_dict)

        if n == 0:
            return {"total": 0.0}
        return {k: v / n for k, v in running.items()}

    def _validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_losses = defaultdict(float)
        n = 0

        model_to_use = self.ema_model if self.ema_model is not None else self.model

        with torch.no_grad():
            batch_pbar: Iterator[TrajectorySample] = tqdm(
                dataloader, desc="Validation", leave=False,
                disable=not self._is_main_process)
            for batch in batch_pbar:
                batch.to(self.tensor_args.device, non_blocking=True)
                self.current_val_data = batch

                with torch.amp.autocast(device_type=self._amp_device_type, enabled=self._amp_enabled):
                    losses, _, _ = self._compute_losses(
                        model_to_use, batch)

                reduced_losses = self._reduce_losses(losses)

                total_loss = 0.0
                for k, v in reduced_losses.items():
                    total_losses[k] += float(v)
                    total_loss += float(v)
                total_losses["total"] += float(total_loss)
                n += 1

        if n == 0:
            avg = {"total": 0.0}
        else:
            avg = {k: v / n for k, v in total_losses.items()}

        return avg

    # -------------------------
    # Helpers
    # -------------------------
    def _compute_losses(
        self, model: ContextTrajectoryDiffusionModel, batch: TrajectorySample
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """
        Returns:
          losses: dict[str, torch.Tensor] of loss tensors (one per loss function).
          Can be reduced or unreduced depending on loss function.
          features: list of captured model output tensors (intermediate features for TorchJD)
        """
        losses, features, info = self.objective.step(model, batch)

        return losses, features, info

    def _backward_and_step(self, losses: Dict[str, torch.Tensor], features: List[torch.Tensor]) -> float:
        """
        Backward + step.

        Supports:
        - Standard single-objective: backprop on summed loss
        - Multi-objective via TorchJD if enabled (and >1 loss)
        - Per-sample losses if provided (dim>0), using your gramian/weighting path
        """
        if not losses:
            raise ValueError("losses dict is empty")

        # Always start clean
        self.optimizer.zero_grad(set_to_none=True)

        loss_list = list(losses.values())
        first_loss = loss_list[0]

        num_losses = len(loss_list)
        has_multiple = num_losses > 1

        # Determine if losses are reduced scalars or per-sample
        reduced = first_loss.numel() == 1
        # (optional) sanity: all losses same "reduced-ness"
        if any((v.numel() == 1) != reduced for v in loss_list):
            raise ValueError(
                "Mixed reduced and per-sample losses in 'losses' dict")

        use_torchjd = bool(getattr(self, "use_torchjd", False)
                           ) and self.aggregator is not None and has_multiple

        # ---- Multi-objective paths ----
        if use_torchjd or not reduced:
            params = [p for p in self.model.parameters() if p.requires_grad]

            # In distributed mode, TorchJD computes gradients with its own
            # backward mechanics that bypass DDP hooks. We use no_sync to
            # prevent premature gradient syncing, then manually all-reduce.
            use_no_sync = (
                self.accelerator is not None
                and self.accelerator.num_processes > 1
            )
            no_sync_ctx = (
                self.accelerator.no_sync(self.model)
                if use_no_sync
                else nullcontext()
            )

            if reduced:
                # reduced multi-loss TorchJD path
                # Use the actual losses returned by the objective
                loss_values = list(losses.values())

                with no_sync_ctx:
                    mtl_backward(
                        losses=loss_values,
                        features=features,
                        aggregator=self.aggregator,
                        retain_graph=True,
                    )
                if use_no_sync:
                    self._sync_gradients(params)
                grad_norm = self._maybe_clip_grad()
                self.optimizer.step()
                return grad_norm

            raise NotImplementedError(
                "Per-sample loss weighting is not implemented yet.")

        # ---- Standard single-objective path ----
        total_loss = torch.sum(torch.stack(
            list(losses.values()), dim=0)) if has_multiple else first_loss

        if self.scaler.is_enabled():
            self.scaler.scale(total_loss).backward()
            grad_norm = self._maybe_clip_grad()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif self.accelerator is not None:
            self.accelerator.backward(total_loss)
            grad_norm = self._maybe_clip_grad()
            self.optimizer.step()
        else:
            total_loss.backward()
            grad_norm = self._maybe_clip_grad()
            self.optimizer.step()

        return grad_norm

    def _grad_global_norm(self):
        sqsum = 0.0
        for p in self.model.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            sqsum += g.pow(2).sum().item()
        return sqsum ** 0.5

    def _maybe_clip_grad(self) -> float:
        grad_norm = self._grad_global_norm()

        clip = self.config.clip_grad
        if clip:
            max_norm = 1.0 if clip is True else float(clip)

            # if scaler enabled, unscale before clipping
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)

            if getattr(self, "accelerator", None) is not None:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), max_norm=max_norm)
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=max_norm)

        return grad_norm

    def _run_callbacks(self, hook_name: str, **kwargs: Any) -> None:
        for cb in self.callbacks:
            hook = getattr(cb, hook_name, None)
            if hook is not None:
                hook(**kwargs)

    def _sync_gradients(self, params: list) -> None:
        """Manually average gradients across distributed processes.

        Used after TorchJD backward calls that bypass DDP's gradient hooks.
        """
        import torch.distributed as dist
        for param in params:
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)


class _TrainCUDAGraphManager:
    """Manages CUDA graph capture and replay for the training step.

    Behavior:
      - First ``warmup_iters`` calls run a normal eager training step.
      - The next call captures the graph and immediately replays it once.
      - Later calls copy new data into static buffers and replay the graph.

    Capture policy:
      - AMP mode: capture forward + scaled backward only.
        Grad unscale / clip / optimizer.step / scaler.update happen outside replay.
      - Non-AMP mode: capture forward + backward only.
        Grad clip / optimizer.step happen outside replay.

    Notes:
      - Scheduler stepping is intentionally NOT handled here. Let the outer
        Trainer own scheduler stepping so behavior matches eager mode.
      - All batch tensor shapes and memory addresses must remain fixed.
    """

    def __init__(
        self,
        cfg: CUDAGraphCfg,
        model: ContextTrajectoryDiffusionModel,
        optimizer: torch.optim.Optimizer,
        objective: TrainingObjective,
        device: torch.device,
        clip_grad: Union[bool, float] = False,
        amp_enabled: bool = False,
        amp_device_type: str = "cuda",
        scaler: Optional[torch.amp.GradScaler] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.objective = objective
        self.device = device
        self.clip_grad = clip_grad
        self.amp_enabled = amp_enabled
        self.amp_device_type = amp_device_type
        self.scaler = scaler

        # Optimizer must be capturable for CUDA graph-safe stepping paths.
        for pg in self.optimizer.param_groups:
            pg["capturable"] = True

        self._warmup_remaining = cfg.warmup_iters
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._captured_losses: Optional[Dict[str, torch.Tensor]] = None
        self._captured_info: Optional[Dict[str, torch.Tensor]] = None

        self._graph_stream = torch.cuda.Stream(device=device)

        # ---- Static buffers ----
        self.traj_buffer = torch.zeros(
            (cfg.batch_size, cfg.n_support_points, cfg.state_dim),
            device=device,
            dtype=torch.float32,
        )
        self.context_buffer: Optional[TrajectoryContext] = None

    # ------------------------------------------------------------------ #
    # Buffer helpers
    # ------------------------------------------------------------------ #
    def _ensure_context_buffer(self, context: TrajectoryContext) -> None:
        """Lazily initialize the context buffer on the first batch."""
        if self.context_buffer is not None:
            return

        buf_items: Dict[str, Any] = {}
        for key in context.keys():
            src = context._items[key]
            feat_dim = src["data"].shape[-1]
            n_obj = self.cfg.context_buffer_sizes.get(key, None)
            if n_obj is None:
                raise ValueError(
                    f"Context key {key} not found in context_buffer_sizes"
                )
            buf_items[key] = torch.zeros(
                (self.cfg.batch_size, n_obj, feat_dim),
                device=self.device,
                dtype=src["data"].dtype,
            )

        self.context_buffer = TrajectoryContext(
            data=buf_items,
            start=torch.zeros(
                (self.cfg.batch_size, self.cfg.state_dim), device=self.device
            ),
            goal=torch.zeros(
                (self.cfg.batch_size, self.cfg.state_dim), device=self.device
            ),
            is_batched=True,
            is_normalized=context.is_normalized,
        )

    # ------------------------------------------------------------------ #
    # Copy real batch data into static buffers
    # ------------------------------------------------------------------ #
    def copy_batch(self, batch: TrajectorySample) -> TrajectorySample:
        """Copy a real batch into static buffers and return a wrapped sample."""
        self.traj_buffer.copy_(batch.trajectory, non_blocking=True)

        ctx_ref = None
        if batch.context is not None:
            self._ensure_context_buffer(batch.context)
            self.context_buffer.copy_from(batch.context)
            ctx_ref = self.context_buffer

        return TrajectorySample(
            trajectory=self.traj_buffer,
            context=ctx_ref,
            metadata=batch.metadata,
            is_normalized=batch.is_normalized,
            is_batched=batch.is_batched,
            shared_context=batch.shared_context,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def step(self, batch: TrajectorySample) -> None:
        """Run one training step, using CUDA graphs when ready.

        This method does not force a device sync.
        """
        buf_batch = self.copy_batch(batch)

        # Ensure graph stream sees the copied batch data before using it.
        self._graph_stream.wait_stream(torch.cuda.current_stream(self.device))

        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            self._eager_step(buf_batch)
            return

        if self._graph is None:
            self._record_and_run(buf_batch)
            return

        self._replay_and_step()

    def read_losses(self) -> Dict[str, float]:
        if self._captured_losses is None:
            return {}
        return {
            k: float(v.detach().mean().item())
            for k, v in self._captured_losses.items()
        }

    def read_info(self) -> Dict[str, float]:
        if self._captured_info is None:
            return {}
        return {
            k: float(v.detach().mean().item())
            for k, v in self._captured_info.items()
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _sum_losses(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss_vals = list(losses.values())
        if not loss_vals:
            raise ValueError("losses dict is empty")
        return (
            torch.sum(torch.stack(loss_vals, dim=0))
            if len(loss_vals) > 1
            else loss_vals[0]
        )

    def _replay_and_step(self) -> None:
        """Replay captured fwd+bwd, then do post-backward optimizer work."""
        self._graph.replay()

        if self.scaler is not None and self.scaler.is_enabled():
            self._maybe_clip_grad(is_scaled=True)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self._maybe_clip_grad(is_scaled=False)
            self.optimizer.step()

    def _eager_step(self, batch: TrajectorySample) -> None:
        """Normal eager training step used during warmup."""
        with torch.cuda.stream(self._graph_stream):
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.amp_device_type,
                enabled=self.amp_enabled,
            ):
                losses, _, info = self.objective.step(self.model, batch)
                total = self._sum_losses(losses)

            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.scale(total).backward()
                self._maybe_clip_grad(is_scaled=True)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total.backward()
                self._maybe_clip_grad(is_scaled=False)
                self.optimizer.step()

            self._captured_losses = losses
            self._captured_info = info

        torch.cuda.current_stream(self.device).wait_stream(self._graph_stream)

    def _record_and_run(self, batch: TrajectorySample) -> None:
        """Capture the training graph, then replay it once for this batch."""
        print("Recording training CUDA graph...")
        torch.cuda.synchronize()

        self._graph = torch.cuda.CUDAGraph()

        with torch.cuda.stream(self._graph_stream):
            # Materialize grad buffers / lazy init without changing params.
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=self.amp_device_type,
                enabled=self.amp_enabled,
            ):
                warm_losses, _, _ = self.objective.step(self.model, batch)
                warm_total = self._sum_losses(warm_losses)

            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.scale(warm_total).backward()
            else:
                warm_total.backward()

            # Clear warmup grads so the captured/replayed step is the only real step.
            self.optimizer.zero_grad(set_to_none=True)

            # Capture forward + backward only.
            with torch.cuda.graph(self._graph, stream=self._graph_stream):
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=self.amp_device_type,
                    enabled=self.amp_enabled,
                ):
                    losses_g, _, info_g = self.objective.step(
                        self.model, batch)
                    total_g = self._sum_losses(losses_g)

                if self.scaler is not None and self.scaler.is_enabled():
                    self.scaler.scale(total_g).backward()
                else:
                    total_g.backward()

            self._captured_losses = losses_g
            self._captured_info = info_g

        torch.cuda.current_stream(self.device).wait_stream(self._graph_stream)

        # Execute one real step for the current batch.
        self._replay_and_step()

        print("Training CUDA graph recorded successfully.")

    def _maybe_clip_grad(self, is_scaled: bool = False) -> None:
        if not self.clip_grad:
            return

        max_norm = 1.0 if self.clip_grad is True else float(self.clip_grad)

        if (
            is_scaled
            and self.scaler is not None
            and self.scaler.is_enabled()
        ):
            self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=max_norm)

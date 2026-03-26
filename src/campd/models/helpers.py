import torch
from typing import Dict


def analytical_first_step(
    scheduler,
    x: torch.Tensor,
    t: torch.Tensor,
    hard_conds: Dict[int | str, torch.Tensor],
) -> torch.Tensor:
    """
    Analytical First Step (AFS) for diffusion sampling.

    At the noisiest timestep the noisy input is a reasonable approximation
    of epsilon, so the network prediction is skipped and the scheduler
    formula is applied directly with ``prediction_type='epsilon'``.

    Args:
        scheduler: A diffusers scheduler instance whose ``.step()`` and
            ``.config.prediction_type`` will be used.  The prediction type
            is temporarily overridden to ``'epsilon'`` and restored after.
        x: Noisy sample tensor.
        t: Current timestep (scalar or per-batch tensor).
        hard_conds: Hard-conditioning dictionary forwarded to
            :func:`apply_hard_conditioning`.

    Returns:
        Denoised sample after one analytical step.
    """
    prev_prediction_type = scheduler.config.prediction_type
    scheduler.config.prediction_type = 'epsilon'
    out = scheduler.step(x, t, x)
    scheduler.config.prediction_type = prev_prediction_type
    return apply_hard_conditioning(out.prev_sample, hard_conds)


def apply_hard_conditioning(x: torch.Tensor, hard_conds: Dict[int | str, torch.Tensor]) -> torch.Tensor:
    """
    Applies hard conditioning to the input tensor x based on the provided hard conditions.
    Args:
        x: Input tensor of shape [batch_size, n_support_points, state_dim].
        hard_conds: Dictionary containing hard conditioning information. Keys can be:
            - "start": tensor of shape [batch_size, state_dim] for the first support point
            - "goal": tensor of shape [batch_size, state_dim] for the last support point
            - Integer time indices: tensor of shape [batch_size, state_dim] for specific support points
    Returns:
        Tensor with hard conditioning applied.
    """
    for key, value in hard_conds.items():
        if value.ndim == 1:
            expand = value.expand(x.shape[0], -1)
        else:
            assert value.shape[0] == x.shape[0], "Hard condition batch size must match input batch size"
            expand = value
        if key == "start":
            x[:, 0, :] = expand
        elif key == "goal":
            x[:, -1, :] = expand
        else:
            x[:, key, :] = expand
    return x


def expand_time(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t

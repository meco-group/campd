from campd.utils.registry import Registry
import torch.nn as nn

MODULES = Registry[nn.Module]("modules")
REVERSE_NETS = Registry[nn.Module]("reverse_diffusion_networks")
CONTEXT_NETS = Registry[nn.Module]("context_networks")

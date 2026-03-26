from pydantic import BaseModel, Field, field_validator, ConfigDict, field_serializer
from time import perf_counter
from collections.abc import Mapping
from typing import Union

import torch
from pydantic import BaseModel, Field, field_validator, ConfigDict


class TimerCUDA(object):
    """ A timer as a context manager

    Wraps around a timer. A custom timer can be passed
    to the constructor. The default timer is timeit.default_timer.

    Note that the latter measures wall clock time, not CPU time!
    On Unix systems, it corresponds to time.time.
    On Windows systems, it corresponds to time.clock.

    Keyword arguments:
        output -- if True, print output after exiting context.
                  if callable, pass output to callable.
        format -- str.format string to be used for output; default "took {} seconds"
        prefix -- string to prepend (plus a space) to output
                  For convenience, if you only specify this, output defaults to True.
    """

    def __init__(self, timer=perf_counter,
                 output=None, fmt="took {:.3f} seconds", prefix="",
                 use_cuda_events=False
                 ):
        self.timer = timer
        self.output = output
        self.fmt = fmt
        self.prefix = prefix
        self.start_time = None
        self.end = None

        self.sync_cuda = True if torch.cuda.is_available() else False
        if use_cuda_events:
            assert self.sync_cuda, "CUDA must be available when using CUDA events"
        self.use_cuda_events = use_cuda_events
        self.factor_cuda_events = 1./1000.  # transform to seconds
        self.start_event = None
        self.end_event = None

    def __call__(self):
        """ Return the current time """
        if self.use_cuda_events:
            self.end_event.record()
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event) * self.factor_cuda_events
        else:
            if self.sync_cuda:
                torch.cuda.synchronize()
            return self.timer()

    def __enter__(self):
        """ Set the start time """
        if self.use_cuda_events:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = self()

        self.start = self()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Set the end time """
        self.end = self()

        if self.prefix and self.output is None:
            self.output = True

        if self.output:
            output = " ".join([self.prefix, self.fmt.format(self.elapsed)])
            if callable(self.output):
                self.output(output)
            else:
                print(output)

    def __str__(self):
        return f'{self.elapsed:.3f}'

    @property
    def elapsed(self):
        """ Return the current elapsed time since start

        If the `elapsed` property is called in the context manager scope,
        the elapsed time between start and property access is returned.
        However, if it is accessed outside of the context manager scope,
        it returns the elapsed time between entering and exiting the scope.

        The `elapsed` property can thus be accessed at different points within
        the context manager scope, to time different parts of the block.

        """
        if self.end is None:
            # if elapsed is called in the context manager scope
            if self.use_cuda_events:
                self()
            else:
                return self() - self.start
        else:
            # if elapsed is called out of the context manager scope
            if self.use_cuda_events:
                return self.end
            else:
                return self.end - self.start


def get_torch_device(device: str = 'cuda') -> torch.device:
    if 'cuda' in device and torch.cuda.is_available():
        if device == 'cuda':
            torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(device)
    elif 'mps' in device:
        device = 'mps'
    else:
        device = 'cpu'
    return torch.device(device)


class TensorArgs(BaseModel, Mapping):
    device: torch.device = Field(
        default_factory=lambda: get_torch_device('cuda'))
    dtype: torch.dtype = torch.float32

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('dtype', mode='before')
    @classmethod
    def validate_dtype(cls, v):
        return str_to_torch_dtype(v)

    @field_validator('device', mode='before')
    @classmethod
    def validate_device(cls, v):
        if isinstance(v, str):
            return get_torch_device(v)
        return v

    @field_serializer('dtype')
    def serialize_dtype(self, dtype: torch.dtype, _info):
        return str(dtype).split('.')[-1]

    @field_serializer('device')
    def serialize_device(self, device: torch.device, _info):
        return str(device)

    _KW_KEYS = ("device", "dtype")

    def keys(self):
        return self._KW_KEYS

    def __getitem__(self, key: str):
        if key in self._KW_KEYS:
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self):
        return iter(self._KW_KEYS)

    def __len__(self):
        return len(self._KW_KEYS)


def str_to_torch_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Convert a string to a torch dtype."""
    if not isinstance(dtype, str):
        return dtype

    dtype_map = {
        'float': torch.float,
        'float32': torch.float32,
        'float64': torch.float64,
        'double': torch.double,
        'half': torch.half,
        'float16': torch.float16,
        'int': torch.int,
        'int32': torch.int32,
        'int64': torch.int64,
        'long': torch.long,
        'short': torch.short,
        'bool': torch.bool,
    }
    if dtype in dtype_map:
        return dtype_map[dtype]

    try:
        return getattr(torch, dtype)
    except AttributeError:
        # If it fails, maybe it's already a valid dtype string for some other parser,
        # or we just let it fail later
        raise ValueError(f"Unknown dtype: {dtype}")


def to_torch(x, device='cpu', dtype=torch.float, requires_grad=False, clone=False):
    if x is None:
        return None

    # Handle string dtype
    dtype = str_to_torch_dtype(dtype)

    if torch.is_tensor(x):
        if clone:
            x = x.clone()
        return x.to(device=device, dtype=dtype)
    elif isinstance(x, dict):
        return {k: to_torch(v, device=device, dtype=dtype, requires_grad=requires_grad) for k, v in x.items()}
    try:
        return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
    except:
        return x


def freeze_torch_model_params(model):
    for param in model.parameters():
        param.requires_grad = False
    # If the model is frozen we do not save it again, since the parameters did not change
    model.is_frozen = True


import pytest
import torch
from campd.utils.torch import TensorArgs


def test_tensor_args_defaults():
    args = TensorArgs()
    assert args.dtype == torch.float32
    assert isinstance(args.device, torch.device)


def test_tensor_args_torch_init():
    args = TensorArgs(device=torch.device('cpu'), dtype=torch.float64)
    assert args.device.type == 'cpu'
    assert args.dtype == torch.float64


def test_tensor_args_string_init():
    args = TensorArgs(device='cpu', dtype='float32')
    assert args.device.type == 'cpu'
    assert args.dtype == torch.float32

    args2 = TensorArgs(dtype='double')
    assert args2.dtype == torch.double

    args3 = TensorArgs(dtype='long')
    assert args3.dtype == torch.long


def test_tensor_args_invalid_string():
    with pytest.raises(ValueError, match="Unknown dtype: invalid_dtype"):
        TensorArgs(dtype='invalid_dtype')


def test_str_to_torch_dtype_helper():
    # Implicitly tested via class, but good to know
    pass


def test_unpacking_behavior():
    args = TensorArgs(device='cpu', dtype='float32')
    # Simulate usage
    t = torch.zeros(5, **args)
    assert t.dtype == torch.float32
    assert t.device.type == 'cpu'

from distutils.version import LooseVersion
import importlib
from importlib.util import find_spec
import operator
import platform
import sys
import torch
from pkg_resources import DistributionNotFound


def _module_available(module_path: str) -> bool:
    """
    Check if a path is available in your environment
    >>> _module_available('os')
    True
    >>> _module_available('some.module')
    False
    """
    try:
        return find_spec(module_path) is not None
    except AttributeError:
        # Python 3.6
        return False
    except ModuleNotFoundError:
        # Python 3.7+
        return False


def _compare_version(package: str, op, version) -> bool:
    """
    Compare package version with some requirements
    >>> _compare_version("torch", operator.ge, "0.1")
    True
    """
    try:
        pkg = importlib.import_module(package)
    except (ModuleNotFoundError, DistributionNotFound):
        return False
    try:
        pkg_version = LooseVersion(pkg.__version__)
    except AttributeError:
        return False
    if not (hasattr(pkg_version, "vstring") and hasattr(pkg_version, "version")):
        # this is mock by sphinx, so it shall return True ro generate all summaries
        return True
    return op(pkg_version, LooseVersion(version))


def _force_version(package: str, op, version, message: str):
    if not _compare_version(package, op, version):
        raise ImportError(message)


_IS_WINDOWS = platform.system() == "Windows"
_IS_INTERACTIVE = hasattr(sys, "ps1")  # https://stackoverflow.com/a/64523765
_TORCH_LOWER_EQUAL_1_4 = _compare_version("torch", operator.le, "1.5.0")
_TORCH_GREATER_EQUAL_1_6 = _compare_version("torch", operator.ge, "1.6.0")
_TORCH_GREATER_EQUAL_1_7 = _compare_version("torch", operator.ge, "1.7.0")
_TORCH_GREATER_EQUAL_1_8 = _compare_version("torch", operator.ge, "1.8.0")

_APEX_AVAILABLE = _module_available("apex")
_HIT_AVAILABLE = _module_available("hit")
_HOROVOD_AVAILABLE = _module_available("horovod.torch")
_KINETO_AVAILABLE = torch.profiler.kineto_available() if _TORCH_GREATER_EQUAL_1_8 else False
_NATIVE_AMP_AVAILABLE = _module_available("torch.cuda.amp") and hasattr(torch.cuda.amp, "autocast")
_RPC_AVAILABLE = not _IS_WINDOWS and _module_available("torch.distributed.rpc")
_TORCH_QUANTIZE_AVAILABLE = bool([eg for eg in torch.backends.quantized.supported_engines if eg != "none"])
_XLA_AVAILABLE = _module_available("torch_xla")
_FMOE_AVAILABLE = _module_available("fmoe")

# Bump to tensorflow 2.0 for dataset transformation to support python 3.8+
_TENSORFLOW_AVAILABLE = _compare_version("tensorflow", operator.ge, "2.0.0")

_force_version(
    "hydra", operator.ge, "1.1.0", "Hydra version mismatch, please update your version according to requirements.txt"
)

# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional platform dependency shims; no Megatron model methods are replaced."""


def pure_torch_hadamard_transform(x, scale=1.0):
    """Fast Walsh-Hadamard transform along the last dim (size must be 2**k).

    Vectorized butterfly accumulated in fp32, numerically equivalent to
    ``F.linear(x, scipy.linalg.hadamard(dim)) * scale`` and bit-for-bit identical to
    the Dao-AILab CUDA kernel it stands in for.
    """
    import torch

    n = x.shape[-1]
    if n < 1 or n & (n - 1) != 0:
        raise ValueError(f"hadamard_transform requires last dim to be a power of 2, got {n}")
    orig_dtype = x.dtype
    orig_shape = x.shape
    # Accumulate in fp32 for numerical stability, cast back at the end.
    y = x.to(torch.float32).reshape(-1, n)
    h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        h *= 2
    y = y * scale
    return y.reshape(orig_shape).to(orig_dtype)


def apply_fast_hadamard_transform_shim():
    """Provide a pure-torch ``fast_hadamard_transform`` when the CUDA package is absent.

    DeepSeek-V4 / V3.2 sparse-attention (DSA) in Megatron-LM does
    ``from fast_hadamard_transform import hadamard_transform`` at import time and
    asserts it is not None inside the indexer's ``rotate_activation``. The upstream
    Dao-AILab package only ships nvcc kernels and cannot be built on ROCm, so that
    import yields ``None`` there and every DSA forward crashes.

    Register ``pure_torch_hadamard_transform`` under the real import name so any
    importer picks it up, and back-fill modules that already captured ``None``
    (e.g. Megatron's ``dsa`` module at import time). This is a no-op once the name
    imports, be it the real package or the stub an earlier call installed.
    """
    import importlib
    import logging
    import sys
    import types

    # Catch every exception, not just ImportError: a CUDA extension built against
    # another toolkit fails to load with OSError. An import that goes through means
    # every importer of the name resolves to that same module, so none of them can
    # be left holding a None binding.
    try:
        importlib.import_module("fast_hadamard_transform")
        return
    except Exception:
        pass

    module = types.ModuleType("fast_hadamard_transform")
    module.hadamard_transform = pure_torch_hadamard_transform
    sys.modules["fast_hadamard_transform"] = module

    # Back-fill modules that already ran `from fast_hadamard_transform import
    # hadamard_transform` and captured None (e.g. Megatron's dsa.py at import).
    # Read `__dict__` directly: `getattr` would trigger PEP-562 module-level
    # `__getattr__` hooks, which can raise or force lazy imports.
    for mod in list(sys.modules.values()):
        mod_dict = getattr(mod, "__dict__", None)
        if mod_dict is not None and mod_dict.get("hadamard_transform", "keep") is None:
            mod_dict["hadamard_transform"] = pure_torch_hadamard_transform

    logging.getLogger(__name__).warning(
        "fast_hadamard_transform is unavailable; falling back to a pure-torch Fast "
        "Walsh-Hadamard transform. Results match the CUDA kernel but DSA forward will be slower."
    )


class _MissingFlashAttnCute:
    """Meta-path finder that reports ``flash_attn.cute`` as an absent module."""

    _PREFIX = "flash_attn.cute"

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self._PREFIX or fullname.startswith(self._PREFIX + "."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


def neutralize_broken_flash_attn_cute():
    """Hide ``flash_attn.cute`` when importing it raises anything but ImportError.

    Megatron-LM probes FA4 with ``from flash_attn.cute import flash_attn_varlen_func``
    guarded by ``except ImportError``. flash-attn 2.8.3 ships that subpackage but
    declares no ``nvidia-cutlass-dsl`` dependency, and the code targets the 4.5.x
    API, so against the 4.6.x sglang pins it raises ``AttributeError`` on
    ``cute.core.ThrMma``. That escapes the guard and aborts
    ``import megatron.core.transformer.attention`` — i.e. every mcore entry point.

    Probe once and only install the finder when the subpackage is actually broken,
    so a build whose FA4 does import keeps it. Must run before anything imports
    megatron's GPT layer specs.
    """
    import importlib
    import importlib.util
    import logging
    import sys

    if any(isinstance(finder, _MissingFlashAttnCute) for finder in sys.meta_path):
        return
    try:
        if importlib.util.find_spec("flash_attn.cute") is None:
            return
    except Exception:
        return  # flash-attn absent or unimportable; the probe already sees ImportError

    try:
        importlib.import_module("flash_attn.cute")
    except ImportError:
        return  # already the exception every probe expects
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
    else:
        return  # FA4 imports here; leave it usable

    # A failed import can leave half-initialized submodules behind; drop them so
    # the finder below is what every later import reaches.
    for name in [n for n in sys.modules if n == "flash_attn.cute" or n.startswith("flash_attn.cute.")]:
        del sys.modules[name]
    sys.meta_path.insert(0, _MissingFlashAttnCute())

    logging.getLogger(__name__).warning(
        "flash_attn.cute (FlashAttention-4) is unusable and is being hidden so optional "
        "FA4 probes fail with ImportError as they expect; FlashAttention-2 is unaffected. "
        "Cause: %s",
        reason,
    )

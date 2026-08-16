# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import os
import warnings

from .base import BaseEngine, EngineRegistry

__all__ = ["BaseEngine", "EngineRegistry"]

if os.environ.get("VERL_ENGINE_LAZY_IMPORTS") != "1":
    from .fsdp import FSDPEngine, FSDPEngineWithLMHead

    __all__ += ["FSDPEngine", "FSDPEngineWithLMHead"]

    try:
        from .torchtitan import TorchTitanEngine, TorchTitanEngineWithLMHead

        __all__ += ["TorchTitanEngine", "TorchTitanEngineWithLMHead"]
    except ImportError as e:
        warnings.warn(f"torchtitan engine is not available: {e!r}", stacklevel=1)

    try:
        from .veomni import VeOmniEngine, VeOmniEngineWithLMHead

        __all__ += ["VeOmniEngine", "VeOmniEngineWithLMHead"]
    except ImportError as e:
        warnings.warn(f"veomni engine is not available: {e!r}", stacklevel=1)

    try:
        from .automodel import AutomodelEngine, AutomodelEngineWithLMHead

        __all__ += ["AutomodelEngine", "AutomodelEngineWithLMHead"]
    except ImportError as e:
        warnings.warn(f"automodel engine is not available: {e!r}", stacklevel=1)

    # Mindspeed must be imported before Megatron so its patches take effect.
    try:
        from .mindspeed import (
            MindspeedEngineWithLMHead,
            MindspeedEngineWithValueHead,
            MindSpeedMegatronEngineWithLMHead,
        )

        __all__ += [
            "MindspeedEngineWithLMHead",
            "MindspeedEngineWithValueHead",
            "MindSpeedMegatronEngineWithLMHead",
        ]
    except ImportError as e:
        warnings.warn(f"mindspeed engine is not available: {e!r}", stacklevel=1)

    try:
        from .megatron import (
            MegatronEngine,
            MegatronEngineWithLMHead,
            MegatronEngineWithValueHead,
        )

        __all__ += [
            "MegatronEngine",
            "MegatronEngineWithLMHead",
            "MegatronEngineWithValueHead",
        ]
    except ImportError as e:
        warnings.warn(f"megatron engine is not available: {e!r}", stacklevel=1)

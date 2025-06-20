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

try:
    from mbridge import AutoBridge
    from mbridge.utils.post_creation_callbacks import freeze_moe_router, make_value_model
except ImportError:
    import subprocess
    import sys

    print("mbridge package not found. This package is required for model bridging functionality.")
    print("Install mbridge with `pip install mbridge`")

    def install_mbridge():
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "mbridge"])
        except subprocess.CalledProcessError:
            print("Failed to install mbridge")
            raise

    install_mbridge()
    from mbridge import *

__all__ = ["AutoBridge", "make_value_model", "freeze_moe_router"]

# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from typing import Optional

from .util import ContextParallelLayout


def build_thd_preprocess_options(
    config,
    *,
    cp_layout: ContextParallelLayout,
    local_cp_size: Optional[int] = None,
) -> dict:
    """Build the shared THD row-layout contract for model forward and router replay."""
    min_local_rows = (
        config.csa_window_size if getattr(config, "experimental_attention_variant", None) == "dsv4_hybrid" else None
    )
    return {
        "use_fp8_padding": config.fp8 in ["e4m3", "hybrid"],
        "local_cp_size": local_cp_size,
        "min_local_rows": min_local_rows,
        "cp_layout": cp_layout,
    }

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""
Shared FSDP context module.
This module provides a unified global registry for fsdp_context,
used by both fsdp.fully_shard_fusion and auto_parallel.fully_shard_fusion.
"""

# Global registry for fsdp_context
_g_fsdp_context = None


def register_fsdp_context(context):
    global _g_fsdp_context
    _g_fsdp_context = context


def get_fsdp_context():
    return _g_fsdp_context

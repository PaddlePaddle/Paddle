#!/usr/bin/env bash
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

target_lists_for_semi_auto_ci=(
    "python/paddle/distributed/auto_parallel"
    "python/paddle/distributed/checkpoint"
    "paddle/fluid/distributed/auto_parallel"
    "paddle/fluid/framework/new_executor"
    "paddle/fluid/pybind/auto_parallel_py.cc"
    "paddle/fluid/pybind/auto_parallel_py.h"
    "paddle/phi/infermeta/spmd_rules"
    "paddle/phi/core/distributed"
    "paddle/phi/api/generator/dist_api_gen.py"
    "paddle/phi/api/generator/dist_bw_api_gen.py"
    "tools/auto_parallel/target_path_lists.sh"
    "test/auto_parallel"
    "paddle/fluid/ir_adaptor/"
    "paddle/fluid/pir/dialect"
    "paddle/fluid/pir/transforms"
    "paddle/fluid/pir/serialize_deserialize"
    "test/auto_parallel/hybrid_strategy/semi_auto_llama_save_load.py"
    "python/paddle/base/executor.py"
)

target_lists_for_dygraph_ci=(
    "python/paddle/distributed/fleet"
    "python/paddle/distributed/communication"
    "python/paddle/distributed/sharding"
    "paddle/fluid/distributed/collective"
    "paddle/phi/core/distributed"
    "tools/auto_parallel/target_path_lists.sh"
    "test/collective/hybrid_strategy"
)

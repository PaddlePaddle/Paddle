#!/bin/bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

env_home=/workspace/env3.7/lib/python3.7/site-packages/paddle
src_home=/workspace/paddle-fork

# cp ${src_home}/python/paddle/fluid/dygraph/amp/auto_cast.py   ${env_home}/fluid/dygraph/amp/auto_cast.py
# cp ${src_home}/python/paddle/fluid/dygraph/dygraph_to_static/partial_program.py   ${env_home}/fluid/dygraph/dygraph_to_static/partial_program.py

# cp ${src_home}/python/paddle/fluid/layers/control_flow.py   ${env_home}/fluid/layers/control_flow.py
# cp ${src_home}/python/paddle/fluid/layers/math_op_patch.py   ${env_home}/fluid/layers/math_op_patch.py
# cp ${src_home}/python/paddle/fluid/dygraph/dygraph_to_static/* ${env_home}/fluid/dygraph/dygraph_to_static/
# cp ${src_home}/python/paddle/tensor/array.py ${env_home}/tensor/array.py
# cp ${src_home}/python/paddle/jit/dy2static/* ${env_home}/jit/dy2static/

# cp ${src_home}/python/paddle/fluid/executor.py ${env_home}/fluid/
# cp ${src_home}/python/paddle/fluid/framework.py ${env_home}/fluid/

cp ${src_home}/python/paddle/fluid/layers/control_flow.py ${env_home}/fluid/layers/
cp ${src_home}/python/paddle/fluid/backward.py ${env_home}/fluid/

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

export FLAGS_call_stack_level=2 
export GLOG_v=4 
export FLAGS_enable_pir_api=1
export ENABLE_FALL_BACK=False

export FLAGS_cinn_map_expr_enable_index_detail=1

export FLAGS_cinn_enable_map_expr=True
python test_dynamic_shape.py > ../../../output.log 2>&1

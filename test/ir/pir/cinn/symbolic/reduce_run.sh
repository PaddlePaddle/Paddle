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

export GLOG_v=4
FLAGS_pir_apply_shape_optimization_pass=1 FLAGS_cinn_bucket_compile=True FLAG_logbuflevel=-1 FLAGS_print_ir=1 FLAGS_call_stack_level=2 FLAGS_enable_pir_api=1 python test_cinn_reduce_symbolic_demo.py > output.log 2>&1

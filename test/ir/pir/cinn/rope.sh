# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# static-frontend && dynamic-backend
export GLOG_vmodule=cuda_module=10
export FLAGS_cinn_convert_static_dim_to_dynamic=2048:S0
export FLAGS_enable_pir_api=1
export FLAGS_cinn_bucket_compile=True
python test_rope.py 2>&1 | tee tmp.txt

# static-frontend && static-backend
#export CUDA_VISIBLE_DEVICES=5
#export GLOG_vmodule=instruction=10
#export FLAGS_enable_pir_api=1
#export FLAGS_cinn_bucket_compile=True
#python test_rope.py 2>&1 | tee tmp.txt

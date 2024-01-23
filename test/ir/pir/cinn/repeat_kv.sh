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
# GLOG_v=4 FLAGS_cinn_convert_static_dim_to_dynamic=2048:S0 FLAGS_enable_pir_api=1 FLAGS_cinn_bucket_compile=True python test_repeat_kv.py 2>&1 | tee /tmp/a

# static-frontend && static-backend
GLOG_v=4 FLAGS_enable_pir_api=1 FLAGS_cinn_bucket_compile=True python test_repeat_kv.py 2>&1 | tee /tmp/a

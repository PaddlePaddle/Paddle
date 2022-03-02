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

def miminize_cg(f,
              x0,
              max_iter=50,
              tolerance_grad=1e-8,
              tolerance_change=0,
              norm_type=np.inf,
              line_search_method='strong_wolfe',
              max_line_search_iters=50,
              initial_step_length=1.0,
              dtype='float32',
              name=None):

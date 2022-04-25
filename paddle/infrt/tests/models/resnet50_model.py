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

import paddle
from paddle.vision.models import resnet50
from paddle.jit import to_static
from paddle.static import InputSpec
import sys

model = resnet50(True)
net = to_static(
    model, input_spec=[InputSpec(
        shape=[None, 3, 256, 256], name='x')])
paddle.jit.save(net, sys.argv[1])

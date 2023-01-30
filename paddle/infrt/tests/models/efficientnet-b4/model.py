# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
<<<<<<< HEAD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
=======
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

<<<<<<< HEAD
import sys

# url: https://aistudio.baidu.com/aistudio/projectdetail/3756986?forkThirdPart=1
from net import EfficientNet

import paddle
from paddle.jit import to_static
from paddle.static import InputSpec

model = EfficientNet.from_name('efficientnet-b4')
net = to_static(
    model, input_spec=[InputSpec(shape=[None, 3, 256, 256], name='x')]
)
=======
# url: https://aistudio.baidu.com/aistudio/projectdetail/3756986?forkThirdPart=1
from net import EfficientNet
from paddle.jit import to_static
from paddle.static import InputSpec
import paddle
import sys

model = EfficientNet.from_name('efficientnet-b4')
net = to_static(
    model, input_spec=[InputSpec(
        shape=[None, 3, 256, 256], name='x')])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
paddle.jit.save(net, sys.argv[1])

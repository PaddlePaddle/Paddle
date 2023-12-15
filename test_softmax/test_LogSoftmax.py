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

import paddle
import paddle.nn.functional as F

x = [
    [[-2.0, 3.0, -4.0, 5.0], [3.0, -4.0, 5.0, -6.0], [-7.0, -8.0, 8.0, 9.0]],
    [[1.0, -2.0, -3.0, 4.0], [-5.0, 6.0, 7.0, -8.0], [6.0, 7.0, 8.0, 9.0]],
]
x = paddle.to_tensor(x)
out1 = F.log_softmax(x)
print(out1)

out2 = F.softmax(x)
print(out2)

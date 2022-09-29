# copyright (c) 2021 paddlepaddle authors. all rights reserved.
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

import unittest

import numpy as np

import paddle
from paddle import fluid

from paddle import Model
from paddle.static import InputSpec
from paddle.nn.layer.loss import CrossEntropyLoss
from paddle.vision.models import LeNet


@unittest.skipIf(not fluid.is_compiled_with_cuda(),
                 'CPU testing is not supported')
class TestDistTraningWithPureFP16(unittest.TestCase):

    def test_amp_training_purefp16(self):
        if not fluid.is_compiled_with_cuda():
            self.skipTest('module not tested when ONLY_CPU compling')
        data = np.random.random(size=(4, 1, 28, 28)).astype(np.float32)
        label = np.random.randint(0, 10, size=(4, 1)).astype(np.int64)

        paddle.enable_static()
        paddle.set_device('gpu')
        net = LeNet()
        amp_level = "O2"
        inputs = InputSpec([None, 1, 28, 28], "float32", 'x')
        labels = InputSpec([None, 1], "int64", "y")
        model = Model(net, inputs, labels)
        optim = paddle.optimizer.Adam(learning_rate=0.001,
                                      parameters=model.parameters(),
                                      multi_precision=True)
        amp_configs = {"level": amp_level, "use_fp16_guard": False}
        model.prepare(optimizer=optim,
                      loss=CrossEntropyLoss(reduction="sum"),
                      amp_configs=amp_configs)
        model.train_batch([data], [label])


if __name__ == '__main__':
    unittest.main()

# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import unittest

import numpy as np

import paddle
from paddle import nn


class LSTMLayer(nn.Layer):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.cell = nn.LSTM(
            in_channels, hidden_size, direction='bidirectional', num_layers=2
        )

    def forward(self, x):
        x, _ = self.cell(x)
        return x


class LinearNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 12)
        self.dropout = nn.Dropout(0.5)

    @paddle.jit.to_static
    def forward(self, x):
        y = self.fc(x)
        y = self.dropout(y)
        return y


class TestSaveInEvalMode(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_in_eval(self):
        paddle.jit.ProgramTranslator().enable(True)
        net = LinearNet()
        x = paddle.randn((2, 10))
        x.stop_gradient = False
        dygraph_out = net(x)
        loss = paddle.mean(dygraph_out)
        sgd = paddle.optimizer.SGD(
            learning_rate=0.001, parameters=net.parameters()
        )
        loss.backward()
        sgd.step()
        # switch eval mode firstly
        net.eval()
        # save directly
        net = paddle.jit.to_static(
            net, input_spec=[paddle.static.InputSpec(shape=[-1, 10])]
        )

        model_path = os.path.join(self.temp_dir.name, 'linear_net')
        paddle.jit.save(net, model_path)
        # load saved model
        load_net = paddle.jit.load(model_path)

        x = paddle.randn((2, 10))
        eval_out = net(x)

        infer_out = load_net(x)
        np.testing.assert_allclose(
            eval_out.numpy(),
            infer_out.numpy(),
            rtol=1e-05,
            err_msg='eval_out is {}\n infer_out is \n{}'.format(
                eval_out, infer_out
            ),
        )


if __name__ == "__main__":
    unittest.main()

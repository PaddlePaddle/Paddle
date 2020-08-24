# copyright (c) 2020 paddlepaddle authors. all rights reserved.
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

from __future__ import division
from __future__ import print_function

import unittest

import numpy as np
import shutil
import tempfile

from paddle import fluid
from paddle.nn import Conv2d, Pool2D, Linear, ReLU, Sequential

from paddle.incubate.hapi.utils import uncombined_weight_to_state_dict


class LeNetDygraph(fluid.dygraph.Layer):
    def __init__(self, num_classes=10, classifier_activation='softmax'):
        super(LeNetDygraph, self).__init__()
        self.num_classes = num_classes
        self.features = Sequential(
            Conv2d(
                1, 6, 3, stride=1, padding=1),
            ReLU(),
            Pool2D(2, 'max', 2),
            Conv2d(
                6, 16, 5, stride=1, padding=0),
            ReLU(),
            Pool2D(2, 'max', 2))

        if num_classes > 0:
            self.fc = Sequential(
                Linear(400, 120),
                Linear(120, 84),
                Linear(
                    84, 10, act=classifier_activation))

    def forward(self, inputs):
        x = self.features(inputs)

        if self.num_classes > 0:
            x = fluid.layers.flatten(x, 1)
            x = self.fc(x)
        return x


class TestUncombinedWeight2StateDict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.save_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.save_dir)

    def test_infer(self):
        start_prog = fluid.Program()
        train_prog = fluid.Program()

        x = fluid.data(name='x', shape=[None, 1, 28, 28], dtype='float32')

        with fluid.program_guard(train_prog, start_prog):
            with fluid.unique_name.guard():
                x = fluid.data(
                    name='x', shape=[None, 1, 28, 28], dtype='float32')
                model = LeNetDygraph()
                output = model.forward(x)

        excutor = fluid.Executor()
        excutor.run(start_prog)

        test_prog = train_prog.clone(for_test=True)

        fluid.io.save_params(excutor, self.save_dir, test_prog)

        rand_x = np.random.rand(1, 1, 28, 28).astype('float32')
        out = excutor.run(program=test_prog,
                          feed={'x': rand_x},
                          fetch_list=[output.name],
                          return_numpy=True)

        state_dict = uncombined_weight_to_state_dict(self.save_dir)

        key2key_dict = {
            'features.0.weight': 'conv2d_0.w_0',
            'features.0.bias': 'conv2d_0.b_0',
            'features.3.weight': 'conv2d_1.w_0',
            'features.3.bias': 'conv2d_1.b_0',
            'fc.0.weight': 'linear_0.w_0',
            'fc.0.bias': 'linear_0.b_0',
            'fc.1.weight': 'linear_1.w_0',
            'fc.1.bias': 'linear_1.b_0',
            'fc.2.weight': 'linear_2.w_0',
            'fc.2.bias': 'linear_2.b_0'
        }

        fluid.enable_imperative()
        dygraph_model = LeNetDygraph()

        converted_state_dict = dygraph_model.state_dict()
        for k1, k2 in key2key_dict.items():
            converted_state_dict[k1] = state_dict[k2]

        dygraph_model.set_dict(converted_state_dict)

        dygraph_model.eval()
        dy_out = dygraph_model(fluid.dygraph.to_variable(rand_x))

        np.testing.assert_allclose(dy_out.numpy(), out[0], atol=1e-5)


if __name__ == '__main__':
    unittest.main()

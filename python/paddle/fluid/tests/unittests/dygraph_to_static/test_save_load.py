#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import os
import tempfile
import unittest

import numpy as np
from test_fetch_feed import Linear

import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer

np.random.seed(2020)

place = (
    fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
)


class TestDyToStaticSaveLoad(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(
            self.temp_dir.name, "test_dy2stat_save_load"
        )
=======
from __future__ import print_function

import unittest
import os
import tempfile

import numpy as np
import paddle.fluid as fluid

from paddle.fluid.dygraph.dygraph_to_static import ProgramTranslator
from paddle.fluid.optimizer import AdamOptimizer
from test_fetch_feed import Linear

np.random.seed(2020)

place = fluid.CUDAPlace(
    0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()


class TestDyToStaticSaveLoad(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name,
                                       "test_dy2stat_save_load")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_load_same_result(self):
<<<<<<< HEAD
=======
        program_translator = ProgramTranslator()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        x_data = np.random.randn(30, 10, 32).astype('float32')
        batch_num = 3

        with fluid.dygraph.guard(place):

<<<<<<< HEAD
            paddle.jit.enable_to_static(True)
            x = fluid.dygraph.to_variable(x_data)
            net = Linear(32, 64)
            adam = AdamOptimizer(
                learning_rate=0.1, parameter_list=net.parameters()
            )
=======
            program_translator.enable(True)
            x = fluid.dygraph.to_variable(x_data)
            net = Linear(32, 64)
            adam = AdamOptimizer(learning_rate=0.1,
                                 parameter_list=net.parameters())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            for i in range(batch_num):
                static_out, static_loss = net(x)
                # Update parameters
                static_loss.backward()
                adam.minimize(static_loss)
                net.clear_gradients()
            # Save parameters

<<<<<<< HEAD
            paddle.save(net.state_dict(), self.model_path + '.pdparams')
=======
            fluid.save_dygraph(net.state_dict(), self.model_path)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            # minimize() will update parameter, call net() to get output and avg_loss.
            # Switch into eval mode.
            net.eval()
            static_out, static_loss = net(x)

        # load parameters into dygraph
        with fluid.dygraph.guard(place):
            dygraph_net = Linear(32, 64)

            # Load parameters
<<<<<<< HEAD
            model_dict = paddle.load(self.model_path + '.pdparams')
=======
            model_dict, _ = fluid.load_dygraph(self.model_path)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            dygraph_net.set_dict(model_dict)
            # Switch into eval mode.
            dygraph_net.eval()

            x = fluid.dygraph.to_variable(x_data)
            # predict output
<<<<<<< HEAD
            paddle.jit.enable_to_static(False)
            dygraph_out, dygraph_loss = dygraph_net(x)

        np.testing.assert_allclose(
            dygraph_out.numpy(), static_out.numpy(), rtol=1e-05
        )
        np.testing.assert_allclose(
            dygraph_loss.numpy(), static_loss.numpy(), rtol=1e-05
        )
=======
            program_translator.enable(False)
            dygraph_out, dygraph_loss = dygraph_net(x)

        np.testing.assert_allclose(dygraph_out.numpy(),
                                   static_out.numpy(),
                                   rtol=1e-05)
        np.testing.assert_allclose(dygraph_loss.numpy(),
                                   static_loss.numpy(),
                                   rtol=1e-05)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == '__main__':
    unittest.main()

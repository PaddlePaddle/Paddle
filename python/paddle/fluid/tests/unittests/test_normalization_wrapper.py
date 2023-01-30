#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
=======
from __future__ import print_function

import unittest
import paddle.fluid as fluid
import paddle.fluid.core as core
import numpy as np
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


class TestNormalization(unittest.TestCase):
    data_desc = {"name": "input", "shape": (2, 3, 7)}

    def gen_random_input(self):
<<<<<<< HEAD
        """Generate random input data."""
        self.data = np.random.random(size=self.data_desc["shape"]).astype(
            "float32"
        )

    def set_program(self, axis, epsilon):
        """Build the test program."""
        data = paddle.static.data(
            name=self.data_desc["name"],
            shape=self.data_desc["shape"],
            dtype="float32",
        )
        data.stop_gradient = False
        l2_norm = paddle.nn.functional.normalize(
            data, axis=axis, epsilon=epsilon
        )
        out = paddle.sum(l2_norm, axis=None)
=======
        """Generate random input data.
        """
        self.data = np.random.random(
            size=self.data_desc["shape"]).astype("float32")

    def set_program(self, axis, epsilon):
        """Build the test program.
        """
        data = fluid.layers.data(name=self.data_desc["name"],
                                 shape=self.data_desc["shape"],
                                 dtype="float32",
                                 append_batch_size=False)
        data.stop_gradient = False
        l2_norm = fluid.layers.l2_normalize(x=data, axis=axis, epsilon=epsilon)
        out = fluid.layers.reduce_sum(l2_norm, dim=None)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        fluid.backward.append_backward(loss=out)
        self.fetch_list = [l2_norm]

    def run_program(self):
<<<<<<< HEAD
        """Run the test program."""
=======
        """Run the test program.
        """
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for place in places:
            self.set_inputs(place)
            exe = fluid.Executor(place)

<<<<<<< HEAD
            (output,) = exe.run(
                fluid.default_main_program(),
                feed=self.inputs,
                fetch_list=self.fetch_list,
                return_numpy=True,
            )
            self.op_output = output

    def set_inputs(self, place):
        """Set the randomly generated data to the test program."""
=======
            output, = exe.run(fluid.default_main_program(),
                              feed=self.inputs,
                              fetch_list=self.fetch_list,
                              return_numpy=True)
            self.op_output = output

    def set_inputs(self, place):
        """Set the randomly generated data to the test program.
        """
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.inputs = {}
        tensor = fluid.Tensor()
        tensor.set(self.data, place)
        self.inputs[self.data_desc["name"]] = tensor

    def l2_normalize(self, data, axis, epsilon):
<<<<<<< HEAD
        """Compute the groundtruth."""
        output = data / np.broadcast_to(
            np.sqrt(np.sum(np.square(data), axis=axis, keepdims=True)),
            data.shape,
        )
        return output

    def test_l2_normalize(self):
        """Test the python wrapper for l2_normalize."""
        axis = 1
        # TODO(caoying) epsilon is not supported due to lack of a maximum_op.
=======
        """ Compute the groundtruth.
        """
        output = data / np.broadcast_to(
            np.sqrt(np.sum(np.square(data), axis=axis, keepdims=True)),
            data.shape)
        return output

    def test_l2_normalize(self):
        """ Test the python wrapper for l2_normalize.
        """
        axis = 1
        #TODO(caoying) epsilon is not supported due to lack of a maximum_op.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        epsilon = 1e-6

        self.gen_random_input()

        self.set_program(axis, epsilon)
        self.run_program()

        expect_output = self.l2_normalize(self.data, axis, epsilon)

        # check output
<<<<<<< HEAD
        np.testing.assert_allclose(
            self.op_output, expect_output, rtol=1e-05, atol=0.001
        )
=======
        np.testing.assert_allclose(self.op_output,
                                   expect_output,
                                   rtol=1e-05,
                                   atol=0.001)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == '__main__':
    unittest.main()

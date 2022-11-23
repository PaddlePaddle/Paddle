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

import unittest
import paddle.fluid as fluid
import paddle.fluid.core as core
import numpy as np


class TestMultiheadAttention(unittest.TestCase):

    def gen_random_input(self):
        """Generate random input data.
        """
        # batch_size, max_sequence_length, hidden dimension
        self.input_shape = (3, 13, 16)
        self.queries = np.random.random(size=self.input_shape).astype("float32")
        self.keys = np.random.random(size=self.input_shape).astype("float32")

    def set_program(self):
        """Build the test program.
        """
        queries = fluid.layers.data(name="queries",
                                    shape=self.input_shape,
                                    dtype="float32",
                                    append_batch_size=False)
        queries.stop_gradient = False
        keys = fluid.layers.data(name="keys",
                                 shape=self.input_shape,
                                 dtype="float32",
                                 append_batch_size=False)
        keys.stop_gradient = False

        contexts = fluid.nets.scaled_dot_product_attention(queries=queries,
                                                           keys=keys,
                                                           values=keys,
                                                           num_heads=8,
                                                           dropout_rate=0.)
        out = fluid.layers.reduce_sum(contexts, dim=None)
        fluid.backward.append_backward(loss=out)

        self.fetch_list = [contexts]

    def run_program(self):
        """Run the test program.
        """
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for place in places:
            self.set_inputs(place)
            exe = fluid.Executor(place)

            exe.run(fluid.default_startup_program())
            output = exe.run(fluid.default_main_program(),
                             feed=self.inputs,
                             fetch_list=self.fetch_list,
                             return_numpy=True)
            self.op_output = output

    def set_inputs(self, place):
        """Set the randomly generated data to the test program.
        """
        self.inputs = {}
        queries = fluid.Tensor()
        queries.set(self.queries, place)

        keys = fluid.Tensor()
        keys.set(self.keys, place)

        self.inputs["keys"] = keys
        self.inputs["queries"] = queries

    def test_multihead_attention(self):
        self.gen_random_input()

        self.set_program()
        self.run_program()

        #fixme(caoying) add more meaningfull unittest.


if __name__ == '__main__':
    unittest.main()

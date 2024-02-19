#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import unittest

import numpy as np

import paddle
import paddle.optimizer
import paddle.static
from paddle.utils.cpp_extension import load

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from op_test_ipu import IPUOpTest, np_dtype_to_base_str


def load_custom_ops():
    # load custom ops
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    custom_ops = load(
        name="custom_jit_ops",
        sources=[
            f"{cur_dir}/leaky_relu_cpu.cc",
            f"{cur_dir}/leaky_relu_ipu.cc",
        ],
        extra_cxx_cflags=['-DONNX_NAMESPACE=onnx'],
    )
    return custom_ops


class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_feed()
        self.set_feed_attr()
        self.set_attrs()

    def set_feed(self):
        self.feed = {
            "x": np.random.uniform(low=-2, high=2, size=[3, 5]).astype(
                'float32'
            ),
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = [
            np_dtype_to_base_str(x.dtype) for x in self.feed.values()
        ]

    def set_attrs(self):
        self.attrs = {'alpha': 0.1}

    def _test_base(self, run_ipu=True):
        scope = paddle.static.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        SEED = self.SEED
        paddle.seed(SEED)
        custom_ops = load_custom_ops()

        with paddle.static.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(
                    name=self.feed_list[0],
                    shape=self.feed_shape[0],
                    dtype=self.feed_dtype[0],
                )
                # custom op
                out = custom_ops.custom_leaky_relu(x, **self.attrs)
                fetch_list = [out.name]

            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            if run_ipu:
                feed_list = self.feed_list
                ipu_strategy = paddle.static.IpuStrategy()
                ipu_strategy.set_graph_config(is_training=False)

                # add name mapping for paddle custom op and popart custom ops
                # `paddle_op` was defined in leaky_relu_cpu.cc
                # `popart_op`, `domain` and `version` was defined in leaky_relu_ipu.cc
                ipu_strategy.add_custom_op(
                    paddle_op="custom_leaky_relu",
                    popart_op="LeakyRelu",
                    domain='custom.ops',
                    version=1,
                )

                program = paddle.static.IpuCompiledProgram(
                    main_prog, scope=scope, ipu_strategy=ipu_strategy
                ).compile(feed_list, fetch_list)
            else:
                program = main_prog

            result = exe.run(program, feed=self.feed, fetch_list=fetch_list)
            return result[0]

    def test_base(self):
        res0 = self._test_base(False)
        res1 = self._test_base(True)

        np.testing.assert_allclose(
            res0.flatten(), res1.flatten(), rtol=1e-05, atol=self.atol
        )

        self.assertTrue(res0.shape == res1.shape)


if __name__ == "__main__":
    unittest.main()

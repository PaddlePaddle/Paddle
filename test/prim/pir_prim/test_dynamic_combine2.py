# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.framework import core
from paddle.static import InputSpec


def apply_to_static(net, use_cinn, input_spec=None):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=input_spec,
        build_strategy=build_strategy,
        full_graph=True,
    )


def stack_net(x, y, z):
    temp = paddle.stack([x, y], axis=-1)
    return paddle.concat([temp, z])


class TestPrimMode1(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [128, 128]
        self.shape_y = [128, 128]
        self.shape_z = [128, 128, 2]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.y = np.random.random(self.shape_y).astype("float32")
        self.z = np.random.random(self.shape_z).astype("float32")
        self.net = stack_net
        self.enable_cinn = False

    def base_net(self, flag=None):
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        z = paddle.to_tensor(self.z)
        if flag == "prim":
            core._set_prim_all_enabled(True)
            fn = apply_to_static(
                self.net,
                use_cinn=self.enable_cinn,
                input_spec=[
                    InputSpec(shape=[None, None], dtype='float32'),
                    InputSpec(shape=[None, None], dtype='float32'),
                    InputSpec(shape=[None, 128, None], dtype='float32'),
                ],
            )
            fn.eval()
        else:
            fn = self.net
        res = fn(x, y, z)

        if flag == "prim":
            ops = [
                op.name()
                for op in fn.program_cache.last()[-1][-1]
                .infer_program.program.global_block()
                .ops
            ]
            assert "pd_op.stack" not in ops
            core._set_prim_all_enabled(False)
        return res

    def test_prim_all_dynamic(self):
        res_ref = self.base_net()
        res = self.base_net("prim")
        np.testing.assert_allclose(res_ref.numpy(), res.numpy(), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

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

import unittest

import numpy as np

import paddle
from paddle.base import core
from paddle.incubate.autograd import primapi

paddle.framework.random._manual_program_seed(2023)


def fn(x):
    dropout1 = paddle.nn.Dropout(p=0.5)
    dropout2 = paddle.nn.Dropout(p=0.6)
    y = dropout1(x)
    z = dropout2(y)
    return z


class TestCompositeCopyOp(unittest.TestCase):
    """This case is set to test copying op process even if some attrs of origin op has been blocked during constructing program."""

    def cal_composite(self, inputs):
        paddle.enable_static()
        core._set_prim_forward_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            y = fn(x)
            blocks = main_program.blocks

            fwd_ops = [op.type for op in blocks[0].ops]
            # Ensure that dropout in original block
            self.assertTrue('dropout' in fwd_ops)

            primapi.to_prim(blocks)

            fwd_ops_new = [op.type for op in blocks[0].ops]
            # Ensure that dropout is not splitted into small ops
            self.assertTrue('dropout' in fwd_ops_new)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(main_program, feed={'x': inputs}, fetch_list=[y])
        paddle.disable_static()
        core._set_prim_forward_enabled(False)
        return res

    def test_forward(self):
        core._set_prim_forward_blacklist("dropout")
        np_data = np.random.random([16, 64, 128, 128]).astype("float32")
        tensor_data = paddle.to_tensor(np_data)

        expect = fn(tensor_data).numpy()
        actual = self.cal_composite(np_data)[0]

        assert expect.dtype == actual.dtype
        np.testing.assert_allclose(
            expect,
            actual,
            rtol=0,
            atol=0,
        )


if __name__ == '__main__':
    unittest.main()

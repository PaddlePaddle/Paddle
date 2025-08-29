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

import paddle
from paddle import pir

paddle.enable_static()


def get_pir_program_and_param_map():
    with paddle.pir_utils.OldIrGuard():
        shape = [3, 3]
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp):
            # construct graph
            x = paddle.static.data('x', shape, dtype='float32')
            x.stop_gradient = False
            y = paddle.static.data('y', shape, dtype='float32')
            y.stop_gradient = False
            z = paddle.static.data('z', shape, dtype='float32')
            z.stop_gradient = False
            tmp1 = paddle.add(x, y)
            tmp2 = paddle.multiply(tmp1, z)
            tmp3 = paddle.matmul(tmp2, z)
            tmp4 = paddle.mean(tmp3, axis=-1, keepdim=True)
            tmp5 = paddle.rsqrt(tmp4)
            scale = paddle.tensor.fill_constant(
                shape=tmp5.shape[1:],
                dtype=tmp5.dtype,
                value=1.0,
            )
            scale.stop_gradient = True
            tmp6 = paddle.nn.functional.layer_norm(
                tmp5, tmp5.shape[1:], scale, None, 1e-5
            )
            tmp7 = paddle.nn.functional.dropout(tmp6, p=0.5)
            tmp8 = paddle.add(x, tmp7)
            tmp9 = paddle.concat(tmp8)

            test = paddle.rand([5, 1, 10])
            tmp_test_1 = paddle.squeeze(test, axis=1)
            out = paddle.mean(tmp9)
            # construct backward graph
            gradients = paddle.static.gradients(out, [x, y, z])

        pir_program, param_mapping = pir.translate_to_pir_with_param_map(
            mp.desc
        )
        return pir_program, param_mapping


if __name__ == "__main__":
    unittest.main()

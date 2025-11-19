# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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


import numpy as np

import paddle
import paddle.incubate
from paddle.base import core

paddle.enable_static()
np.random.seed(0)


def test_fuse_resnet_unit():
    tests = [[1, 4096, 768, 12], [10, 4097, 756, 12], [10, 4097, 756, 12]]
    for test in tests:
        batch_size = test[0]
        token_size = test[1]
        hidden_size = test[2]
        num_heads = test[3]
        dtype = np.float32

        place = paddle.CPUPlace()
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(program, startup_program):
            x = paddle.static.data(
                "x", [batch_size, token_size, hidden_size * 3], dtype=dtype
            )
            qkv = x.reshape(
                (batch_size, token_size, 3, num_heads, hidden_size // num_heads)
            ).transpose((2, 0, 3, 1, 4))

            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = q.matmul(k.transpose((0, 1, 3, 2)))

            attn = paddle.nn.functional.softmax(attn, axis=-1)

            out = (
                (attn.matmul(v))
                .transpose((0, 2, 1, 3))
                .reshape((-1, token_size, hidden_size))
            )

        graph = core.Graph(program.desc)
        core.get_pass("self_attention_fuse_pass").apply(graph)
        after_program = paddle.framework.IrGraph(graph).to_program()
        exe = paddle.static.Executor(place)
        exe.run(startup_program)

        feed = {
            "x": np.random.randn(
                batch_size, token_size, hidden_size * 3
            ).astype(dtype)
        }
        before_out = exe.run(program, feed=feed, fetch_list=[out.name])
        after_out = exe.run(after_program, feed=feed, fetch_list=[out.name])
        np.testing.assert_allclose(
            before_out[0], after_out[0], rtol=1e-05, atol=0.005
        )


if __name__ == '__main__':
    test_fuse_resnet_unit()

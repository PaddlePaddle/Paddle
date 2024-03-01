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

import numpy as np

import paddle
from paddle.autograd.ir_backward import grad as ir_grad
from paddle.base import core
from paddle.decomposition import decompose


def rms_norm(weight, hidden):
    variance = paddle.mean(paddle.pow(hidden, 2), axis=-1, keepdim=True)
    hidden = paddle.rsqrt(variance + 0.00001) * hidden
    return hidden * weight


np_weight = np.random.random([4096, 4096]).astype("float32")
np_hidden = np.random.random([4096, 4096]).astype("float32")
main_program0 = paddle.static.Program()
main_program1 = paddle.static.Program()
core._set_prim_all_enabled(True)
paddle.enable_static()
with paddle.static.program_guard(main_program0):
    weight = paddle.static.data(
        name="weight", shape=[4096, 4096], dtype="float32"
    )
    hidden = paddle.static.data(
        name="hidden", shape=[4096, 4096], dtype="float32"
    )
    weight.stop_gradient = False
    hidden.stop_gradient = False
    out = rms_norm(weight, hidden)
    out_grad = paddle.full(shape=[4096, 4096], fill_value=3, dtype="float32")
    [out] = decompose(main_program0, [out])
    [dweight, dhidden] = ir_grad(out, [weight, hidden], out_grad)
    exe = paddle.static.Executor(paddle.CUDAPlace(0))
    res = exe.run(
        feed={'weight': np_weight, 'hidden': np_hidden},
        fetch_list=[dweight, dhidden],
    )


with paddle.static.program_guard(main_program1):
    core._set_prim_all_enabled(True)
    weight = paddle.static.data(
        name="weight", shape=[4096, 4096], dtype="float32"
    )
    hidden = paddle.static.data(
        name="hidden", shape=[4096, 4096], dtype="float32"
    )
    weight.stop_gradient = False
    hidden.stop_gradient = False
    out = rms_norm(weight, hidden)
    out_grad = paddle.full(shape=[4096, 4096], fill_value=3, dtype="float32")
    [out] = decompose(main_program1, [out])
    [dweight, dhidden] = ir_grad(out, [weight, hidden], out_grad)
    main_program1, _ = paddle.decomposition.min_cut_auto_recompute(
        main_program1,
        [weight, hidden],
        [out],
        grad_outputs=[out_grad],
        fwd_op_end_idx=13,
    )
    exe = paddle.static.Executor(paddle.CUDAPlace(0))
    res_recompute = exe.run(
        feed={'weight': np_weight, 'hidden': np_hidden},
        fetch_list=[dweight, dhidden],
    )
np.testing.assert_allclose(res[0], res_recompute[0], rtol=1e-6, atol=1e-6)
np.testing.assert_allclose(res[1], res_recompute[1], rtol=1e-6, atol=1e-6)

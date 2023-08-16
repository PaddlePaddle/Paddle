# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np
from utils import (
    check_output_allclose,
    extra_cc_args,
    extra_nvcc_args,
    paddle_includes,
)

import paddle
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = f'{get_build_directory()}\\custom_tanh\\custom_tanh.pyd'
if os.name == 'nt' and os.path.isfile(file):
    cmd = f'del {file}'
    run_cmd(cmd, True)

custom_ops = load(
    name='custom_tanh_jit',
    sources=['custom_tanh_op.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cc flags
    extra_cuda_cflags=extra_nvcc_args,  # test for nvcc flags
    verbose=True,
)


def custom_tanh_double_grad_dynamic(func, device, dtype, np_x):
    paddle.set_device(device)

    t = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=False)
    t.retain_grads()

    out = func(t)
    out.stop_gradient = False
    out.retain_grads()

    dx = paddle.grad(
        outputs=[out], inputs=[t], create_graph=True, retain_graph=True
    )

    dx[0].retain_grads()
    dx[0].backward()

    assert out.grad is not None
    assert dx[0].grad is not None
    return dx[0].numpy(), dx[0].grad.numpy(), out.grad.numpy()


class TestCustomTanhDoubleGradJit(unittest.TestCase):
    def setUp(self):
        paddle.set_device('cpu')
        self.dtypes = ['float32', 'float64']
        self.devices = ['cpu']

    def test_double_grad_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                out, dx_grad, dout = custom_tanh_double_grad_dynamic(
                    custom_ops.custom_tanh, device, dtype, x
                )
                pd_out, pd_dx_grad, pd_dout = custom_tanh_double_grad_dynamic(
                    paddle.tanh, device, dtype, x
                )
                check_output_allclose(out, pd_out, "out", rtol=1e-05)
                check_output_allclose(dx_grad, pd_dx_grad, "out", rtol=1e-05)
                check_output_allclose(dout, pd_dout, "dout", rtol=1e-05)


if __name__ == "__main__":
    unittest.main()

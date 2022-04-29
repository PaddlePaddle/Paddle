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

import paddle
import paddle.static as static
from paddle.utils.cpp_extension import load, get_build_directory
from paddle.utils.cpp_extension.extension_utils import run_cmd
from utils import paddle_includes, extra_cc_args, extra_nvcc_args
from paddle.fluid.framework import _test_eager_guard, _enable_legacy_dygraph
_enable_legacy_dygraph()

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = '{}\\custom_tanh\\custom_tanh.pyd'.format(get_build_directory())
if os.name == 'nt' and os.path.isfile(file):
    cmd = 'del {}'.format(file)
    run_cmd(cmd, True)

custom_ops = load(
    name='custom_tanh_jit',
    sources=['custom_tanh_op.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cc flags
    extra_cuda_cflags=extra_nvcc_args,  # test for nvcc flags
    verbose=True)


def custom_tanh_double_grad_dynamic(func, device, dtype, np_x):
    paddle.set_device(device)

    t = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=False)

    out = func(t)
    out.stop_gradient = False

    dx = paddle.grad(
        outputs=[out], inputs=[t], create_graph=True, retain_graph=True)

    dx[0].backward()

    assert out.grad is not None
    assert dx[0].grad is not None
    return dx[0].numpy(), dx[0].grad.numpy(), out.grad.numpy()


class TestCustomTanhDoubleGradJit(unittest.TestCase):
    def setUp(self):
        paddle.set_device('cpu')
        self.dtypes = ['float32', 'float64']
        self.devices = ['cpu']

    def test_func_double_grad_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                out, dx_grad, dout = custom_tanh_double_grad_dynamic(
                    custom_ops.custom_tanh, device, dtype, x)
                pd_out, pd_dx_grad, pd_dout = custom_tanh_double_grad_dynamic(
                    paddle.tanh, device, dtype, x)
                self.assertTrue(
                    np.allclose(out, pd_out),
                    "custom op out: {},\n paddle api out: {}".format(out,
                                                                     pd_out))
                self.assertTrue(
                    np.allclose(dx_grad, pd_dx_grad),
                    "custom op dx grad: {},\n paddle api dx grad: {}".format(
                        dx_grad, pd_dx_grad))
                self.assertTrue(
                    np.allclose(dout, pd_dout),
                    "custom op out grad: {},\n paddle api out grad: {}".format(
                        dout, pd_dout))


if __name__ == "__main__":
    unittest.main()

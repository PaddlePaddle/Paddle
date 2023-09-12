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

import unittest

import numpy as np
import parameterized as param

import paddle
from paddle.base import core
from paddle.incubate.autograd import primapi

np.random.seed(2023)


place = (
    paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
)


@param.parameterized_class(
    ('name', 'x', 'p', 'is_test', 'mode', 'seed', 'dtype', 'place'),
    (
        (
            'fp32',
            np.random.rand(100000),
            0.3,
            False,
            'upscale_in_train',
            1002,
            'float32',
            place,
        ),
        (
            'fp64',
            np.random.rand(100000),
            0.7,
            False,
            'upscale_in_train',
            9999,
            'float64',
            place,
        ),
        (
            'is_test=True',
            np.random.rand(100000),
            0.5,
            True,
            'upscale_in_train',
            1002,
            'float32',
            place,
        ),
        (
            'p=1.0',
            np.random.rand(100000),
            1.0,
            True,
            'upscale_in_train',
            1002,
            'float32',
            place,
        ),
        (
            'p=1.0,test=False',
            np.random.rand(100000),
            1.0,
            False,
            'upscale_in_train',
            1002,
            'float32',
            place,
        ),
        (
            'p=0.0',
            np.random.rand(100000),
            1.0,
            True,
            'upscale_in_train',
            1002,
            'float32',
            place,
        ),
        (
            'downgrade_train',
            np.random.rand(100000),
            0.5,
            False,
            'downscale_in_infer',
            1002,
            'float32',
            place,
        ),
        (
            'fp32_cpu',
            np.random.rand(100000),
            0.6,
            False,
            'upscale_in_train',
            9899,
            'float64',
            paddle.CPUPlace(),
        ),
        (
            'fp64_cpu',
            np.random.rand(100000),
            0.6,
            False,
            'upscale_in_train',
            9899,
            'float64',
            paddle.CPUPlace(),
        ),
        (
            'downgrade_train_cpu',
            np.random.rand(100000),
            0.5,
            False,
            'downscale_in_infer',
            1002,
            'float32',
            paddle.CPUPlace(),
        ),
    ),
)
class TestCompositeDropout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.enable_static()
        cls.x = cls.x.astype(cls.dtype)

    @classmethod
    def tearDownClass(cls):
        paddle.disable_static()

    def test_comp(self):
        def dropout(x, p, is_test, mode, seed=0):
            paddle.seed(seed)
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                input_ = paddle.static.data('x', shape=x.shape, dtype=x.dtype)
                input_.stop_gradient = False
                output = paddle.nn.functional.dropout(
                    input_, p, training=(not is_test), mode=mode
                )
                if core._is_fwd_prim_enabled():
                    primapi.to_prim(mp.blocks)
                grad = paddle.static.gradients(output, input_)[0]
            exe = paddle.static.Executor(self.place)
            exe.run(sp)
            fwd, rev = exe.run(
                mp, feed={input_.name: x}, fetch_list=[output, grad]
            )
            return fwd, rev, mp

        core._set_prim_forward_enabled(False)
        core._set_prim_backward_enabled(False)
        desired_fwd, desired_rev, _ = dropout(
            self.x, self.p, self.is_test, self.mode, self.seed
        )

        core._set_prim_forward_enabled(True)
        core._set_prim_backward_enabled(False)
        actual_fwd, actual_rev, prog = dropout(
            self.x, self.p, self.is_test, self.mode, self.seed
        )

        self.assertTrue('dropout' not in [op.type for op in prog.block(0).ops])

        np.testing.assert_allclose(
            actual_fwd.sum(),
            desired_fwd.sum(),
            rtol=1e-2,  # mean of uniform distribution, scale for avoid random failed
            atol=0,
        )
        np.testing.assert_allclose(
            actual_rev.sum(),
            desired_rev.sum(),
            rtol=1e-2,  # mean of uniform distribution, scale for avoid random failed
            atol=0,
        )

        core._set_prim_forward_enabled(False)
        core._set_prim_backward_enabled(True)
        actual_fwd, actual_rev, _ = dropout(
            self.x, self.p, self.is_test, self.mode, self.seed
        )
        np.testing.assert_allclose(
            actual_fwd.sum(),
            desired_fwd.sum(),
            rtol=1e-2,  # mean of uniform distribution, scale for avoid random failed
            atol=0,
        )
        np.testing.assert_allclose(
            actual_rev.sum(),
            desired_rev.sum(),
            rtol=1e-2,  # mean of uniform distribution, scale for avoid random failed
            atol=0,
        )
        core._set_prim_all_enabled(True)
        actual_fwd, actual_rev, _ = dropout(
            self.x, self.p, self.is_test, self.mode, self.seed
        )
        np.testing.assert_allclose(
            actual_fwd.sum(),
            desired_fwd.sum(),
            rtol=1e-2,  # mean of uniform distribution, scale for avoid random failed
            atol=0,
        )
        np.testing.assert_allclose(
            actual_rev.sum(),
            desired_rev.sum(),
            rtol=1e-2,  # mean of uniform distribution, scale for avoid random failed
            atol=0,
        )


if __name__ == '__main__':
    unittest.main()

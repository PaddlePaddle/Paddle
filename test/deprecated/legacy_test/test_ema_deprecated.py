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

import os
import unittest

import numpy as np

import paddle
from paddle import base

paddle.enable_static()


class TestExponentialMovingAverage(unittest.TestCase):
    def setUp(self):
        self._places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            self._places.append(base.CPUPlace())
        if base.core.is_compiled_with_cuda():
            self._places.append(base.CUDAPlace(0))
        self._ema_decay = 0.999
        self._param_name = "fc.weight"

        self._train_program = base.Program()
        self._startup_prog = base.Program()
        with base.program_guard(self._train_program, self._startup_prog):
            with base.unique_name.guard():
                data = paddle.static.data(
                    name='x', shape=[-1, 5], dtype='float32'
                )
                hidden = paddle.static.nn.fc(
                    x=data, size=10, weight_attr=self._param_name
                )
                cost = paddle.mean(hidden)

                self._test_program = base.default_main_program().clone(
                    for_test=True
                )

                optimizer = paddle.optimizer.Adam(learning_rate=0.001)
                optimizer.minimize(cost)

                self._ema = paddle.static.ExponentialMovingAverage(
                    self._ema_decay
                )
                self._ema.update()

    def train(self, place):
        exe = base.Executor(place)
        exe.run(self._startup_prog)

        params = []
        for pass_id in range(2):
            for batch_id in range(3):
                data = np.random.random(size=(10, 5)).astype('float32')
                tmp_param = np.array(
                    base.global_scope().find_var(self._param_name).get_tensor()
                )
                exe.run(program=self._train_program, feed={'x': data})
                tmp_param = np.array(
                    base.global_scope().find_var(self._param_name).get_tensor()
                )
                params.append(tmp_param)

        with self._ema.apply(exe):
            final_ema = np.array(
                base.global_scope().find_var(self._param_name).get_tensor()
            )
            data = np.random.random(size=(10, 5)).astype('float32')
            exe.run(program=self._test_program, feed={'x': data})
        return params, final_ema

    def test_check_ema(self):
        for place in self._places:
            params, final_ema = self.train(place)
            manu_ema = np.zeros_like(final_ema)
            if len(params) > 0:
                for param in params:
                    manu_ema = (
                        self._ema_decay * manu_ema
                        + (1 - self._ema_decay) * param
                    )
                manu_ema = manu_ema / (1.0 - self._ema_decay ** len(params))
            np.testing.assert_allclose(manu_ema, final_ema, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()

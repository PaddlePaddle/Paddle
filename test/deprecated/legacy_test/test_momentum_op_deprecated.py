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

import numpy
import numpy as np

import paddle


def calculate_momentum_by_numpy(
    param,
    grad,
    mu,
    velocity,
    use_nesterov,
    learning_rate,
    regularization_method=None,
    regularization_coeff=1.0,
):
    if regularization_method == "l2_decay":
        grad = grad + regularization_coeff * param

        velocity_out = mu * velocity + grad
        if use_nesterov:
            param_out = param - (grad + velocity_out * mu) * learning_rate
        else:
            param_out = param - learning_rate * velocity_out
    else:
        velocity_out = mu * velocity + grad
        if use_nesterov:
            param_out = (
                param - grad * learning_rate - velocity_out * mu * learning_rate
            )
        else:
            param_out = param - learning_rate * velocity_out

    return param_out, velocity_out


def momentum_wrapper(
    param,
    grad,
    velocity,
    learning_rate=1.0,
    master_param=None,
    mu=0.0,
    use_nesterov=False,
    regularization_method="",
    regularization_coeff=0.0,
    multi_precision=False,
    rescale_grad=1.0,
):
    return paddle._C_ops.momentum_(
        param,
        grad,
        velocity,
        learning_rate,
        master_param,
        mu,
        use_nesterov,
        regularization_method,
        regularization_coeff,
        multi_precision,
        rescale_grad,
    )


class TestMultiTensorMomentumStatic(unittest.TestCase):
    def _momentum_optimize_static(
        self, place, use_amp=False, use_multi_tensor=False
    ):
        paddle.enable_static()
        paddle.seed(10)
        np.random.seed(10)
        if place == 'cpu':
            use_amp = False
        exe = paddle.static.Executor(place=place)
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        optimizer = paddle.optimizer.Momentum(
            multi_precision=use_amp, use_multi_tensor=use_multi_tensor
        )
        if use_amp:
            optimizer = paddle.static.amp.decorate(
                optimizer,
                init_loss_scaling=128.0,
                use_dynamic_loss_scaling=True,
                use_pure_fp16=True,
                use_fp16_guard=False,
            )
        with paddle.static.program_guard(train_program, startup_program):
            if use_amp:
                data = paddle.static.data(
                    shape=[2, 2], name='X', dtype='float16'
                )
            else:
                data = paddle.static.data(
                    shape=[2, 2], name='X', dtype='float32'
                )
            hidden = paddle.static.nn.fc(x=data, size=10)
            loss = paddle.mean(hidden)
            optimizer.minimize(loss)
        exe.run(startup_program)
        if use_amp:
            optimizer.amp_init(
                place=paddle.CUDAPlace(0), scope=paddle.static.global_scope()
            )
            x = numpy.random.random(size=(2, 2)).astype('float16')
        else:
            x = numpy.random.random(size=(2, 2)).astype('float32')
        out = []
        for idx in range(5):
            (loss_data,) = exe.run(
                train_program, feed={"X": x}, fetch_list=[loss]
            )
            out.append(loss_data)
        return out

    def _get_places(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            places.append('cpu')
        if paddle.is_compiled_with_cuda():
            places.append('gpu')
        return places

    def _check_with_place_amp(self, place, use_amp):
        output1 = self._momentum_optimize_static(
            place=place, use_amp=use_amp, use_multi_tensor=True
        )
        output2 = self._momentum_optimize_static(
            place=place, use_amp=use_amp, use_multi_tensor=False
        )
        for idx in range(len(output1)):
            np.testing.assert_allclose(output1[idx], output2[idx], rtol=1e-05)

    def test_main(self):
        for place in self._get_places():
            use_amp_list = [True, False]
            for use_amp in use_amp_list:
                self._check_with_place_amp(place, use_amp)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()

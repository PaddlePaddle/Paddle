#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from collections import defaultdict

import numpy as np

import paddle
from paddle import base
from paddle.base.backward import _append_grad_suffix_

paddle.enable_static()

np.random.seed(10)

SHAPE = [16, 10]


class TestModel(paddle.nn.Layer):
    def __init__(self, param_lr, y_no_grad, cond_i):
        super().__init__()
        self.param_lr = param_lr

        self.shape = SHAPE
        self.y_no_grad = y_no_grad
        self.cond_i = cond_i
        self._init_param()

        self.param_x = paddle.create_parameter(
            dtype="float32",
            shape=self.shape,
            attr=base.ParamAttr(learning_rate=self.param_lr, name="param_x"),
            default_initializer=paddle.nn.initializer.Assign(self.x),
        )

        self.param_y = paddle.create_parameter(
            dtype="float32",
            shape=self.shape,
            attr=base.ParamAttr(learning_rate=self.param_lr, name="param_y"),
            default_initializer=paddle.nn.initializer.Assign(self.y),
        )
        self.param_z = paddle.create_parameter(
            dtype="float32",
            shape=self.shape,
            attr=base.ParamAttr(learning_rate=self.param_lr, name="param_z"),
            default_initializer=paddle.nn.initializer.Assign(self.z),
        )

    def _init_param(self):
        self.x = np.ones(self.shape).astype('float32')
        self.y = np.ones(self.shape).astype('float32') * 2.0
        self.z = np.ones(self.shape).astype('float32') * 3.0

    def forward(self):
        sum_xy = paddle.add(self.param_x, self.param_y, name='sum_xy')
        sub_yz = paddle.subtract(self.param_y, self.param_z, name='sub_yz')
        # useless = paddle.static.nn.fc(self.param_x, size=1, name='fc_useless')

        def cond_true():
            cond_yz = paddle.add(self.param_y, self.param_z, name='sum_cond_yz')
            # param_y will not be updated
            self.param_y.stop_gradient = self.y_no_grad
            cond_res = paddle.add(cond_yz, self.param_z, name='sum_cond_true')
            cond_useless = paddle.multiply(self.param_x, self.param_y)
            return cond_res

        def cond_false():
            cond_res = paddle.add(
                self.param_y, self.param_z, name='sum_cond_false'
            )
            cond_useless = paddle.multiply(self.param_z, self.param_z)
            return cond_res

        self.cond_i = paddle.assign(np.array([self.cond_i], dtype='float32'))
        sum_cond = paddle.static.nn.cond(
            self.cond_i > 1.0, cond_true, cond_false
        )
        sum_all = paddle.add_n([sum_xy, sub_yz, sum_cond])

        return sum_all

    def _calc_gradient(self, cond_i):
        """
        Calculate grads of params
        """
        grads = []
        d_out_val = np.ones_like(self.x).astype("float32") / np.prod(self.shape)
        grads.append(d_out_val)  # x_grad
        if cond_i > 1:
            y_grad_ratio, z_grad_ratio = 0 if self.y_no_grad else 3, 1
        else:
            y_grad_ratio, z_grad_ratio = 3, 0
        if not self.y_no_grad:
            grads.append(d_out_val * y_grad_ratio)  # y_grad
        grads.append(d_out_val * z_grad_ratio)  # z_grad

        return grads

    def _apply_gradient(self, param, grad, name):
        """
        The way of updating grad in optimizer.(such as SGD)
        This method should be override.
        """
        return param - self.attr['lr'] * grad


class SimpleNetWithCond:
    """
    Build net with conditional Block and useless layers.
    """

    def __init__(self, test_optimizer, param_lr=1.0, y_no_grad=False):
        self.optimizer = test_optimizer
        self.param_lr = param_lr
        self.shape = SHAPE
        self.y_no_grad = y_no_grad
        self._init_param()

    def _init_param(self):
        self.x = np.ones(self.shape).astype('float32')
        self.y = np.ones(self.shape).astype('float32') * 2.0
        self.z = np.ones(self.shape).astype('float32') * 3.0

    def _calc_gradient(self, cond_i):
        """
        Calculate grads of params
        """
        grads = []
        d_out_val = np.ones_like(self.x).astype("float32") / np.prod(self.shape)
        grads.append(d_out_val)  # x_grad
        if cond_i > 1:
            y_grad_ratio, z_grad_ratio = 0 if self.y_no_grad else 3, 1
        else:
            y_grad_ratio, z_grad_ratio = 3, 0
        if not self.y_no_grad:
            grads.append(d_out_val * y_grad_ratio)  # y_grad
        grads.append(d_out_val * z_grad_ratio)  # z_grad

        return grads

    def build_net(self, cond_i, use_bf16=False):
        """
        pseudo code:
            sum_xy = x + y
            sub_yz = y - z
            if i > 1:
                internal = y + z
                sum_cond = internal + z
            else:
                sum_cond = y + z
            sum_all = sum_xy + sum_yz + sum_cond
            mean_out = mean(sum_all)
            optimizer.minimize(mean_out)
        """
        param_x = paddle.create_parameter(
            dtype="float32",
            shape=self.shape,
            attr=base.ParamAttr(learning_rate=self.param_lr, name="param_x"),
            default_initializer=paddle.nn.initializer.Assign(self.x),
        )

        param_y = paddle.create_parameter(
            dtype="float32",
            shape=self.shape,
            attr=base.ParamAttr(learning_rate=self.param_lr, name="param_y"),
            default_initializer=paddle.nn.initializer.Assign(self.y),
        )
        param_z = paddle.create_parameter(
            dtype="float32",
            shape=self.shape,
            attr=base.ParamAttr(learning_rate=self.param_lr, name="param_z"),
            default_initializer=paddle.nn.initializer.Assign(self.z),
        )

        sum_xy = paddle.add(param_x, param_y, name='sum_xy')
        sub_yz = paddle.subtract(param_y, param_z, name='sub_yz')
        useless = paddle.static.nn.fc(param_x, size=1, name='fc_useless')

        def cond_true():
            cond_yz = paddle.add(param_y, param_z, name='sum_cond_yz')
            # param_y will not be updated
            param_y.stop_gradient = self.y_no_grad
            cond_res = paddle.add(cond_yz, param_z, name='sum_cond_true')
            cond_useless = paddle.multiply(param_x, param_y)
            return cond_res

        def cond_false():
            cond_res = paddle.add(param_y, param_z, name='sum_cond_false')
            cond_useless = paddle.multiply(param_z, param_z)
            return cond_res

        cond_i = paddle.assign(np.array([cond_i], dtype='float32'))
        sum_cond = paddle.static.nn.cond(cond_i > 1.0, cond_true, cond_false)
        sum_all = paddle.add_n([sum_xy, sub_yz, sum_cond])
        mean_out = paddle.mean(sum_all)
        if use_bf16:
            from paddle.static import amp

            self.optimizer = amp.bf16.decorate_bf16(
                self.optimizer,
                amp_lists=amp.bf16.AutoMixedPrecisionListsBF16(
                    custom_fp32_list={'elementwise_add'}
                ),
                use_bf16_guard=False,
                use_pure_bf16=True,
            )

        _, params_grads = self.optimizer.minimize(mean_out)

        if paddle.framework.in_pir_mode():
            for param, grad in params_grads:
                if param.is_same(param_x):
                    param_x_grad = grad
                elif param.is_same(param_y):
                    param_y_grad = grad
                elif param.is_same(param_z):
                    param_z_grad = grad
            fetch_list = (
                [param_x, param_z, param_x_grad, param_z_grad]
                if self.y_no_grad
                else [
                    param_x,
                    param_y,
                    param_z,
                    param_x_grad,
                    param_y_grad,
                    param_z_grad,
                ]
            )
        else:
            fetch_list = (
                ["param_x", "param_z"]
                if self.y_no_grad
                else ["param_x", "param_y", "param_z"]
            )
            fetch_list += [_append_grad_suffix_(param) for param in fetch_list]

        return fetch_list, self.optimizer


class TestOptimizer(unittest.TestCase):
    """
    TestOptimizer BaseClass to be inherited to test other Optimizer.
    And only need to implement two functions:
        setUp(): to set config info of optimizer, including Optimizer and its hyper-parameter.
        _apply_gradient(): to implement the way of updating grad.
    """

    def setUp(self):
        self._init_config()
        self.optimizer = paddle.optimizer.SGD(learning_rate=0.001)
        self.attr = {}

    def _init_config(self):
        self.NetClass = SimpleNetWithCond
        self.param_lr = [1.0, 2.0]
        self.cond_i = [0.1, 3]
        self.y_no_grad = [True, False]

    def test_optimizer(self):
        self._check_grads()

    def _apply_gradient(self, param, grad, name):
        """
        The way of updating grad in optimizer.(such as SGD)
        This method should be override.
        """
        return param - self.attr['lr'] * grad

    def _apply_optimize(self, net, grads):
        """
        apply to update all params in the net.
        """
        net.x = self._apply_gradient(net.x, grads[0], 'x')
        if len(grads) == 2:
            net.z = self._apply_gradient(net.z, grads[1], 'z')
            res = [net.x, net.z]
        else:
            net.y = self._apply_gradient(net.y, grads[1], 'y')
            net.z = self._apply_gradient(net.z, grads[2], 'z')
            res = [net.x, net.y, net.z]

        return res

    def _init_param_attr(self):
        self.param_attr = {}
        for key in ['x', 'y', 'z']:
            self.param_attr[key] = self.attr.copy()

    def _check_grads(self, use_bf16=False):
        """
        main logic code to check the validity of apply_optimize.
        """
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        # test on CPU and GPU
        for place in places:
            for param_lr in self.param_lr:
                for cond_i in self.cond_i:
                    for y_no_grad in self.y_no_grad:
                        self.attr['lr'] = (
                            param_lr * self.optimizer._learning_rate
                        )
                        self._init_param_attr()

                        main_program = base.Program()
                        init_program = base.Program()
                        with base.program_guard(main_program, init_program):
                            # reset optimizer._accumulators to avoid duplicate name in loop.
                            self.optimizer._accumulators = defaultdict(
                                lambda: {}
                            )
                            fetch_list = []
                            if paddle.framework.in_pir_mode():
                                model = TestModel(param_lr, y_no_grad, cond_i)
                                self.optimizer = paddle.optimizer.SGD(
                                    learning_rate=1.0,
                                    parameters=model.parameters(),
                                )
                                params_grads = []
                                if use_bf16:
                                    model, self.optimizer = paddle.amp.decorate(
                                        models=model,
                                        optimizers=self.optimizer,
                                        level='O2',
                                        dtype='bfloat16',
                                    )

                                    with paddle.amp.auto_cast(
                                        level='O2',
                                        dtype='bfloat16',
                                        use_promote=True,
                                    ):
                                        out = model()
                                        loss = paddle.mean(out)
                                        (
                                            _,
                                            params_grads,
                                        ) = self.optimizer.minimize(loss)
                                else:
                                    out = model()
                                    loss = paddle.mean(out)
                                    _, params_grads = self.optimizer.minimize(
                                        loss
                                    )

                                param_x, param_y, param_z = model.parameters()

                                for param, grad in params_grads:
                                    if param.is_same(param_x):
                                        param_x_grad = grad
                                    elif param.is_same(param_y):
                                        param_y_grad = grad
                                    elif param.is_same(param_z):
                                        param_z_grad = grad
                                fetch_list = (
                                    [
                                        param_x,
                                        param_z,
                                        param_x_grad,
                                        param_z_grad,
                                    ]
                                    if y_no_grad
                                    else [
                                        param_x,
                                        param_y,
                                        param_z,
                                        param_x_grad,
                                        param_y_grad,
                                        param_z_grad,
                                    ]
                                )
                                exe = base.Executor(place)
                                exe.run(init_program)
                                if not paddle.framework.in_pir_mode():
                                    if use_bf16:
                                        self.optimizer.amp_init(exe.place)

                                for batch_i in range(2):
                                    res = exe.run(
                                        main_program, fetch_list=fetch_list
                                    )
                            else:
                                test_net = self.NetClass(
                                    self.optimizer, param_lr, y_no_grad
                                )
                                (
                                    fetch_list,
                                    decorated_optimizer,
                                ) = test_net.build_net(cond_i, use_bf16)
                                if use_bf16:
                                    self.optimizer = decorated_optimizer

                                exe = base.Executor(place)
                                exe.run(init_program)
                                if use_bf16:
                                    self.optimizer.amp_init(exe.place)

                                # Train 2 steps to check validity
                                for batch_i in range(2):
                                    res = exe.run(
                                        main_program, fetch_list=fetch_list
                                    )
                                    gt_grads = test_net._calc_gradient(cond_i)
                                    gt_params = self._apply_optimize(
                                        test_net, gt_grads
                                    )
                                    param_grads = gt_params + gt_grads
                                    for i in range(len(res)):
                                        np.testing.assert_allclose(
                                            res[i], param_grads[i]
                                        )


@unittest.skipIf(
    not base.core.supports_bfloat16(), "place does not support BF16 evaluation"
)
class TestSGDOptimizer(TestOptimizer):
    def test_optimizer_multiblock_except(self):
        if not paddle.framework.in_pir_mode():
            with self.assertRaisesRegex(
                ValueError, "var param_y not in this block"
            ):
                self._check_grads(use_bf16=True)

    def test_optimizer_amp(self):
        if paddle.framework.in_pir_mode():
            self._check_grads(use_bf16=True)


if __name__ == '__main__':
    unittest.main()

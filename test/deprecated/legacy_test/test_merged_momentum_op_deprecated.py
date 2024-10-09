# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from collections import OrderedDict

import numpy as np

import paddle
from paddle.base.layer_helper import LayerHelper


def run_momentum_op(
    params,
    grads,
    velocities,
    master_params,
    learning_rate,
    place,
    multi_precision,
    mu=0.9,
    rescale_grad=0.01,
    use_merged=False,
):
    assert len(params) == len(grads)
    assert len(params) == len(velocities)
    if multi_precision:
        assert len(params) == len(master_params)
    op_type = 'merged_momentum' if use_merged else 'momentum'
    main = paddle.static.Program()
    startup = paddle.static.Program()
    with paddle.static.program_guard(main, startup):
        helper = LayerHelper(op_type, **locals())
        attrs = {
            'mu': mu,
            'multi_precision': multi_precision,
            'rescale_grad': rescale_grad,
        }

        param_vars = [
            helper.create_variable(
                persistable=True, shape=p.shape, dtype=p.dtype
            )
            for p in params
        ]
        grad_vars = [
            helper.create_variable(shape=g.shape, dtype=g.dtype) for g in grads
        ]
        velocity_vars = [
            helper.create_variable(
                persistable=True, shape=v.shape, dtype=v.dtype
            )
            for v in velocities
        ]
        lr_var = helper.create_variable(
            persistable=True,
            shape=learning_rate.shape,
            dtype=learning_rate.dtype,
        )

        feed_dict = OrderedDict()

        feed_dict.update(
            OrderedDict(
                [
                    (p_var.name, p_val)
                    for p_var, p_val in zip(param_vars, params)
                ]
            )
        )
        feed_dict.update(
            OrderedDict(
                [
                    (v_var.name, v_val)
                    for v_var, v_val in zip(velocity_vars, velocities)
                ]
            )
        )
        fetch_list = list(feed_dict.keys())

        feed_dict.update(
            OrderedDict(
                [(g_var.name, g_val) for g_var, g_val in zip(grad_vars, grads)]
            )
        )
        feed_dict.update({lr_var.name: learning_rate})

        if multi_precision:
            master_param_vars = [
                helper.create_variable(
                    persistable=True, shape=p.shape, dtype=p.dtype
                )
                for p in master_params
            ]
            feed_dict.update(
                OrderedDict(
                    [
                        (mp_var.name, mp_val)
                        for mp_var, mp_val in zip(
                            master_param_vars, master_params
                        )
                    ]
                )
            )
            # CPUPlace does not use MasterParam
            if isinstance(place, paddle.CUDAPlace):
                fetch_list = fetch_list + [
                    mp_var.name for mp_var in master_param_vars
                ]
        else:
            master_param_vars = None

        if not use_merged:
            for i, (p, g, v) in enumerate(
                zip(param_vars, grad_vars, velocity_vars)
            ):
                inputs = {
                    'Param': p,
                    'Grad': g,
                    'Velocity': v,
                    'LearningRate': lr_var,
                }
                outputs = {'ParamOut': p, 'VelocityOut': v}
                if multi_precision:
                    inputs['MasterParam'] = master_param_vars[i]
                    outputs['MasterParamOut'] = master_param_vars[i]
                helper.append_op(
                    type=op_type, inputs=inputs, outputs=outputs, attrs=attrs
                )
        else:
            inputs = {
                'Param': param_vars,
                'Grad': grad_vars,
                'Velocity': velocity_vars,
                'LearningRate': lr_var,
            }
            outputs = {'ParamOut': param_vars, 'VelocityOut': velocity_vars}
            if multi_precision:
                inputs['MasterParam'] = master_param_vars
                outputs['MasterParamOut'] = master_param_vars
            helper.append_op(
                type=op_type, inputs=inputs, outputs=outputs, attrs=attrs
            )

    exe = paddle.static.Executor(place)
    with paddle.static.scope_guard(paddle.static.Scope()):
        exe.run(startup)
        return exe.run(main, feed=feed_dict, fetch_list=fetch_list)


def run_momentum_op2(
    params,
    grads,
    velocities,
    master_params,
    learning_rate,
    place,
    multi_precision,
    mu=0.9,
    rescale_grad=0.01,
    use_merged=False,
    use_nesterov=True,
):
    assert len(params) == len(grads)
    assert len(params) == len(velocities)
    if multi_precision:
        assert len(params) == len(master_params)
    op_type = 'merged_momentum' if use_merged else 'momentum'
    main = paddle.static.Program()
    startup = paddle.static.Program()
    with paddle.static.program_guard(main, startup):
        helper = LayerHelper(op_type, **locals())

        param_vars = [
            helper.create_variable(
                persistable=True, shape=p.shape, dtype=p.dtype
            )
            for p in params
        ]
        grad_vars = [
            helper.create_variable(shape=g.shape, dtype=g.dtype) for g in grads
        ]
        velocity_vars = [
            helper.create_variable(
                persistable=True, shape=v.shape, dtype=v.dtype
            )
            for v in velocities
        ]
        lr_var = helper.create_variable(
            persistable=True,
            shape=learning_rate.shape,
            dtype=learning_rate.dtype,
        )

        feed_dict = OrderedDict()

        feed_dict.update(
            OrderedDict(
                [
                    (p_var.name, p_val)
                    for p_var, p_val in zip(param_vars, params)
                ]
            )
        )
        feed_dict.update(
            OrderedDict(
                [
                    (v_var.name, v_val)
                    for v_var, v_val in zip(velocity_vars, velocities)
                ]
            )
        )
        fetch_list = list(feed_dict.keys())

        feed_dict.update(
            OrderedDict(
                [(g_var.name, g_val) for g_var, g_val in zip(grad_vars, grads)]
            )
        )
        feed_dict.update({lr_var.name: learning_rate})

        if multi_precision:
            master_param_vars = [
                helper.create_variable(
                    persistable=True, shape=p.shape, dtype=p.dtype
                )
                for p in master_params
            ]
            feed_dict.update(
                OrderedDict(
                    [
                        (mp_var.name, mp_val)
                        for mp_var, mp_val in zip(
                            master_param_vars, master_params
                        )
                    ]
                )
            )
            # CPUPlace does not use MasterParam
            if isinstance(place, paddle.CUDAPlace):
                fetch_list = fetch_list + [
                    mp_var.name for mp_var in master_param_vars
                ]
        else:
            master_param_vars = None

        if not use_merged:
            for i, (p, g, v) in enumerate(
                zip(param_vars, grad_vars, velocity_vars)
            ):
                inputs = {
                    'Param': p,
                    'Grad': g,
                    'Velocity': v,
                    'LearningRate': lr_var,
                }
                outputs = {'ParamOut': p, 'VelocityOut': v}
                if multi_precision:
                    inputs['MasterParam'] = master_param_vars[i]
                    outputs['MasterParamOut'] = master_param_vars[i]
                attrs = {
                    'mu': mu,
                    'multi_precision': multi_precision,
                    'rescale_grad': rescale_grad,
                    'use_nesterov': use_nesterov,
                    'regularization_method': 'l2_decay',
                    'regularization_coeff': 2.0,
                }
                helper.append_op(
                    type=op_type, inputs=inputs, outputs=outputs, attrs=attrs
                )
        else:
            inputs = {
                'Param': param_vars,
                'Grad': grad_vars,
                'Velocity': velocity_vars,
                'LearningRate': lr_var,
            }
            outputs = {'ParamOut': param_vars, 'VelocityOut': velocity_vars}
            if multi_precision:
                inputs['MasterParam'] = master_param_vars
                outputs['MasterParamOut'] = master_param_vars
            attrs = {
                'mu': mu,
                'multi_precision': multi_precision,
                'rescale_grad': rescale_grad,
                'use_nesterov': use_nesterov,
                'regularization_method': [
                    'l2_decay' for i in range(len(param_vars))
                ],
                'regularization_coeff': [2.0 for i in range(len(param_vars))],
            }
            helper.append_op(
                type=op_type, inputs=inputs, outputs=outputs, attrs=attrs
            )

    exe = paddle.static.Executor(place)
    with paddle.static.scope_guard(paddle.static.Scope()):
        exe.run(startup)
        return exe.run(main, feed=feed_dict, fetch_list=fetch_list)


class TestMergedMomentum(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.shapes = [[3, 4], [2, 7], [5, 6], [7, 8]]

        self.seed = 10

    def gen_rand_data(self, shapes, dtype):
        return [np.random.random(s).astype(dtype) for s in shapes]

    def prepare_data(self, shapes, multi_precision, seed, place):
        np.random.seed(seed)
        mp_dtype = np.float32
        dtype = (
            np.float16
            if multi_precision and isinstance(place, paddle.CUDAPlace)
            else np.float32
        )
        params = self.gen_rand_data(shapes, dtype)
        grads = self.gen_rand_data(shapes, dtype)
        velocities = self.gen_rand_data(shapes, mp_dtype)
        learning_rate = self.gen_rand_data([[1]], mp_dtype)[0]
        if multi_precision:
            master_params = [p.astype(mp_dtype) for p in params]
        else:
            master_params = None
        return params, grads, velocities, master_params, learning_rate

    def check_with_place(self, place, multi_precision):
        (
            params,
            grads,
            velocities,
            master_params,
            learning_rate,
        ) = self.prepare_data(self.shapes, multi_precision, self.seed, place)

        def run_op(use_merged):
            # FIXME(zengjinle): CPU Momentum Op does not support rescale_grad
            rescale_grad = 1.0 if isinstance(place, paddle.CPUPlace) else 0.01
            return run_momentum_op(
                params,
                grads,
                velocities,
                master_params,
                learning_rate,
                place,
                multi_precision,
                rescale_grad=rescale_grad,
                use_merged=use_merged,
            )

        outs1 = run_op(True)
        outs2 = run_op(False)
        self.assertEqual(len(outs1), len(outs2))
        for i, (out1, out2) in enumerate(zip(outs1, outs2)):
            if isinstance(place, paddle.CUDAPlace):
                np.testing.assert_array_equal(out1, out2)
            else:
                np.testing.assert_allclose(out1, out2, rtol=1e-05, atol=1e-07)

    def get_places(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            places.append(paddle.CPUPlace())
        if paddle.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        return places

    def test_main(self):
        for multi_precision in [False, True]:
            for place in self.get_places():
                self.check_with_place(place, multi_precision)


class TestMergedMomentum2(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.shapes = [[3, 4], [2, 7], [5, 6], [7, 8]]
        self.seed = 10

    def gen_rand_data(self, shapes, dtype):
        return [np.random.random(s).astype(dtype) for s in shapes]

    def prepare_data(self, shapes, multi_precision, seed, place):
        np.random.seed(seed)
        mp_dtype = np.float32
        dtype = (
            np.float16
            if multi_precision and isinstance(place, paddle.CUDAPlace)
            else np.float32
        )
        params = self.gen_rand_data(shapes, dtype)
        grads = self.gen_rand_data(shapes, dtype)
        velocities = self.gen_rand_data(shapes, mp_dtype)
        learning_rate = self.gen_rand_data([[1]], mp_dtype)[0]
        if multi_precision:
            master_params = [p.astype(mp_dtype) for p in params]
        else:
            master_params = None
        return params, grads, velocities, master_params, learning_rate

    def check_with_place(self, place, multi_precision):
        (
            params,
            grads,
            velocities,
            master_params,
            learning_rate,
        ) = self.prepare_data(self.shapes, multi_precision, self.seed, place)

        def run_op(use_nesterov, use_merged):
            # FIXME(zengjinle): CPU Momentum Op does not support rescale_grad
            rescale_grad = 1.0 if isinstance(place, paddle.CPUPlace) else 0.01
            return run_momentum_op2(
                params,
                grads,
                velocities,
                master_params,
                learning_rate,
                place,
                multi_precision,
                rescale_grad=rescale_grad,
                use_merged=use_merged,
                use_nesterov=use_nesterov,
            )

        outs1 = run_op(use_nesterov=True, use_merged=True)
        outs2 = run_op(use_nesterov=True, use_merged=False)
        self.assertEqual(len(outs1), len(outs2))
        for i, (out1, out2) in enumerate(zip(outs1, outs2)):
            if isinstance(place, paddle.CUDAPlace):
                np.testing.assert_array_equal(out1, out2)
            else:
                np.testing.assert_allclose(out1, out2, rtol=1e-05, atol=1e-07)

        outs3 = run_op(use_nesterov=False, use_merged=True)
        outs4 = run_op(use_nesterov=False, use_merged=False)
        self.assertEqual(len(outs3), len(outs4))
        for j, (out3, out4) in enumerate(zip(outs3, outs4)):
            if isinstance(place, paddle.CUDAPlace):
                np.testing.assert_array_equal(out3, out4)
            else:
                np.testing.assert_allclose(out3, out4, rtol=1e-05, atol=1e-07)

    def get_places(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            places.append(paddle.CPUPlace())
        if paddle.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        return places

    def test_main(self):
        for multi_precision in [False, True]:
            for place in self.get_places():
                self.check_with_place(place, multi_precision)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()

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

import copy
import random
import unittest

import numpy as np

import paddle
from paddle.base.framework import EagerParamBase
from paddle.distributed.fleet.utils import recompute


class Model(paddle.nn.Layer):
    def __init__(self, block_idx, input_size, is_last=False):
        super().__init__()
        block_name = "block_" + str(block_idx)
        self.block = paddle.nn.Sequential(
            (
                block_name + "_fc_0",
                paddle.nn.Linear(input_size, input_size, bias_attr=False),
            ),
            (block_name + "_dropout", paddle.nn.Dropout(p=0.5)),
            (block_name + "_relu_1", paddle.nn.ReLU()),
            (
                block_name + "_fc_1",
                paddle.nn.Linear(input_size, input_size, bias_attr=False),
            ),
            (block_name + "_relu_2", paddle.nn.ReLU()),
        )
        if is_last:
            self.block.add_sublayer(
                block_name + "_fc_2",
                paddle.nn.Linear(input_size, 1, bias_attr=False),
            )  # add sublayer
        else:
            self.block.add_sublayer(
                block_name + "_fc_2",
                paddle.nn.Linear(input_size, input_size, bias_attr=False),
            )  # add sublayer

    # add pos param for test kwargs of recompute.
    def forward(self, x, pos=None):
        if pos is None:
            return self.block(x)
        else:
            if isinstance(pos, tuple):
                pos = pos[0]
            return self.block(x) + pos


def get_fc_block(block_idx, input_size, is_last=False):
    return Model(block_idx, input_size, is_last=False)


class Naive_fc_net(paddle.nn.Layer):
    def __init__(
        self,
        input_size=10,
        recompute_blocks=[1, 3],
        use_fleet_sq=False,
        segments=1,
        use_raw_recompute=False,
        recompute_kwargs={},
        raise_value_error=False,
        recompute_use_kwargs_as_inputs=False,
    ):
        super().__init__()
        self.recompute_blocks = recompute_blocks
        self.recompute_kwargs = recompute_kwargs
        self.use_fleet_sq = use_fleet_sq
        self.use_raw_recompute = use_raw_recompute
        self.raise_value_error = raise_value_error
        self.segments = segments

        self.runfunc0 = get_fc_block(0, input_size, is_last=False)
        self.runfunc1 = get_fc_block(1, input_size, is_last=False)
        self.runfunc2 = get_fc_block(2, input_size, is_last=False)
        self.runfunc3 = get_fc_block(3, input_size, is_last=False)
        self.runfunc4 = get_fc_block(4, input_size, is_last=True)

        if self.use_fleet_sq and not use_raw_recompute:
            self.runfuncs = paddle.nn.Sequential(
                self.runfunc0,
                self.runfunc1,
                self.runfunc2,
                self.runfunc3,
                self.runfunc4,
            )

        self.layers = [
            self.runfunc0,
            self.runfunc1,
            self.runfunc2,
            self.runfunc3,
            self.runfunc4,
        ]

        # default segments = 2
        if use_raw_recompute:
            self.layers = [
                paddle.nn.Sequential(self.runfunc0, self.runfunc1),
                paddle.nn.Sequential(
                    self.runfunc2, self.runfunc3, self.runfunc4
                ),
            ]
        self.recompute_use_kwargs_as_inputs = recompute_use_kwargs_as_inputs

    def forward(self, inputs):
        if self.use_fleet_sq and not self.use_raw_recompute:
            return paddle.incubate.distributed.fleet.recompute_sequential(
                {"segments": self.segments}, self.runfuncs, inputs
            )

        if self.use_raw_recompute:
            inputs = recompute(self.layers[0], inputs)
            return self.layers[1](inputs)

        recompute_kwargs = copy.deepcopy(self.recompute_kwargs)

        pos = (
            recompute_kwargs.pop("pos", None)
            if not self.raise_value_error
            else None
        )
        for i in range(len(self.layers)):
            if i in self.recompute_blocks:
                if self.recompute_use_kwargs_as_inputs:
                    inputs = recompute(
                        self.layers[i], pos=pos, x=inputs, **recompute_kwargs
                    )
                else:
                    inputs = recompute(
                        self.layers[i], inputs, pos, **recompute_kwargs
                    )
            else:
                inputs = self.layers[i](inputs, pos)

        return inputs


def run_model(
    recompute_block=[],
    recompute_kwargs={},
    raise_value_error=False,
    use_fleet_sq=False,
    use_raw_recompute=False,
    segments=1,
    enable_autocast=False,
    pure_fp16=False,
    recompute_use_kwargs_as_inputs=False,
):
    gen = paddle.seed(10)
    gen.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    batch_size, input_size = 1, 10
    model = Naive_fc_net(
        input_size,
        recompute_blocks=recompute_block,
        use_fleet_sq=use_fleet_sq,
        use_raw_recompute=use_raw_recompute,
        segments=segments,
        recompute_kwargs=recompute_kwargs,
        raise_value_error=raise_value_error,
        recompute_use_kwargs_as_inputs=recompute_use_kwargs_as_inputs,
    )

    if pure_fp16:
        model = paddle.amp.decorate(models=model, level='O2')

    loss_fn = paddle.nn.MSELoss(reduction='mean')
    optimizer = paddle.optimizer.SGD(
        learning_rate=0.01, parameters=model.parameters()
    )

    if enable_autocast:
        scaler = paddle.amp.GradScaler(init_loss_scaling=4096)

    loss_ = []
    param_ = []
    grad_ = []
    for step in range(10):
        x_data = np.random.randn(batch_size, input_size).astype(np.float32)
        x = paddle.to_tensor(x_data)
        x.stop_gradient = False
        level = 'O2' if pure_fp16 else 'O1'
        with paddle.amp.auto_cast(True, level=level):
            y_pred = model(x)
            loss = y_pred.mean()
        if enable_autocast:
            scaler.scale(loss).backward()
            scaler.minimize(optimizer, loss)
        else:
            loss_.append(np.asarray(loss).tolist())
            loss.backward()
            optimizer.step()

        param_.append(np.asarray(model.parameters()[9]).tolist())
        grad_.append(np.asarray(model.parameters()[3]._grad_ivar()).tolist())

        optimizer.clear_grad()
    return loss_, param_, grad_


class TestRecompute(unittest.TestCase):
    def test_base_case(
        self,
        enable_autocast=False,
        pure_fp16=False,
        recompute_use_kwargs_as_inputs=False,
    ):
        def check_identical(loss_ref, param_ref, grad_ref, loss, param, grad):
            self.assertEqual(loss_ref, loss)
            self.assertEqual(param_ref, param)
            self.assertEqual(grad_ref, grad)

        # without recompute
        loss_ref, param_ref, grad_ref = run_model(
            recompute_block=[],
            enable_autocast=enable_autocast,
            pure_fp16=pure_fp16,
        )

        # test for recompute
        # True: PyLayer of recompute
        # False: HooK of recompute
        for flag in [True, False]:
            # recompute second block
            loss, param, grad = run_model(
                recompute_block=[1],
                enable_autocast=enable_autocast,
                pure_fp16=pure_fp16,
                recompute_kwargs={"use_reentrant": flag},
                recompute_use_kwargs_as_inputs=recompute_use_kwargs_as_inputs,
            )
            check_identical(loss_ref, param_ref, grad_ref, loss, param, grad)

            # recompute fourth block
            loss, param, grad = run_model(
                recompute_block=[3],
                enable_autocast=enable_autocast,
                pure_fp16=pure_fp16,
                recompute_kwargs={"use_reentrant": flag},
                recompute_use_kwargs_as_inputs=recompute_use_kwargs_as_inputs,
            )
            check_identical(loss_ref, param_ref, grad_ref, loss, param, grad)

            # recompute second to fourth block
            loss, param, grad = run_model(
                recompute_block=[1, 2, 3],
                enable_autocast=enable_autocast,
                pure_fp16=pure_fp16,
                recompute_kwargs={"use_reentrant": flag},
                recompute_use_kwargs_as_inputs=recompute_use_kwargs_as_inputs,
            )
            check_identical(loss_ref, param_ref, grad_ref, loss, param, grad)

            # recompute second & fourth block
            loss, param, grad = run_model(
                recompute_block=[1, 3],
                enable_autocast=enable_autocast,
                pure_fp16=pure_fp16,
                recompute_kwargs={"use_reentrant": flag},
                recompute_use_kwargs_as_inputs=recompute_use_kwargs_as_inputs,
            )
            check_identical(loss_ref, param_ref, grad_ref, loss, param, grad)

            # recompute_sequential with segments=1 using fleet
            loss, param, grad = run_model(
                recompute_block=[],
                use_fleet_sq=True,
                enable_autocast=enable_autocast,
                pure_fp16=pure_fp16,
                recompute_kwargs={"use_reentrant": flag},
                recompute_use_kwargs_as_inputs=recompute_use_kwargs_as_inputs,
            )
            check_identical(loss_ref, param_ref, grad_ref, loss, param, grad)

        # with base recompute, and segments=2
        loss_ref, param_ref, grad_ref = run_model(
            recompute_block=[],
            enable_autocast=enable_autocast,
            use_raw_recompute=True,
            pure_fp16=pure_fp16,
        )

        # recompute using paddle.incubate.distributed.fleet.recompute_sequential, segments=2
        loss, param, grad = run_model(
            recompute_block=[],
            use_fleet_sq=True,
            segments=2,
            enable_autocast=enable_autocast,
            pure_fp16=pure_fp16,
        )
        check_identical(loss_ref, param_ref, grad_ref, loss, param, grad)

    def test_fc_net_with_dropout(self):
        self.test_base_case()
        self.test_base_case(recompute_use_kwargs_as_inputs=True)

    def test_fc_net_without_restore_rng(self):
        for flag in [True, False]:
            for recompute_use_kwargs_as_inputs in [True, False]:
                loss_ref, param_ref, grad_ref = run_model(
                    recompute_block=[2],
                    recompute_kwargs={
                        "preserve_rng_state": False,
                        "use_reentrant": flag,
                    },
                    enable_autocast=True,
                    recompute_use_kwargs_as_inputs=recompute_use_kwargs_as_inputs,
                )

    def test_fc_net_with_amp(self):
        self.test_base_case(enable_autocast=True)
        self.test_base_case(
            enable_autocast=True, recompute_use_kwargs_as_inputs=True
        )

    def test_fc_net_with_fp16(self):
        self.test_base_case(enable_autocast=True, pure_fp16=True)
        self.test_base_case(
            enable_autocast=True,
            pure_fp16=True,
            recompute_use_kwargs_as_inputs=True,
        )

    def test_recompute_kwargs(self):
        paddle.set_device(
            "xpu" if paddle.base.core.is_compiled_with_xpu() else "gpu"
        )
        pos = paddle.randn(shape=[10, 10], dtype="float32")
        pos.stop_gradient = False

        kwargs = {"pos": pos, "use_reentrant": True}
        with self.assertRaises(TypeError):
            loss_ref, param_ref, grad_ref = run_model(
                recompute_block=[2],
                recompute_kwargs=kwargs,
                raise_value_error=True,
            )

        kwargs = {"pos": pos, "use_reentrant": False}
        loss_ref, param_ref, grad_ref = run_model(
            recompute_block=[2], recompute_kwargs=kwargs
        )

    def test_recompute_inputs_with_param(self):
        for flag in [True, False]:
            for recompute_use_kwargs_as_inputs in [True, False]:
                pos = paddle.randn(shape=[10, 10], dtype="float32")
                new_pos = EagerParamBase(
                    shape=pos.shape, dtype=pos.dtype, name=pos.name
                )
                pos._share_buffer_to(new_pos)
                new_pos.stop_gradient = False

                loss, param, grad = run_model(
                    recompute_block=[2, 4],
                    recompute_kwargs={"pos": new_pos, "use_reentrant": flag},
                    recompute_use_kwargs_as_inputs=recompute_use_kwargs_as_inputs,
                )

                loss_ref, param_ref, grad_ref = run_model(
                    recompute_block=[1, 2, 3],
                    recompute_kwargs={"pos": new_pos, "use_reentrant": flag},
                    recompute_use_kwargs_as_inputs=recompute_use_kwargs_as_inputs,
                )

                self.assertEqual(loss_ref, loss)
                self.assertEqual(param_ref, param)
                self.assertEqual(grad_ref, grad)

    def test_recompute_inputs_with_tuple(self):
        for flag in [True, False]:
            for recompute_use_kwargs_as_inputs in [True, False]:
                pos = paddle.randn(shape=[10, 10], dtype="float32")
                new_pos = EagerParamBase(
                    shape=pos.shape, dtype=pos.dtype, name=pos.name
                )
                pos._share_buffer_to(new_pos)
                pos.stop_gradient = False
                new_pos.stop_gradient = False

                loss, param, grad = run_model(
                    recompute_block=[2, 4],
                    recompute_kwargs={"pos": (pos,), "use_reentrant": flag},
                    recompute_use_kwargs_as_inputs=recompute_use_kwargs_as_inputs,
                )

                loss_ref, param_ref, grad_ref = run_model(
                    recompute_block=[1, 2, 3],
                    recompute_kwargs={"pos": (new_pos,), "use_reentrant": flag},
                    recompute_use_kwargs_as_inputs=recompute_use_kwargs_as_inputs,
                )

                self.assertEqual(loss_ref, loss)
                self.assertEqual(param_ref, param)
                self.assertEqual(grad_ref, grad)


if __name__ == '__main__':
    unittest.main()

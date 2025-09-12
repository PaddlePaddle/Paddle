#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.distributed.fleet.meta_parallel import zero_bubble_utils


class TestZuroBubble(unittest.TestCase):
    def setUp(self):
        paddle.seed(42)

    def test_weight_grad_store(self):
        def fake_func():
            return

        zero_bubble_utils.WeightGradStore.put(fake_func)
        np.testing.assert_equal(
            zero_bubble_utils.WeightGradStore.funcs_queue.empty(), True
        )
        zero_bubble_utils.WeightGradStore.flush()
        np.testing.assert_equal(
            zero_bubble_utils.WeightGradStore.funcs_queue.empty(), False
        )
        zero_bubble_utils.WeightGradStore.pop()
        np.testing.assert_equal(
            zero_bubble_utils.WeightGradStore.funcs_queue.empty(), True
        )

    def test_zero_bubble_utils(self):
        zero_bubble_utils.WeightGradStore.enabled = False

        paddle.seed(42)
        input = paddle.randn([2, 4096, 2048])
        input.stop_gradient = False
        splitbw_linear = zero_bubble_utils.SplitBWLinear(
            2048, 2048, bias_attr=True
        )
        o = splitbw_linear(input)
        o.mean().backward()

        paddle.seed(42)
        ref_input = paddle.randn([2, 4096, 2048])
        ref_input.stop_gradient = False
        ref_linear = paddle.nn.Linear(2048, 2048, bias_attr=True)
        o_ref = ref_linear(ref_input)
        o_ref.mean().backward()

        np.testing.assert_equal(o._md5sum(), o_ref._md5sum())
        np.testing.assert_equal(input.grad._md5sum(), ref_input.grad._md5sum())
        np.testing.assert_equal(
            splitbw_linear.weight.grad._md5sum(),
            ref_linear.weight.grad._md5sum(),
        )
        np.testing.assert_equal(
            splitbw_linear.bias.grad._md5sum(), ref_linear.bias.grad._md5sum()
        )

        zero_bubble_utils.WeightGradStore.enabled = True

        paddle.seed(42)
        input = paddle.randn([2, 4096, 2048])
        input.stop_gradient = False
        splitbw_linear = zero_bubble_utils.SplitBWLinear(
            2048, 2048, bias_attr=True
        )
        o = splitbw_linear(input)
        o.mean().backward()
        np.testing.assert_equal(splitbw_linear.weight.grad, None)
        zero_bubble_utils.WeightGradStore.flush()
        zero_bubble_utils.WeightGradStore.pop()
        np.testing.assert_equal(
            splitbw_linear.weight.grad._md5sum(),
            ref_linear.weight.grad._md5sum(),
        )

    def test_zero_bubble_utils_no_bias(self):
        zero_bubble_utils.WeightGradStore.enabled = True

        paddle.seed(42)
        ref_input = paddle.randn([2, 4096, 2048])
        ref_input.stop_gradient = False
        ref_linear = paddle.nn.Linear(2048, 2048, bias_attr=False)
        o_ref = ref_linear(ref_input)
        o_ref.mean().backward()

        paddle.seed(42)
        input = paddle.randn([2, 4096, 2048])
        input.stop_gradient = False
        splitbw_linear = zero_bubble_utils.SplitBWLinear(
            2048, 2048, bias_attr=False
        )
        o = splitbw_linear(input)
        o.mean().backward()

        zero_bubble_utils.WeightGradStore.flush()
        zero_bubble_utils.WeightGradStore.pop()

        np.testing.assert_equal(o._md5sum(), o_ref._md5sum())
        np.testing.assert_equal(input.grad._md5sum(), ref_input.grad._md5sum())
        np.testing.assert_equal(
            splitbw_linear.weight.grad._md5sum(),
            ref_linear.weight.grad._md5sum(),
        )

    def test_zero_bubble_with_main_grad(self):
        def _update_main_grad_hook(param):
            @paddle.autograd.no_grad()
            def param_hook(tmp_grad):
                if tmp_grad is not None and tmp_grad._is_initialized():
                    if param.main_grad is None:
                        param.main_grad = (
                            paddle.base.framework.core.eager.Tensor(
                                value=tmp_grad.cast(paddle.float32).value(),
                                place=tmp_grad.place,
                                name="main_grad@" + param.name,
                            )
                        )
                    else:
                        # 梯度累加
                        param.main_grad.add_(tmp_grad)

                    tmp_grad._clear_data()

            return param_hook

        zero_bubble_utils.WeightGradStore.enabled = False

        paddle.seed(42)
        ref_input = paddle.randn([2, 4096, 2048])
        ref_input.stop_gradient = False
        ref_linear = paddle.nn.Linear(2048, 2048, bias_attr=False)

        paddle.seed(42)
        input = paddle.randn([2, 4096, 2048])
        input.stop_gradient = False
        splitbw_linear = zero_bubble_utils.SplitBWLinear(
            2048, 2048, bias_attr=False
        )

        for param in splitbw_linear.parameters():
            if not hasattr(param, "main_grad"):
                param.main_grad = None
                param._register_grad_hook(_update_main_grad_hook(param))
        for param in ref_linear.parameters():
            if not hasattr(param, "main_grad"):
                param.main_grad = None
                param._register_grad_hook(_update_main_grad_hook(param))

        o_ref = ref_linear(ref_input)
        o_ref.mean().backward()
        o = splitbw_linear(input)
        o.mean().backward()

        np.testing.assert_equal(o._md5sum(), o_ref._md5sum())
        np.testing.assert_equal(input.grad._md5sum(), ref_input.grad._md5sum())
        np.testing.assert_equal(
            splitbw_linear.weight.main_grad._md5sum(),
            ref_linear.weight.main_grad._md5sum(),
        )


if __name__ == "__main__":
    unittest.main()

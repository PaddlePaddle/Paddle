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
import sys
import unittest

import numpy as np

sys.path.append("../deprecated/legacy_test")
from test_imperative_lod_tensor_to_selected_rows_deprecated import SimpleNet

import paddle
from paddle import base
from paddle.base import core

call_forward_post_hook = False
call_forward_pre_hook = False


def forward_post_hook(layer, input, output):
    global call_forward_post_hook
    call_forward_post_hook = True


def forward_pre_hook(layer, input):
    global call_forward_pre_hook
    call_forward_pre_hook = True


def forward_post_hook1(layer, input, output):
    return output * 2


def forward_pre_hook1(layer, input):
    input_return = (input[0] * 2, input[1])
    return input_return


class Test_Forward_Hook(unittest.TestCase):
    # test forward_pre_hook and forward_post_hook that have return value
    def test_forward_hook_return_value(self):
        seed = 90

        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))

        for place in places:
            with base.dygraph.guard(place):
                paddle.seed(seed)
                base.set_flags({'FLAGS_sort_sum_gradient': True})

                input_word = (
                    np.array(
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8]
                    )
                    .reshape(6, 3)
                    .astype('int64')
                )
                input_word1 = input_word * 2
                input_word = input_word.reshape((-1, 3, 1))
                input_word1 = input_word1.reshape((-1, 3, 1))
                y_data = (
                    np.array(
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                    )
                    .reshape(6, 3)
                    .astype('int64')
                )
                y_data = y_data.reshape((-1, 1))

                input = paddle.to_tensor(input_word)
                input1 = paddle.to_tensor(input_word1)
                y = paddle.to_tensor(y_data)

                simplenet = SimpleNet(
                    hidden_size=20,
                    vocab_size=32,
                    num_steps=3,
                    init_scale=0.1,
                    is_sparse=False,
                    dtype="float32",
                )

                # origin, don't register any hook
                outs_origin = simplenet(input, y)
                outs_origin1 = simplenet(input1, y)

                # register forward_pre_hook
                forward_pre_hook_handle1 = simplenet.register_forward_pre_hook(
                    forward_pre_hook1
                )
                outs_pre_hook = simplenet(input, y)
                np.testing.assert_array_equal(
                    outs_pre_hook.numpy(), outs_origin1.numpy()
                )

                # remove forward_pre_hook
                forward_pre_hook_handle1.remove()
                outs_pre_hook = simplenet(input, y)
                np.testing.assert_array_equal(
                    outs_pre_hook.numpy(), outs_origin.numpy()
                )

                # register forward_posst_hook
                forward_post_hook_handle1 = (
                    simplenet.register_forward_post_hook(forward_post_hook1)
                )
                outs_forward_hook = simplenet(input, y)
                np.testing.assert_array_equal(
                    outs_forward_hook.numpy(), outs_origin.numpy() * 2
                )

                # remove forward_post_hook
                forward_post_hook_handle1.remove()
                outs_forward_hook = simplenet(input, y)
                np.testing.assert_array_equal(
                    outs_forward_hook.numpy(), outs_origin.numpy()
                )

    # test forward_pre_hook and forward_post_hook that don't have return value
    def test_forward_hook(self):
        seed = 90

        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))

        for place in places:
            with base.dygraph.guard(place):
                paddle.seed(seed)
                base.set_flags({'FLAGS_sort_sum_gradient': True})

                global call_forward_post_hook
                global call_forward_pre_hook

                input_word = (
                    np.array(
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8]
                    )
                    .reshape(6, 3)
                    .astype('int64')
                )
                input_word = input_word.reshape((-1, 3, 1))
                y_data = (
                    np.array(
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                    )
                    .reshape(6, 3)
                    .astype('int64')
                )
                y_data = y_data.reshape((-1, 1))

                input = paddle.to_tensor(input_word)
                y = paddle.to_tensor(y_data)

                simplenet = SimpleNet(
                    hidden_size=20,
                    vocab_size=32,
                    num_steps=3,
                    init_scale=0.1,
                    is_sparse=False,
                    dtype="float32",
                )

                # origin, don't register any hook
                outs_origin = simplenet(input, y)
                self.assertFalse(call_forward_post_hook)
                self.assertFalse(call_forward_pre_hook)

                # register forward_post_hook and forward_pre_hook
                forward_post_hook_handle = simplenet.register_forward_post_hook(
                    forward_post_hook
                )
                forward_pre_hook_handle = simplenet.register_forward_pre_hook(
                    forward_pre_hook
                )
                outs_hook = simplenet(input, y)
                self.assertTrue(call_forward_post_hook)
                self.assertTrue(call_forward_pre_hook)

                outs_hook = simplenet(input, y)
                self.assertTrue(call_forward_post_hook)
                self.assertTrue(call_forward_pre_hook)

                # remove forward_post_hook
                forward_post_hook_handle.remove()
                call_forward_post_hook = False
                call_forward_pre_hook = False
                outs_remove_forward_hook = simplenet(input, y)
                self.assertFalse(call_forward_post_hook)
                self.assertTrue(call_forward_pre_hook)

                # remove forward_pre_hook
                forward_pre_hook_handle.remove()
                call_forward_post_hook = False
                call_forward_pre_hook = False
                outs_remove_hook = simplenet(input, y)
                self.assertFalse(call_forward_post_hook)
                self.assertFalse(call_forward_pre_hook)


if __name__ == '__main__':
    unittest.main()

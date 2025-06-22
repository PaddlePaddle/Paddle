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


import os
import unittest

import numpy as np

import paddle


def call_MultiLabelMarginLoss_layer(
    input,
    label,
    reduction='mean',
):
    multi_label_margin_loss = paddle.nn.MultiLabelMarginLoss(
        reduction=reduction
    )
    res = multi_label_margin_loss(
        input=input,
        label=label,
    )
    return res


def call_MultiLabelMarginLoss_functional(
    input,
    label,
    reduction='mean',
):
    res = paddle.nn.functional.multi_label_margin_loss(
        input=input,
        label=label,
        reduction=reduction,
    )
    return res


def test_static(
    place,
    input_np,
    label_np,
    reduction='mean',
    functional=False,
):
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        input = paddle.static.data(
            name='input', shape=input_np.shape, dtype=input_np.dtype
        )
        label = paddle.static.data(
            name='label', shape=label_np.shape, dtype=label_np.dtype
        )
        feed_dict = {
            "input": input_np,
            "label": label_np,
        }

        if functional:
            res = call_MultiLabelMarginLoss_functional(
                input=input,
                label=label,
                reduction=reduction,
            )
        else:
            res = call_MultiLabelMarginLoss_layer(
                input=input,
                label=label,
                reduction=reduction,
            )

        exe = paddle.static.Executor(place)
        static_result = exe.run(prog, feed=feed_dict, fetch_list=[res])
    return static_result[0]


def test_static_data_shape(
    place,
    input_np,
    label_np,
    wrong_label_shape=None,
    functional=False,
):
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        input = paddle.static.data(
            name='input', shape=input_np.shape, dtype=input_np.dtype
        )
        if wrong_label_shape is None:
            label_shape = label_np.shape
        else:
            label_shape = wrong_label_shape
        label = paddle.static.data(
            name='label', shape=label_shape, dtype=label_np.dtype
        )
        feed_dict = {
            "input": input_np,
            "label": label_np,
        }

        if functional:
            res = call_MultiLabelMarginLoss_functional(
                input=input,
                label=label,
            )
        else:
            res = call_MultiLabelMarginLoss_layer(
                input=input,
                label=label,
            )

        exe = paddle.static.Executor(place)
        static_result = exe.run(prog, feed=feed_dict, fetch_list=[res])
    return static_result


def test_dygraph(
    place,
    input,
    label,
    reduction='mean',
    functional=False,
):
    paddle.disable_static()
    input = paddle.to_tensor(input)
    label = paddle.to_tensor(label)

    if functional:
        dy_res = call_MultiLabelMarginLoss_functional(
            input=input,
            label=label,
            reduction=reduction,
        )
    else:
        dy_res = call_MultiLabelMarginLoss_layer(
            input=input,
            label=label,
            reduction=reduction,
        )
    dy_result = dy_res.numpy()
    paddle.enable_static()
    return dy_result


def calc_multi_label_margin_loss(
    input,
    label,
    reduction='mean',
):
    nframe, dim = input.shape
    losses = np.zeros(nframe, dtype=input.dtype)

    for i in range(nframe):
        sample_input = input[i]
        sample_label = label[i]

        valid_label_indices = []
        for j in range(dim):
            if sample_label[j] < 0:
                break
            valid_label_indices.append(sample_label[j])

        if len(valid_label_indices) == 0:
            continue

        is_target = np.zeros(dim, dtype=bool)
        for label_idx in valid_label_indices:
            is_target[label_idx] = True

        sample_loss = 0.0
        for label_idx in valid_label_indices:
            input_target = sample_input[label_idx]

            for d in range(dim):
                if not is_target[d]:
                    margin = 1.0 - input_target + sample_input[d]
                    if margin > 0:
                        sample_loss += margin

        losses[i] = sample_loss / dim

    if reduction == 'mean':
        return np.mean(losses)
    elif reduction == 'sum':
        return np.sum(losses)
    else:
        return losses


class TestMultiLabelMarginLoss(unittest.TestCase):

    def test_MultiLabelMarginLoss(self):
        batch_size = 5
        num_classes = 4
        shape = (batch_size, num_classes)

        # Create test data with multi-label format
        input = np.random.uniform(0.1, 0.8, size=shape).astype(np.float64)

        # Create multi-label targets (2D array with -1 padding)
        label = np.full(shape, -1, dtype=np.int64)
        for i in range(batch_size):
            # Random number of valid labels (0-3)
            num_valid = np.random.randint(0, 4)
            valid_labels = np.random.choice(
                num_classes, size=num_valid, replace=False
            )
            label[i, :num_valid] = valid_labels

        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.device.is_compiled_with_cuda()
        ):
            places.append(paddle.CPUPlace())
        if paddle.device.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))

        reductions = ['sum', 'mean', 'none']
        for place in places:
            for reduction in reductions:
                expected = calc_multi_label_margin_loss(
                    input=input, label=label, reduction=reduction
                )

                dy_result = test_dygraph(
                    place=place,
                    input=input,
                    label=label,
                    reduction=reduction,
                )

                static_result = test_static(
                    place=place,
                    input_np=input,
                    label_np=label,
                    reduction=reduction,
                )
                np.testing.assert_allclose(static_result, expected, rtol=1e-5)
                np.testing.assert_allclose(static_result, dy_result, rtol=1e-5)
                np.testing.assert_allclose(dy_result, expected, rtol=1e-5)

                static_functional = test_static(
                    place=place,
                    input_np=input,
                    label_np=label,
                    reduction=reduction,
                    functional=True,
                )
                dy_functional = test_dygraph(
                    place=place,
                    input=input,
                    label=label,
                    reduction=reduction,
                    functional=True,
                )
                np.testing.assert_allclose(
                    static_functional, expected, rtol=1e-5
                )
                np.testing.assert_allclose(
                    static_functional, dy_functional, rtol=1e-5
                )
                np.testing.assert_allclose(dy_functional, expected, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

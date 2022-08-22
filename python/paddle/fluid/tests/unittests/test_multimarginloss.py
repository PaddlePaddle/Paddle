# -*- coding: utf-8 -*
#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numpy as np
import unittest


def call_MultiMarginLoss_layer(
    input,
    label,
    p=1,
    margin=1.0,
    weight=None,
    reduction='mean',
):
    triplet_margin_loss = paddle.nn.MultiMarginLoss(p=p,
                                                    margin=margin,
                                                    weight=weight,
                                                    reduction=reduction)
    res = triplet_margin_loss(
        input=input,
        label=label,
    )
    return res


def call_MultiMarginLoss_functional(
    input,
    label,
    p=1,
    margin=1.0,
    weight=None,
    reduction='mean',
):
    res = paddle.nn.functional.multi_margin_loss(input=input,
                                                 label=label,
                                                 p=p,
                                                 margin=margin,
                                                 weight=weight,
                                                 reduction=reduction)
    return res


def test_static(place,
                input_np,
                label_np,
                p=1,
                margin=1.0,
                weight=None,
                reduction='mean',
                functional=False):
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        input = paddle.static.data(name='input',
                                   shape=input_np.shape,
                                   dtype='float64')
        label = paddle.static.data(name='label',
                                   shape=label_np.shape,
                                   dtype='float64')
        feed_dict = {
            "input": input_np,
            "label": label_np,
        }

        if functional:
            res = call_MultiMarginLoss_functional(input=input,
                                                  label=label,
                                                  p=p,
                                                  margin=margin,
                                                  weight=weight,
                                                  reduction=reduction)
        else:
            res = call_MultiMarginLoss_layer(input=input,
                                             label=label,
                                             p=p,
                                             margin=margin,
                                             weight=weight,
                                             reduction=reduction)

        exe = paddle.static.Executor(place)
        static_result = exe.run(prog, feed=feed_dict, fetch_list=[res])
    return static_result


def test_dygraph(place,
                 input,
                 label,
                 p=1,
                 margin=1.0,
                 weight=None,
                 reduction='mean',
                 functional=False):
    paddle.disable_static()
    input = paddle.to_tensor(input)
    label = paddle.to_tensor(label)

    if functional:
        dy_res = call_MultiMarginLoss_functional(input=input,
                                                 label=label,
                                                 p=p,
                                                 margin=margin,
                                                 weight=weight,
                                                 reduction=reduction)
    else:
        dy_res = call_MultiMarginLoss_layer(input=input,
                                            label=label,
                                            p=p,
                                            margin=margin,
                                            weight=weight,
                                            reduction=reduction)
    dy_result = dy_res.numpy()
    paddle.enable_static()
    return dy_result


def calc_multi_margin_loss(
    input,
    label,
    p=1,
    margin=1.0,
    weight=None,
    reduction='mean',
):
    label = label.reshape(-1, 1)
    index_sample = []
    for i in range(len(label)):
        index_sample.append(input[i, label[i]])
    index_sample = np.array(index_sample).reshape(-1, 1)

    if weight is None:
        expected = np.mean(np.maximum(margin + input - index_sample, 0.0)**p,
                           axis=1) - margin**p / input.shape[1]
    else:
        weight = weight.reshape(-1, 1)
        expected = np.mean(np.maximum(weight * (margin + input - index_sample), 0.0) ** p, axis=1) - margin ** p / \
               input.shape[1]

    if reduction == 'mean':
        expected = np.mean(expected)
    elif reduction == 'sum':
        expected = np.sum(expected)
    else:
        expected = expected

    return expected


class TestMultiMarginLoss(unittest.TestCase):

    def test_MultiMarginLoss(self):
        shape = (2, 2)
        input = np.random.uniform(0.1, 0.8, size=shape).astype(np.float64)
        label = np.random.uniform(0, 2, size=(2, )).astype(np.float64)

        places = [paddle.CPUPlace()]
        if paddle.device.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        reductions = ['sum', 'mean', 'none']
        for place in places:
            for reduction in reductions:
                expected = calc_multi_margin_loss(input=input,
                                                  label=label,
                                                  reduction=reduction)

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
                self.assertTrue(np.allclose(static_result, expected))
                self.assertTrue(np.allclose(static_result, dy_result))
                self.assertTrue(np.allclose(dy_result, expected))
                static_functional = test_static(place=place,
                                                input_np=input,
                                                label_np=label,
                                                reduction=reduction,
                                                functional=True)
                dy_functional = test_dygraph(place=place,
                                             input=input,
                                             label=label,
                                             reduction=reduction,
                                             functional=True)
                self.assertTrue(np.allclose(static_functional, expected))
                self.assertTrue(np.allclose(static_functional, dy_functional))
                self.assertTrue(np.allclose(dy_functional, expected))

    def test_MultiMarginLoss_error(self):
        paddle.disable_static()
        self.assertRaises(ValueError,
                          paddle.nn.loss.MultiMarginLoss,
                          reduction="unsupport reduction")
        input = paddle.to_tensor([[0.1, 0.3]], dtype='float32')
        label = paddle.to_tensor([0.0], dtype='float32')
        self.assertRaises(ValueError,
                          paddle.nn.functional.multi_margin_loss,
                          input=input,
                          label=label,
                          reduction="unsupport reduction")
        paddle.enable_static()

    def test_MultiMarginLoss_dimension(self):
        paddle.disable_static()

        input = paddle.to_tensor([[0.1, 0.3], [1, 2]], dtype='float32')
        label = paddle.to_tensor([0.0, 1.0, 2.0], dtype='float32')

        self.assertRaises(
            ValueError,
            paddle.nn.functional.multi_margin_loss,
            input=input,
            label=label,
        )
        MMLoss = paddle.nn.loss.MultiMarginLoss()
        self.assertRaises(
            ValueError,
            MMLoss,
            input=input,
            label=label,
        )
        paddle.enable_static()

    def test_MultiMarginLoss_p(self):
        p = 2
        shape = (2, 2)
        reduction = 'mean'
        place = paddle.CPUPlace()
        input = np.random.uniform(0.1, 0.8, size=shape).astype(np.float64)
        label = np.random.uniform(0, 2, size=(2, )).astype(np.float64)
        expected = calc_multi_margin_loss(input=input,
                                          p=p,
                                          label=label,
                                          reduction=reduction)

        dy_result = test_dygraph(
            place=place,
            p=p,
            input=input,
            label=label,
            reduction=reduction,
        )

        static_result = test_static(
            place=place,
            p=p,
            input_np=input,
            label_np=label,
            reduction=reduction,
        )
        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))
        static_functional = test_static(place=place,
                                        p=p,
                                        input_np=input,
                                        label_np=label,
                                        reduction=reduction,
                                        functional=True)
        dy_functional = test_dygraph(place=place,
                                     p=p,
                                     input=input,
                                     label=label,
                                     reduction=reduction,
                                     functional=True)
        self.assertTrue(np.allclose(static_functional, expected))
        self.assertTrue(np.allclose(static_functional, dy_functional))
        self.assertTrue(np.allclose(dy_functional, expected))

    def test_MultiMarginLoss_weight(self):
        shape = (2, 2)
        reduction = 'mean'
        place = paddle.CPUPlace()
        input = np.random.uniform(0.1, 0.8, size=shape).astype(np.float64)
        label = np.random.uniform(0, 2, size=(2, )).astype(np.float64)
        weight = np.random.uniform(0, 2, size=(2, )).astype(np.float64)
        expected = calc_multi_margin_loss(input=input,
                                          label=label,
                                          weight=weight,
                                          reduction=reduction)

        dy_result = test_dygraph(
            place=place,
            input=input,
            label=label,
            weight=weight,
            reduction=reduction,
        )

        static_result = test_static(
            place=place,
            input_np=input,
            label_np=label,
            weight=weight,
            reduction=reduction,
        )
        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))
        static_functional = test_static(place=place,
                                        input_np=input,
                                        label_np=label,
                                        weight=weight,
                                        reduction=reduction,
                                        functional=True)
        dy_functional = test_dygraph(place=place,
                                     input=input,
                                     label=label,
                                     weight=weight,
                                     reduction=reduction,
                                     functional=True)
        self.assertTrue(np.allclose(static_functional, expected))
        self.assertTrue(np.allclose(static_functional, dy_functional))
        self.assertTrue(np.allclose(dy_functional, expected))


if __name__ == "__main__":
    unittest.main()

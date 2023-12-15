# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

paddle.seed(99999)


# soft label + not use softmax + axis_dim = 1
def test0():
    axis = -1
    shape = [4, 1]
    reduction = 'mean'
    weight = None
    logits = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
    logits_out = paddle.nn.functional.softmax(logits, axis=axis)
    labels = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
    labels /= paddle.sum(labels, axis=axis, keepdim=True)
    paddle_loss_mean = paddle.nn.functional.cross_entropy(
        logits_out,
        labels,
        soft_label=True,
        axis=axis,
        use_softmax=False,
        weight=weight,
        reduction=reduction,
    )


# soft label + not use softmax + axis_dim != 1
def test1():
    axis = -1
    shape = [4, 6]
    reduction = 'mean'
    weight = None
    logits = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
    logits_out = paddle.nn.functional.softmax(logits, axis=axis)
    labels = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
    labels /= paddle.sum(labels, axis=axis, keepdim=True)
    paddle_loss_mean = paddle.nn.functional.cross_entropy(
        logits_out,
        labels,
        soft_label=True,
        axis=axis,
        use_softmax=False,
        weight=weight,
        reduction=reduction,
    )


# soft label + use softmax + axis_dim = 1
def test2():
    axis = -1
    shape = [4, 1]
    reduction = 'mean'
    weight = None
    logits = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
    labels = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
    labels /= paddle.sum(labels, axis=axis, keepdim=True)
    paddle_loss_mean = paddle.nn.functional.cross_entropy(
        logits,
        labels,
        soft_label=True,
        axis=axis,
        use_softmax=True,
        weight=weight,
        reduction=reduction,
    )


# soft label + use softmax + axis_dim != 1
def test3():
    axis = -1
    shape = [4, 6]
    reduction = 'mean'
    weight = None
    logits = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
    labels = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
    labels /= paddle.sum(labels, axis=axis, keepdim=True)
    paddle_loss_mean = paddle.nn.functional.cross_entropy(
        logits,
        labels,
        soft_label=True,
        axis=axis,
        use_softmax=True,
        weight=weight,
        reduction=reduction,
    )


def test4():
    axis = 0
    shape = [4, 6]
    reduction = 'mean'
    weight = None
    logits = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
    labels = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
    labels /= paddle.sum(labels, axis=axis, keepdim=True)
    paddle_loss_mean = paddle.nn.functional.cross_entropy(
        logits,
        labels,
        soft_label=True,
        axis=axis,
        use_softmax=True,
        weight=weight,
        reduction=reduction,
    )


def test5():
    # hard labels
    axis = -1
    N = 100
    C = 1
    reduction = 'mean'
    input = paddle.rand([N, C], dtype='float64')
    logits = paddle.nn.functional.softmax(input, axis=axis)
    label = paddle.randint(0, C, shape=[N], dtype='int64')
    weight = paddle.rand([C], dtype='float64')

    paddle_loss_mean = paddle.nn.functional.cross_entropy(
        logits,
        label,
        soft_label=False,
        axis=axis,
        use_softmax=False,
        weight=weight,
        reduction=reduction,
    )


def test6():
    # hard labels
    axis = -1
    N = 100
    C = 200
    reduction = 'mean'
    input = paddle.rand([N, C], dtype='float64')
    logits = paddle.nn.functional.softmax(input, axis=axis)
    label = paddle.randint(0, C, shape=[N], dtype='int64')
    weight = paddle.rand([C], dtype='float64')

    paddle_loss_mean = paddle.nn.functional.cross_entropy(
        logits,
        label,
        soft_label=False,
        axis=axis,
        use_softmax=False,
        weight=weight,
        reduction=reduction,
    )


def test7():
    # hard labels
    axis = -1
    N = 100
    C = 1
    reduction = 'mean'
    input = paddle.rand([N, C], dtype='float64')
    label = paddle.randint(0, C, shape=[N], dtype='int64')
    weight = paddle.rand([C], dtype='float64')

    paddle_loss_mean = paddle.nn.functional.cross_entropy(
        input,
        label,
        soft_label=False,
        axis=axis,
        use_softmax=True,
        weight=weight,
        reduction=reduction,
    )


def test8():
    # hard labels
    axis = -1
    N = 100
    C = 200
    reduction = 'mean'
    input = paddle.rand([N, C], dtype='float64')
    label = paddle.randint(0, C, shape=[N], dtype='int64')
    weight = paddle.rand([C], dtype='float64')

    paddle_loss_mean = paddle.nn.functional.cross_entropy(
        input,
        label,
        soft_label=False,
        axis=axis,
        use_softmax=True,
        weight=weight,
        reduction=reduction,
    )


def test9():
    # hard labels
    axis = -1
    N = 100
    C = 400
    reduction = 'mean'
    input = paddle.rand([N, C], dtype='float64')
    label = paddle.randint(0, C, shape=[N], dtype='int64')
    weight = paddle.rand([C], dtype='float64')

    paddle_loss_mean = paddle.nn.functional.cross_entropy(
        input,
        label,
        soft_label=False,
        axis=axis,
        use_softmax=True,
        weight=weight,
        reduction=reduction,
    )


# def test1():
#     axis = -1
#     N = 4
#     C = 3
#     shape = [N, C]
#     reduction='mean'
#     weight = None
#     logits = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
#     labels = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
#     labels /= paddle.sum(labels, axis=axis, keepdim=True)
#     paddle_loss_mean = paddle.nn.functional.cross_entropy(
#                                                             logits,
#                                                             labels,
#                                                             soft_label=True,
#                                                             axis=axis,
#                                                             weight=weight,
#                                                             reduction=reduction)

# def test2():
#     axis = -1
#     # N = 4
#     # C = 3
#     # shape = [N, C]
#     shape = [5, 4, 1]
#     reduction='mean'
#     weight = None
#     logits = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
#     logits_out = paddle.nn.functional.softmax(logits, axis=axis)
#     labels = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
#     labels /= paddle.sum(labels, axis=axis, keepdim=True)
#     paddle_loss_mean = paddle.nn.functional.cross_entropy(
#                                                             logits_out,
#                                                             labels,
#                                                             soft_label=True,
#                                                             axis=axis,
#                                                             use_softmax=False,
#                                                             weight=weight,
#                                                             reduction=reduction)

# test0()
# test1()
# test2()
# test3()
# test4()
# test5()
# test6()
# test7()
# test8()
test9()

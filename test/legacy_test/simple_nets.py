# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from functools import reduce

import numpy as np

import paddle
from paddle import base
from paddle.common_ops_import import LayerHelper


def pir_fc(hidden, size, activation, param_attr, bias_attr):
    helper = LayerHelper("fc", **locals())
    if not isinstance(hidden, (list, tuple)):
        hidden = [hidden]
    matmul_results = []
    for i, input in enumerate(hidden):
        input_shape = input.shape
        num_flatten_dims = len(input_shape) - 1
        param_shape = [
            reduce(lambda a, b: a * b, input_shape[num_flatten_dims:], 1),
            size,
        ]

        w = helper.create_parameter(
            attr=param_attr, shape=param_shape, dtype=input.dtype, is_bias=False
        )
        out = paddle.matmul(input, w)
        matmul_results.append(out)

    if len(matmul_results) == 1:
        pre_bias = matmul_results[0]
    else:
        pre_bias = paddle.add_n(matmul_results)
    bias = helper.create_parameter(
        attr=bias_attr,
        shape=pre_bias.shape[-1:],
        dtype=pre_bias.dtype,
        is_bias=True,
    )
    out = paddle.add(pre_bias, bias)
    act_op = getattr(paddle._C_ops, activation)
    if activation == 'softmax':
        return act_op(out, -1)
    return act_op(out)


def pir_simple_fc_net_with_inputs(img, label, class_num=10):
    hidden = img
    param_attr = base.ParamAttr(initializer=paddle.nn.initializer.Uniform())
    bias_attr = base.ParamAttr(
        initializer=paddle.nn.initializer.Constant(value=1.0)
    )
    for _ in range(2):
        hidden = pir_fc(
            hidden,
            size=100,
            activation='relu',
            param_attr=param_attr,
            bias_attr=bias_attr,
        )
    prediction = pir_fc(
        hidden,
        size=class_num,
        activation='softmax',
        param_attr=param_attr,
        bias_attr=bias_attr,
    )
    loss = paddle.nn.functional.softmax_with_cross_entropy(prediction, label)

    loss = paddle.mean(loss)
    return loss


def pir_batchnorm_fc_with_inputs(img, label, class_num=10):
    hidden = img
    param_attr = base.ParamAttr(initializer=paddle.nn.initializer.Uniform())
    bias_attr = base.ParamAttr(
        initializer=paddle.nn.initializer.Constant(value=1.0)
    )
    for _ in range(2):
        hidden = pir_fc(
            hidden,
            size=200,
            activation='relu',
            param_attr=param_attr,
            bias_attr=bias_attr,
        )
        batch_norm = paddle.nn.BatchNorm(200)
        hidden = batch_norm(hidden)

    prediction = pir_fc(
        hidden,
        size=class_num,
        activation='softmax',
        param_attr=param_attr,
        bias_attr=bias_attr,
    )
    loss = paddle.nn.functional.softmax_with_cross_entropy(prediction, label)
    loss = paddle.mean(loss)
    return loss


def simple_fc_net_with_inputs(img, label, class_num=10):
    if paddle.framework.in_pir_mode():
        return pir_simple_fc_net_with_inputs(img, label, class_num)

    hidden = img
    for _ in range(2):
        hidden = paddle.static.nn.fc(
            hidden,
            size=100,
            activation='relu',
            bias_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)
            ),
        )
    prediction = paddle.static.nn.fc(
        hidden, size=class_num, activation='softmax'
    )
    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    loss = paddle.mean(loss)
    return loss


def simple_fc_net(use_feed=None):
    img = paddle.static.data(name='image', shape=[-1, 784], dtype='float32')
    label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
    return simple_fc_net_with_inputs(img, label, class_num=10)


def batchnorm_fc_with_inputs(img, label, class_num=10):
    if paddle.framework.in_pir_mode():
        return pir_batchnorm_fc_with_inputs(img, label, class_num)

    hidden = img
    for _ in range(2):
        hidden = paddle.static.nn.fc(
            hidden,
            size=200,
            activation='relu',
            bias_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)
            ),
        )

        hidden = paddle.static.nn.batch_norm(input=hidden)

    prediction = paddle.static.nn.fc(
        hidden, size=class_num, activation='softmax'
    )
    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    loss = paddle.mean(loss)
    return loss


def fc_with_batchnorm(use_feed=None):
    img = paddle.static.data(name='image', shape=[-1, 784], dtype='float32')
    label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
    return batchnorm_fc_with_inputs(img, label, class_num=10)


def bow_net(
    use_feed,
    dict_dim,
    is_sparse=False,
    emb_dim=128,
    hid_dim=128,
    hid_dim2=96,
    class_dim=2,
):
    """
    BOW net
    This model is from https://github.com/PaddlePaddle/models:
    base/PaddleNLP/text_classification/nets.py
    """
    data = paddle.static.data(name="words", shape=[-1, 1], dtype="int64")
    label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64")
    emb = paddle.static.nn.embedding(
        input=data, is_sparse=is_sparse, size=[dict_dim, emb_dim]
    )
    bow = paddle.static.nn.sequence_lod.sequence_pool(
        input=emb, pool_type='sum'
    )
    bow_tanh = paddle.tanh(bow)
    fc_1 = paddle.static.nn.fc(x=bow_tanh, size=hid_dim, activation="tanh")
    fc_2 = paddle.static.nn.fc(x=fc_1, size=hid_dim2, activation="tanh")
    prediction = paddle.static.nn.fc(
        x=[fc_2], size=class_dim, activation="softmax"
    )
    cost = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    avg_cost = paddle.mean(x=cost)

    return avg_cost


def init_data(batch_size=32, img_shape=[784], label_range=9):
    np.random.seed(5)
    assert isinstance(img_shape, list)
    input_shape = [batch_size, *img_shape]
    img = np.random.random(size=input_shape).astype(np.float32)
    label = (
        np.array([np.random.randint(0, label_range) for _ in range(batch_size)])
        .reshape((-1, 1))
        .astype("int64")
    )
    return img, label

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

import numpy as np

import paddle
import paddle.fluid as fluid


def simple_fc_net_with_inputs(img, label, class_num=10):
    hidden = img
    for _ in range(2):
        hidden = fluid.layers.fc(
            hidden,
            size=100,
            act='relu',
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=1.0)
            ),
        )
    prediction = fluid.layers.fc(hidden, size=class_num, act='softmax')
    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    loss = paddle.mean(loss)
    return loss


def simple_fc_net(use_feed=None):
    img = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    return simple_fc_net_with_inputs(img, label, class_num=10)


def batchnorm_fc_with_inputs(img, label, class_num=10):
    hidden = img
    for _ in range(2):
        hidden = fluid.layers.fc(
            hidden,
            size=200,
            act='relu',
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=1.0)
            ),
        )

        hidden = paddle.static.nn.batch_norm(input=hidden)

    prediction = fluid.layers.fc(hidden, size=class_num, act='softmax')
    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    loss = paddle.mean(loss)
    return loss


def fc_with_batchnorm(use_feed=None):
    img = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
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
    fluid/PaddleNLP/text_classification/nets.py
    """
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1
    )
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    emb = fluid.layers.embedding(
        input=data, is_sparse=is_sparse, size=[dict_dim, emb_dim]
    )
    bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    bow_tanh = paddle.tanh(bow)
    fc_1 = fluid.layers.fc(input=bow_tanh, size=hid_dim, act="tanh")
    fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim2, act="tanh")
    prediction = fluid.layers.fc(input=[fc_2], size=class_dim, act="softmax")
    cost = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    avg_cost = paddle.mean(x=cost)

    return avg_cost


def init_data(batch_size=32, img_shape=[784], label_range=9):
    np.random.seed(5)
    assert isinstance(img_shape, list)
    input_shape = [batch_size] + img_shape
    img = np.random.random(size=input_shape).astype(np.float32)
    label = (
        np.array([np.random.randint(0, label_range) for _ in range(batch_size)])
        .reshape((-1, 1))
        .astype("int64")
    )
    return img, label

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

import os
import unittest

import numpy as np
from test_imperative_base import new_program_scope
from utils import DyGraphProgramDescTracerTestHelper

import paddle
from paddle import base
from paddle.base import core


class SimpleNet(paddle.nn.Layer):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        num_steps=20,
        init_scale=0.1,
        is_sparse=False,
        dtype='float32',
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.init_scale = init_scale
        self.num_steps = num_steps
        paddle.set_default_dtype(dtype)
        self.embedding = paddle.nn.Embedding(
            vocab_size,
            hidden_size,
            sparse=is_sparse,
            weight_attr=base.ParamAttr(
                name='embedding_para',
                initializer=paddle.nn.initializer.Uniform(
                    low=-init_scale, high=init_scale
                ),
            ),
        )
        self.softmax_bias = self.create_parameter(
            attr=base.ParamAttr(),
            shape=[self.vocab_size],
            dtype=dtype,
            default_initializer=paddle.nn.initializer.Uniform(
                low=-self.init_scale, high=self.init_scale
            ),
        )

    def forward(self, input, label):
        x_emb = self.embedding(input)
        projection = paddle.matmul(
            x_emb, paddle.transpose(self.embedding.weight, perm=[1, 0])
        )
        projection = paddle.add(projection, self.softmax_bias)
        projection = paddle.reshape(projection, shape=[-1, self.vocab_size])
        loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False
        )
        loss = paddle.reshape(loss, shape=[-1, self.num_steps])
        loss = paddle.mean(loss, axis=[0])
        loss = paddle.sum(loss)

        return loss


class TestDygraphSimpleNet(unittest.TestCase):
    def test_simple_net(self):
        for is_sparse in [True, False]:
            dtype_list = ["float32"]
            if not core.is_compiled_with_rocm():
                dtype_list.append("float64")
            for dtype in dtype_list:
                self.simple_net_float32(is_sparse, dtype)

    def simple_net_float32(self, is_sparse, dtype):
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
            seed = 90
            hidden_size = 10
            vocab_size = 1000
            num_steps = 3
            init_scale = 0.1
            batch_size = 4
            batch_num = 200

            for is_sort_sum_gradient in [True, False]:
                with base.dygraph.guard(place):
                    paddle.seed(seed)
                    paddle.framework.random._manual_program_seed(seed)

                    simple_net = SimpleNet(
                        hidden_size=hidden_size,
                        vocab_size=vocab_size,
                        num_steps=num_steps,
                        init_scale=init_scale,
                        is_sparse=is_sparse,
                        dtype=dtype,
                    )

                    sgd = paddle.optimizer.SGD(
                        learning_rate=1e-3,
                        parameters=simple_net.parameters(),
                    )
                    dy_param_updated = {}
                    dy_param_init = {}
                    dy_loss = None

                    helper = DyGraphProgramDescTracerTestHelper(self)
                    base.set_flags(
                        {'FLAGS_sort_sum_gradient': is_sort_sum_gradient}
                    )

                    for i in range(batch_num):
                        x_data = np.arange(12).reshape(4, 3).astype('int64')
                        y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                        x_data = x_data.reshape((-1, num_steps))
                        y_data = y_data.reshape((-1, 1))

                        x = paddle.to_tensor(x_data)
                        y = paddle.to_tensor(y_data)
                        outs = simple_net(x, y)
                        dy_loss = outs
                        if i == 0:
                            for param in simple_net.parameters():
                                dy_param_init[param.name] = param.numpy()
                        dy_loss.backward()
                        sgd.minimize(dy_loss)
                        sgd.clear_gradients()
                        if i == batch_num - 1:
                            for param in simple_net.parameters():
                                dy_param_updated[param.name] = param.numpy()
                    dy_loss_value = dy_loss.numpy()

                with new_program_scope():
                    paddle.seed(seed)
                    paddle.framework.random._manual_program_seed(seed)

                    simple_net = SimpleNet(
                        hidden_size=hidden_size,
                        vocab_size=vocab_size,
                        num_steps=num_steps,
                        is_sparse=is_sparse,
                        dtype=dtype,
                    )

                    exe = base.Executor(place)
                    sgd = paddle.optimizer.SGD(learning_rate=1e-3)
                    x = paddle.static.data(
                        name="x", shape=[-1, num_steps], dtype='int64'
                    )
                    x.desc.set_need_check_feed(False)
                    y = paddle.static.data(name="y", shape=[-1, 1], dtype=dtype)
                    y.desc.set_need_check_feed(False)
                    static_loss = simple_net(x, y)
                    sgd.minimize(static_loss)
                    static_param_updated = {}
                    static_param_init = {}
                    static_param_name_list = []
                    for param in simple_net.parameters():
                        static_param_name_list.append(param.name)

                    out = exe.run(
                        base.default_startup_program(),
                        fetch_list=static_param_name_list,
                    )
                    for i in range(len(static_param_name_list)):
                        static_param_init[static_param_name_list[i]] = out[i]
                    static_loss_value = None
                    for i in range(batch_num):
                        x_data = np.arange(12).reshape(4, 3).astype('int64')
                        y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                        x_data = x_data.reshape((-1, num_steps))
                        y_data = y_data.reshape((-1, 1))
                        fetch_list = [static_loss]
                        fetch_list.extend(static_param_name_list)
                        out = exe.run(
                            base.default_main_program(),
                            feed={"x": x_data, "y": y_data},
                            fetch_list=fetch_list,
                        )
                        static_loss_value = out[0]

                        if i == batch_num - 1:
                            for k in range(3, len(out)):
                                static_param_updated[
                                    static_param_name_list[k - 1]
                                ] = out[k]

                np.testing.assert_allclose(
                    static_loss_value, dy_loss_value, rtol=0.001
                )
                for key, value in static_param_init.items():
                    np.testing.assert_array_equal(value, dy_param_init[key])
                for key, value in static_param_updated.items():
                    np.testing.assert_array_equal(value, dy_param_updated[key])


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()

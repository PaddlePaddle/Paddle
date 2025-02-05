#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
import inspect
import sys
import unittest

sys.path.append("../../legacy_test")
import numpy as np
from test_imperative_base import new_program_scope

import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core, dygraph
from paddle.tensor import random


class LayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = 111

    @classmethod
    def tearDownClass(cls):
        pass

    def _get_place(self, force_to_use_cpu=False):
        # this option for ops that only have cpu kernel
        if force_to_use_cpu:
            return core.CPUPlace()
        else:
            if core.is_compiled_with_cuda():
                return core.CUDAPlace(0)
            return core.CPUPlace()

    @contextlib.contextmanager
    def static_graph(self):
        paddle.seed(self.seed)
        if paddle.framework.use_pir_api():
            with paddle.pir_utils.OldIrGuard():
                paddle.framework.random._manual_program_seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
        else:
            paddle.framework.random._manual_program_seed(self.seed)
        with new_program_scope():
            yield

    def get_static_graph_result(
        self, feed, fetch_list, with_lod=False, force_to_use_cpu=False
    ):
        exe = base.Executor(self._get_place(force_to_use_cpu))
        exe.run(paddle.static.default_startup_program())
        return exe.run(
            paddle.static.default_main_program(),
            feed=feed,
            fetch_list=fetch_list,
            return_numpy=(not with_lod),
        )

    @contextlib.contextmanager
    def dynamic_graph(self, force_to_use_cpu=False):
        paddle.seed(self.seed)
        if paddle.framework.use_pir_api():
            with paddle.pir_utils.OldIrGuard():
                paddle.framework.random._manual_program_seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
        else:
            paddle.framework.random._manual_program_seed(self.seed)
        with base.dygraph.guard(
            self._get_place(force_to_use_cpu=force_to_use_cpu)
        ):
            yield


class TestLayer(LayerTest):
    def test_custom_layer_with_kwargs(self):
        class CustomLayer(paddle.nn.Layer):
            def __init__(self, input_size, linear1_size=4):
                super().__init__()
                self.linear1 = paddle.nn.Linear(
                    input_size, linear1_size, bias_attr=False
                )
                self.linear2 = paddle.nn.Linear(
                    linear1_size, 1, bias_attr=False
                )

            def forward(self, x, do_linear2=False):
                ret = self.linear1(x)
                if do_linear2:
                    ret = self.linear2(ret)
                return ret

        with self.dynamic_graph():
            inp = np.ones([3, 3], dtype='float32')
            x = paddle.to_tensor(inp)
            custom = CustomLayer(input_size=3, linear1_size=2)
            ret = custom(x, do_linear2=False)
            np.testing.assert_array_equal(ret.numpy().shape, [3, 2])
            ret = custom(x, do_linear2=True)
            np.testing.assert_array_equal(ret.numpy().shape, [3, 1])

    def test_dropout(self):
        inp = np.ones([3, 32, 32], dtype='float32')
        with self.static_graph():
            t = paddle.static.data(
                name='data',
                shape=[3, 32, 32],
                dtype='float32',
            )
            dropout = paddle.nn.Dropout(p=0.35)
            ret = dropout(t)
            ret2 = paddle.nn.functional.dropout(t, p=0.35)
            static_ret, static_ret2 = self.get_static_graph_result(
                feed={'data': inp}, fetch_list=[ret, ret2]
            )
        with self.dynamic_graph():
            t = paddle.to_tensor(inp)
            dropout = paddle.nn.Dropout(p=0.35)
            dy_ret = dropout(t)
            dy_ret2 = paddle.nn.functional.dropout(t, p=0.35)
            dy_ret_value = dy_ret.numpy()
            dy_ret2_value = dy_ret2.numpy()

        np.testing.assert_array_equal(static_ret, static_ret2)
        np.testing.assert_array_equal(dy_ret_value, dy_ret2_value)
        np.testing.assert_array_equal(static_ret, dy_ret_value)

    def test_linear(self):
        inp = np.ones([3, 32, 32], dtype='float32')
        with self.static_graph():
            t = paddle.static.data(
                name='data', shape=[3, 32, 32], dtype='float32'
            )
            linear = paddle.nn.Linear(
                32,
                4,
                bias_attr=paddle.nn.initializer.Constant(value=1),
            )
            ret = linear(t)
            static_ret = self.get_static_graph_result(
                feed={'data': inp}, fetch_list=[ret]
            )[0]
        with self.dynamic_graph():
            t = paddle.to_tensor(inp)
            linear = paddle.nn.Linear(
                32,
                4,
                bias_attr=paddle.nn.initializer.Constant(value=1),
            )
            dy_ret = linear(t)
            dy_ret_value = dy_ret.numpy()

        np.testing.assert_array_equal(static_ret, dy_ret_value)

        with self.static_graph():
            # the input of Linear must be Variable.
            def test_Variable():
                inp = np.ones([3, 32, 32], dtype='float32')
                linear = paddle.nn.Linear(
                    32,
                    4,
                    bias_attr=paddle.nn.initializer.Constant(value=1),
                )
                linear_ret1 = linear(inp)

            self.assertRaises(TypeError, test_Variable)

            # the input dtype of Linear must be float16 or float32 or float64
            # float16 only can be set on GPU place
            def test_type():
                inp = np.ones([3, 32, 32], dtype='int32')
                linear = paddle.nn.Linear(
                    32,
                    4,
                    bias_attr=paddle.nn.initializer.Constant(value=1),
                )
                linear_ret2 = linear(inp)

            self.assertRaises(TypeError, test_type)

    def test_Flatten(self):
        inp = np.ones([3, 4, 4, 5], dtype='float32')
        with self.static_graph():
            t = paddle.static.data(
                name='data', shape=[3, 4, 4, 5], dtype='float32'
            )
            flatten = paddle.nn.Flatten()
            ret = flatten(t)
            static_ret = self.get_static_graph_result(
                feed={'data': inp}, fetch_list=[ret]
            )[0]
        with self.dynamic_graph():
            t = paddle.to_tensor(inp)
            flatten = paddle.nn.Flatten()
            dy_ret = flatten(t)
            dy_ret_value = dy_ret.numpy()

        np.testing.assert_array_equal(static_ret, dy_ret_value)

        with self.static_graph():
            # the input of Linear must be Variable.
            def test_Variable():
                inp = np.ones([3, 32, 32], dtype='float32')
                linear = paddle.nn.Linear(
                    32,
                    4,
                    bias_attr=paddle.nn.initializer.Constant(value=1),
                )
                linear_ret1 = linear(inp)

            self.assertRaises(TypeError, test_Variable)

            # the input dtype of Linear must be float16 or float32 or float64
            # float16 only can be set on GPU place
            def test_type():
                inp = np.ones([3, 32, 32], dtype='int32')
                linear = paddle.nn.Linear(
                    32,
                    4,
                    bias_attr=paddle.nn.initializer.Constant(value=1),
                )
                linear_ret2 = linear(inp)

            self.assertRaises(TypeError, test_type)

    def test_SyncBatchNorm(self):
        if core.is_compiled_with_cuda():
            with self.static_graph():
                t = paddle.static.data(
                    name='t', shape=[-1, 3, 5, 5], dtype='float32'
                )
                my_sync_bn = paddle.nn.SyncBatchNorm(3)
                ret = my_sync_bn(t)
                static_ret = self.get_static_graph_result(
                    feed={'t': np.ones([3, 3, 5, 5], dtype='float32')},
                    fetch_list=[ret],
                )[0]

            with self.dynamic_graph():
                t = np.ones([3, 3, 5, 5], dtype='float32')
                my_syncbn = paddle.nn.SyncBatchNorm(3)
                dy_ret = my_syncbn(paddle.to_tensor(t))
                dy_ret_value = dy_ret.numpy()
            np.testing.assert_array_equal(static_ret, dy_ret_value)

    def test_relu(self):
        with self.static_graph():
            t = paddle.static.data(name='t', shape=[-1, 3, 3], dtype='float32')
            ret = F.relu(t)
            static_ret = self.get_static_graph_result(
                feed={'t': np.ones([3, 3], dtype='float32')}, fetch_list=[ret]
            )[0]

        with self.dynamic_graph():
            t = np.ones([3, 3], dtype='float32')
            dy_ret = F.relu(paddle.to_tensor(t))
            dy_ret_value = dy_ret.numpy()

        np.testing.assert_allclose(static_ret, dy_ret_value, rtol=1e-05)

    def test_matmul(self):
        with self.static_graph():
            t = paddle.static.data(name='t', shape=[-1, 3, 3], dtype='float32')
            t2 = paddle.static.data(
                name='t2', shape=[-1, 3, 3], dtype='float32'
            )
            ret = paddle.matmul(t, t2)
            static_ret = self.get_static_graph_result(
                feed={
                    't': np.ones([3, 3], dtype='float32'),
                    't2': np.ones([3, 3], dtype='float32'),
                },
                fetch_list=[ret],
            )[0]

        with self.dynamic_graph():
            t = np.ones([3, 3], dtype='float32')
            t2 = np.ones([3, 3], dtype='float32')
            dy_ret = paddle.matmul(paddle.to_tensor(t), paddle.to_tensor(t2))
            dy_ret_value = dy_ret.numpy()

        np.testing.assert_allclose(static_ret, dy_ret_value, rtol=1e-05)

    def test_elementwise_math(self):
        n = np.ones([3, 3], dtype='float32')
        n2 = np.ones([3, 3], dtype='float32') * 1.1
        n3 = np.ones([3, 3], dtype='float32') * 2
        n4 = np.ones([3, 3], dtype='float32') * 3
        n5 = np.ones([3, 3], dtype='float32') * 4
        n6 = np.ones([3, 3], dtype='float32') * 5

        with self.static_graph():
            t = paddle.static.data(name='t', shape=[-1, 3, 3], dtype='float32')
            t2 = paddle.static.data(
                name='t2', shape=[-1, 3, 3], dtype='float32'
            )
            t3 = paddle.static.data(
                name='t3', shape=[-1, 3, 3], dtype='float32'
            )
            t4 = paddle.static.data(
                name='t4', shape=[-1, 3, 3], dtype='float32'
            )
            t5 = paddle.static.data(
                name='t5', shape=[-1, 3, 3], dtype='float32'
            )
            t6 = paddle.static.data(
                name='t6', shape=[-1, 3, 3], dtype='float32'
            )

            ret = paddle.add(t, t2)
            ret = paddle.pow(ret, t3)
            ret = paddle.divide(ret, t4)
            ret = paddle.subtract(ret, t5)
            ret = paddle.multiply(ret, t6)

            static_ret = self.get_static_graph_result(
                feed={'t': n, 't2': n2, 't3': n3, 't4': n4, 't5': n5, 't6': n6},
                fetch_list=[ret],
            )[0]

        with self.dynamic_graph():
            ret = paddle.add(paddle.to_tensor(n), paddle.to_tensor(n2))
            ret = paddle.pow(ret, paddle.to_tensor(n3))
            ret = paddle.divide(ret, paddle.to_tensor(n4))
            ret = paddle.subtract(ret, paddle.to_tensor(n5))
            dy_ret = paddle.multiply(ret, paddle.to_tensor(n6))
            dy_ret_value = dy_ret.numpy()

        np.testing.assert_allclose(static_ret, dy_ret_value, rtol=1e-05)

    def test_elementwise_minmax(self):
        n = np.ones([3, 3], dtype='float32')
        n2 = np.ones([3, 3], dtype='float32') * 2

        with self.dynamic_graph():
            min_ret = paddle.minimum(paddle.to_tensor(n), paddle.to_tensor(n2))
            max_ret = paddle.maximum(paddle.to_tensor(n), paddle.to_tensor(n2))
            min_ret_value = min_ret.numpy()
            max_ret_value = max_ret.numpy()

        np.testing.assert_allclose(n, min_ret_value, rtol=1e-05)
        np.testing.assert_allclose(n2, max_ret_value, rtol=1e-05)

    def test_one_hot(self):
        with self.dynamic_graph():
            label = paddle.to_tensor(np.array([[1], [1], [3], [0]]))
            one_hot_label1 = paddle.nn.functional.one_hot(label, 4)
            one_hot_label2 = paddle.nn.functional.one_hot(
                label, paddle.to_tensor(np.array([4]))
            )
            np.testing.assert_array_equal(
                one_hot_label1.numpy(), one_hot_label2.numpy()
            )

    def test_split(self):
        with self.dynamic_graph():
            input = paddle.to_tensor(np.random.random((3, 8, 5)))
            x0, x1 = paddle.split(input, num_or_sections=2, axis=1)
            x00, x11 = paddle.split(
                input,
                num_or_sections=2,
                axis=paddle.to_tensor(np.array([1])),
            )
            np.testing.assert_array_equal(x0.numpy(), x00.numpy())
            np.testing.assert_array_equal(x1.numpy(), x11.numpy())

    def test_topk(self):
        with self.dynamic_graph():
            input = paddle.to_tensor(np.random.random((13, 11)))
            top5_values1, top5_indices1 = paddle.topk(input, k=5)
            top5_values2, top5_indices2 = paddle.topk(
                input, k=paddle.to_tensor(np.array([5]))
            )
            np.testing.assert_array_equal(
                top5_values1.numpy(), top5_values2.numpy()
            )
            np.testing.assert_array_equal(
                top5_indices1.numpy(), top5_indices2.numpy()
            )

    def test_compare(self):
        value_a = np.arange(3)
        value_b = np.arange(3)
        # less than
        with self.static_graph():
            a = paddle.static.data(name='a', shape=[-1, 1], dtype='int64')
            b = paddle.static.data(name='b', shape=[-1, 1], dtype='int64')
            cond = paddle.less_than(x=a, y=b)
            cond_ = paddle.less(x=a, y=b)
            static_ret = self.get_static_graph_result(
                feed={"a": value_a, "b": value_b}, fetch_list=[cond]
            )[0]
        with self.dynamic_graph():
            da = paddle.to_tensor(value_a)
            db = paddle.to_tensor(value_b)
            dcond = paddle.less_than(x=da, y=db)
            dcond_ = paddle.less(x=da, y=db)

            for i in range(len(static_ret)):
                self.assertTrue(dcond.numpy()[i] == static_ret[i])
                self.assertTrue(dcond_.numpy()[i] == static_ret[i])

        # less equal
        with self.static_graph():
            a1 = paddle.static.data(name='a1', shape=[-1, 1], dtype='int64')
            b1 = paddle.static.data(name='b1', shape=[-1, 1], dtype='int64')
            cond1 = paddle.less_equal(x=a1, y=b1)
            static_ret1 = self.get_static_graph_result(
                feed={"a1": value_a, "b1": value_b}, fetch_list=[cond1]
            )[0]
        with self.dynamic_graph():
            da1 = paddle.to_tensor(value_a)
            db1 = paddle.to_tensor(value_b)
            dcond1 = paddle.less_equal(x=da1, y=db1)

            for i in range(len(static_ret1)):
                self.assertTrue(dcond1.numpy()[i] == static_ret1[i])

        # greater than
        with self.static_graph():
            a2 = paddle.static.data(name='a2', shape=[-1, 1], dtype='int64')
            b2 = paddle.static.data(name='b2', shape=[-1, 1], dtype='int64')
            cond2 = paddle.greater_than(x=a2, y=b2)
            static_ret2 = self.get_static_graph_result(
                feed={"a2": value_a, "b2": value_b}, fetch_list=[cond2]
            )[0]
        with self.dynamic_graph():
            da2 = paddle.to_tensor(value_a)
            db2 = paddle.to_tensor(value_b)
            dcond2 = paddle.greater_than(x=da2, y=db2)

            for i in range(len(static_ret2)):
                self.assertTrue(dcond2.numpy()[i] == static_ret2[i])

        # greater equal
        with self.static_graph():
            a3 = paddle.static.data(name='a3', shape=[-1, 1], dtype='int64')
            b3 = paddle.static.data(name='b3', shape=[-1, 1], dtype='int64')
            cond3 = paddle.greater_equal(x=a3, y=b3)
            static_ret3 = self.get_static_graph_result(
                feed={"a3": value_a, "b3": value_b}, fetch_list=[cond3]
            )[0]
        with self.dynamic_graph():
            da3 = paddle.to_tensor(value_a)
            db3 = paddle.to_tensor(value_b)
            dcond3 = paddle.greater_equal(x=da3, y=db3)

            for i in range(len(static_ret3)):
                self.assertTrue(dcond3.numpy()[i] == static_ret3[i])

        # equal
        with self.static_graph():
            a4 = paddle.static.data(name='a4', shape=[-1, 1], dtype='int64')
            b4 = paddle.static.data(name='b4', shape=[-1, 1], dtype='int64')
            cond4 = paddle.equal(x=a4, y=b4)
            static_ret4 = self.get_static_graph_result(
                feed={"a4": value_a, "b4": value_b}, fetch_list=[cond4]
            )[0]
        with self.dynamic_graph():
            da4 = paddle.to_tensor(value_a)
            db4 = paddle.to_tensor(value_b)
            dcond4 = paddle.equal(x=da4, y=db4)

            for i in range(len(static_ret4)):
                self.assertTrue(dcond4.numpy()[i] == static_ret4[i])

        # not equal
        with self.static_graph():
            a5 = paddle.static.data(name='a5', shape=[-1, 1], dtype='int64')
            b5 = paddle.static.data(name='b5', shape=[-1, 1], dtype='int64')
            cond5 = paddle.equal(x=a5, y=b5)
            static_ret5 = self.get_static_graph_result(
                feed={"a5": value_a, "b5": value_b}, fetch_list=[cond5]
            )[0]
        with self.dynamic_graph():
            da5 = paddle.to_tensor(value_a)
            db5 = paddle.to_tensor(value_b)
            dcond5 = paddle.equal(x=da5, y=db5)

            for i in range(len(static_ret5)):
                self.assertTrue(dcond5.numpy()[i] == static_ret5[i])

    def test_crop_tensor(self):
        with self.static_graph():
            x = paddle.static.data(
                name="x1", shape=[-1, 6, 5, 8], dtype="float32"
            )

            dim1 = paddle.static.data(name="dim1", shape=[1], dtype="int32")
            dim2 = paddle.static.data(name="dim2", shape=[1], dtype="int32")
            crop_shape1 = (1, 2, 4, 4)
            crop_shape2 = paddle.static.data(
                name="crop_shape", shape=[4], dtype="float32"
            )
            crop_shape3 = [-1, dim1, dim2, 4]
            crop_offsets1 = [0, 0, 1, 0]
            crop_offsets2 = paddle.static.data(
                name="crop_offset", shape=[4], dtype="float32"
            )
            crop_offsets3 = [0, dim1, dim2, 0]

            out1 = paddle.crop(x, shape=crop_shape1, offsets=crop_offsets1)
            out2 = paddle.crop(x, shape=crop_shape2, offsets=crop_offsets2)
            out3 = paddle.crop(x, shape=crop_shape3, offsets=crop_offsets3)

            self.assertIsNotNone(out1)
            self.assertIsNotNone(out2)
            self.assertIsNotNone(out3)

    def test_shard_index(self):
        with self.static_graph():
            x = paddle.static.data(
                name="label", shape=[-1, 4, 1], dtype='int64'
            )
            shard_label = paddle.shard_index(
                input=x, index_num=20, nshards=2, shard_id=0
            )

        self.assertIsNotNone(shard_label)

    def test_accuracy(self):
        x = np.random.rand(3, 32, 32).astype("float32")
        y = np.array([[1], [0], [1]])
        with self.static_graph():
            data = paddle.static.data(
                name="input", shape=[-1, 32, 32], dtype="float32"
            )
            label = paddle.static.data(name="label", shape=[-1, 1], dtype="int")
            data_new = paddle.reshape(data, [3, 32 * 32])
            fc_out = paddle.nn.Linear(32 * 32, 10)(data_new)
            predict = paddle.nn.functional.softmax(fc_out)
            result = paddle.static.accuracy(input=predict, label=label, k=5)
            place = base.CPUPlace()
            exe = base.Executor(place)

            exe.run(base.default_startup_program())
            # x = np.random.rand(3, 32, 32).astype("float32")
            # y = np.array([[1], [0], [1]])

            static_out = exe.run(
                feed={"input": x, "label": y}, fetch_list=result
            )

        with self.dynamic_graph(force_to_use_cpu=True):
            data = paddle.to_tensor(x)
            label = paddle.to_tensor(y)
            data_new = paddle.reshape(data, [3, 32 * 32])
            fc_out = paddle.nn.Linear(32 * 32, 10)(data_new)
            predict = paddle.nn.functional.softmax(fc_out)
            dynamic_out = paddle.static.accuracy(
                input=predict, label=label, k=5
            )

        np.testing.assert_array_equal(static_out[0], dynamic_out.numpy())


class TestBook(LayerTest):
    def setUp(self):
        self.only_static_set = set({"make_word_embedding"})
        self.not_compare_static_dygraph_set = set(
            {
                "make_gaussian_random",
                "make_kldiv_loss",
                "make_uniform_random_batch_size_like",
            }
        )
        self.all_close_compare = set({"make_spectral_norm"})

    def test_all_layers(self):
        attrs = (getattr(self, name) for name in dir(self))
        methods = filter(inspect.ismethod, attrs)
        for method in methods:
            if not method.__name__.startswith('make_'):
                continue
            self._low_data_bound = 0
            self._high_data_bound = 2
            self._batch_size = 2
            self._feed_dict = {}
            self._force_to_use_cpu = False
            with self.static_graph():
                static_var = method()
                if isinstance(static_var, tuple):
                    static_var = static_var[0]

                if static_var is not None:
                    static_result = self.get_static_graph_result(
                        feed=self._feed_dict,
                        fetch_list=[static_var],
                        force_to_use_cpu=self._force_to_use_cpu,
                    )

                else:
                    continue
            if method.__name__ in self.only_static_set:
                continue

            with self.dynamic_graph(self._force_to_use_cpu):
                dy_result = method()
                if isinstance(dy_result, tuple):
                    dy_result = dy_result[0]
                dy_result_value = dy_result.numpy()

            if method.__name__ in self.all_close_compare:
                np.testing.assert_allclose(
                    static_result[0],
                    dy_result_value,
                    rtol=1e-05,
                    atol=0,
                    err_msg=f'Result of function [{method.__name__}] compare failed',
                )
                continue

            if method.__name__ not in self.not_compare_static_dygraph_set:
                np.testing.assert_array_equal(
                    static_result[0],
                    dy_result_value,
                    err_msg=f'Result of function [{method.__name__}] not equal',
                )

    def _get_np_data(self, shape, dtype, append_batch_size=True):
        np.random.seed(self.seed)
        if append_batch_size:
            shape = [self._batch_size, *shape]
        if dtype == 'float32':
            return np.random.random(shape).astype(dtype)
        elif dtype == 'float64':
            return np.random.random(shape).astype(dtype)
        elif dtype == 'int32':
            return np.random.randint(
                self._low_data_bound, self._high_data_bound, shape
            ).astype(dtype)
        elif dtype == 'int64':
            return np.random.randint(
                self._low_data_bound, self._high_data_bound, shape
            ).astype(dtype)

    def _get_data(
        self, name, shape, dtype, set_feed_dict=True, append_batch_size=True
    ):
        if dygraph.base.enabled():
            return paddle.to_tensor(
                self._get_np_data(shape, dtype, append_batch_size),
            )
        else:
            if set_feed_dict:
                self._feed_dict[name] = self._get_np_data(
                    shape, dtype, append_batch_size
                )
            if append_batch_size:
                shape = [-1, *shape]
            data = paddle.static.data(
                name=name,
                shape=shape,
                dtype=dtype,
            )
            if not paddle.framework.use_pir_api():
                data.desc.set_need_check_feed(False)
            return data

    def make_fit_a_line(self):
        with base.program_guard(
            base.default_main_program(),
            startup_program=base.default_startup_program(),
        ):
            x = self._get_data(name='x', shape=[13], dtype='float32')
            y_predict = paddle.nn.Linear(13, 1)(x)
            y = self._get_data(name='y', shape=[1], dtype='float32')
            cost = paddle.nn.functional.square_error_cost(
                input=y_predict, label=y
            )
            avg_cost = paddle.mean(cost)
            return avg_cost

    def make_recognize_digits_mlp(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            # Change g_program, so the rest layers use `g_program`
            images = self._get_data(name='pixel', shape=[784], dtype='float32')
            label = self._get_data(name='label', shape=[1], dtype='int64')
            hidden1 = paddle.nn.Linear(784, 128)(images)
            hidden1 = paddle.nn.functional.relu(hidden1)
            hidden2 = paddle.nn.Linear(128, 64)(hidden1)
            hidden2 = paddle.nn.functional.relu(hidden2)
            hidden1 = paddle.nn.Linear(128, 10, "sftmax.w1")(hidden1)
            hidden2 = paddle.nn.Linear(64, 10, "sftmax.w2")(hidden2)
            hidden = hidden1 + hidden2
            predict = paddle.nn.functional.softmax(hidden)
            cost = paddle.nn.functional.cross_entropy(
                input=predict, label=label, reduction='none', use_softmax=False
            )
            avg_cost = paddle.mean(cost)
            return avg_cost

    def make_pool2d(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x = self._get_data(name='x', shape=[3, 224, 224], dtype='float32')
            return paddle.nn.functional.max_pool2d(
                x, kernel_size=[5, 3], stride=[1, 2], padding=(2, 1)
            )

    def make_pool2d_infershape(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            theta = self._get_data("theta", shape=[2, 3], dtype='float32')
            x = paddle.nn.functional.affine_grid(
                theta, out_shape=[2, 3, 244, 244]
            )
            return paddle.nn.functional.max_pool2d(
                x, kernel_size=[5, 3], stride=[1, 2], padding=(2, 1)
            )

    def make_softmax(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            data = self._get_data(name='data', shape=[10], dtype='float32')
            hid = paddle.nn.Linear(10, 20)(data)
            return paddle.nn.functional.softmax(hid, axis=1)

    def make_multiplex(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x1 = self._get_data(name='x1', shape=[4], dtype='float32')
            x2 = self._get_data(name='x2', shape=[4], dtype='float32')
            index = self._get_data(name='index', shape=[1], dtype='int32')
            out = paddle.multiplex(inputs=[x1, x2], index=index)
            return out

    def make_softmax_with_cross_entropy(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x = self._get_data(name='x', shape=[16], dtype='float32')
            y = self._get_data(name='label', shape=[1], dtype='int64')
            loss, softmax = paddle.nn.functional.softmax_with_cross_entropy(
                x, y, return_softmax=True
            )
            self.assertIsNotNone(loss)
            self.assertIsNotNone(softmax)

            loss = paddle.nn.functional.softmax_with_cross_entropy(x, y)
            self.assertIsNotNone(loss)

            x1 = self._get_data(name='x1', shape=[16, 32, 64], dtype='float32')
            y1 = self._get_data(name='label1', shape=[1, 32, 64], dtype='int64')
            y2 = self._get_data(name='label2', shape=[16, 1, 64], dtype='int64')
            y3 = self._get_data(name='label3', shape=[16, 32, 1], dtype='int64')
            loss1 = paddle.nn.functional.softmax_with_cross_entropy(
                x1, y1, axis=1
            )
            loss2 = paddle.nn.functional.softmax_with_cross_entropy(
                x1, y2, axis=2
            )
            loss3 = paddle.nn.functional.softmax_with_cross_entropy(
                x1, y3, axis=3
            )
            loss4 = paddle.nn.functional.softmax_with_cross_entropy(
                x1, y3, axis=-1
            )
            self.assertIsNotNone(loss1)
            self.assertIsNotNone(loss2)
            self.assertIsNotNone(loss3)
            self.assertIsNotNone(loss4)
            return loss4

    def make_scatter(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x = self._get_data(
                name='x', shape=[3, 3], append_batch_size=False, dtype='float32'
            )
            idx = self._get_data(
                name='idx', shape=[2], append_batch_size=False, dtype='int32'
            )
            updates = self._get_data(
                name='updates',
                shape=[2, 3],
                dtype='float32',
                append_batch_size=False,
            )
            out = paddle.scatter(x, index=idx, updates=updates)
            return out

    def make_one_hot(self):
        with base.framework._dygraph_place_guard(place=base.CPUPlace()):
            label = self._get_data(name="label", shape=[1], dtype="int32")
            one_hot_label = paddle.nn.functional.one_hot(label, 10)
            return one_hot_label

    def make_label_smooth(self):
        # TODO(minqiyang): support gpu ut
        self._force_to_use_cpu = True
        with base.framework._dygraph_place_guard(place=base.CPUPlace()):
            label = self._get_data(name="label", shape=[1], dtype="int32")
            one_hot_label = paddle.nn.functional.one_hot(label, 10)
            smooth_label = F.label_smooth(label=one_hot_label, epsilon=0.1)
            return smooth_label

    def make_topk(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            data = self._get_data(name="label", shape=[200], dtype="float32")
            values, indices = paddle.topk(data, k=5)
            return values
            return indices

    def make_l2_normalize(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x = self._get_data(name='x', shape=[8, 7, 10], dtype="float32")
            output = paddle.nn.functional.normalize(x, axis=1)
            return output

    def make_shape(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            input = self._get_data(
                name="input", shape=[3, 100, 100], dtype="float32"
            )
            out = paddle.shape(input)
            return out

    def make_pad2d(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            input = self._get_data(
                name="input", shape=[3, 100, 100], dtype="float32"
            )

            tmp_pad = paddle.nn.Pad2D(
                padding=[1, 2, 3, 4],
                mode='reflect',
                data_format='NCHW',
                name="shape",
            )
            out = tmp_pad(input)
            return out

    def make_mish(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = paddle.nn.functional.mish(input, name='mish')
            return out

    def make_cross_entropy(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x = self._get_data(name="x", shape=[30, 10], dtype="float32")
            label = self._get_data(name="label", shape=[30, 1], dtype="int64")
            mode = 'channel'
            out = paddle.nn.functional.cross_entropy(
                x,
                label,
                soft_label=False,
                ignore_index=4,
                reduction='none',
                use_softmax=False,
            )
            return out

    def make_gaussian_random(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            out = random.gaussian(shape=[20, 30])
            return out

    def make_sum(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            input = self._get_data(
                name="input", shape=[13, 11], dtype='float32'
            )

            out = paddle.add_n(input)
            return out

    def make_slice(self):
        starts = [1, 0, 2]
        ends = [3, 3, 4]
        axes = [0, 1, 2]

        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            input = self._get_data(
                name="input", shape=[3, 4, 5, 6], dtype='float32'
            )

            out = paddle.slice(input, axes=axes, starts=starts, ends=ends)
            return out

    def make_scale_variable(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            input = self._get_data(
                name="input", shape=[3, 4, 5, 6], dtype='float32'
            )
            scale_var = self._get_data(
                name="scale",
                shape=[1],
                dtype='float32',
                append_batch_size=False,
            )
            out = paddle.scale(input, scale=scale_var)
            return out

    def make_range(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            paddle.arange(0, 10, 2, 'int32')
            paddle.arange(0.1, 10.0, 0.2, 'float32')
            paddle.arange(0.1, 10.0, 0.2, 'float64')
            start = paddle.tensor.fill_constant(
                shape=[1], value=0.1, dtype="float32"
            )
            end = paddle.tensor.fill_constant(
                shape=[1], value=10.0, dtype="float32"
            )
            step = paddle.tensor.fill_constant(
                shape=[1], value=0.2, dtype="float32"
            )
            y = paddle.arange(start, end, step, 'float64')
            return y

    def make_kldiv_loss(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x = self._get_data(
                name='x',
                shape=[32, 128, 128],
                dtype="float32",
                append_batch_size=False,
            )
            target = self._get_data(
                name='target',
                shape=[32, 128, 128],
                dtype="float32",
                append_batch_size=False,
            )
            loss = paddle.nn.functional.kl_div(
                input=x, label=target, reduction='batchmean'
            )
            return loss

    def make_pixel_shuffle(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x = self._get_data(name="X", shape=[9, 4, 4], dtype="float32")
            out = paddle.nn.functional.pixel_shuffle(x, upscale_factor=3)
            return out

    def make_mse_loss(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x = self._get_data(name="X", shape=[1], dtype="float32")
            y = self._get_data(name="Y", shape=[1], dtype="float32")
            out = paddle.nn.functional.mse_loss(input=x, label=y)
            return out

    def make_square_error_cost(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x = self._get_data(name="X", shape=[1], dtype="float32")
            y = self._get_data(name="Y", shape=[1], dtype="float32")
            out = paddle.nn.functional.square_error_cost(input=x, label=y)
            return out

    def test_affine_grid(self):
        with self.static_graph():
            data = paddle.static.data(
                name='data', shape=[-1, 2, 3, 3], dtype="float32"
            )
            out = paddle.argsort(x=data, axis=1)

            theta = paddle.static.data(
                name="theta", shape=[-1, 2, 3], dtype="float32"
            )
            out_shape = paddle.static.data(
                name="out_shape", shape=[-1], dtype="int32"
            )
            data_0 = paddle.nn.functional.affine_grid(theta, out_shape)
            data_1 = paddle.nn.functional.affine_grid(theta, [5, 3, 28, 28])

            self.assertIsNotNone(data_0)
            self.assertIsNotNone(data_1)

    def test_stridedslice(self):
        axes = [0, 1, 2]
        starts = [1, 0, 2]
        ends = [3, 3, 4]
        strides = [1, 1, 1]
        with self.static_graph():
            x = paddle.static.data(
                name="x", shape=[-1, 245, 30, 30], dtype="float32"
            )
            out = paddle.strided_slice(
                x, axes=axes, starts=starts, ends=ends, strides=strides
            )
            return out

    def test_squeeze(self):
        # TODO(minqiyang): dygraph do not support layers with param now
        with self.static_graph():
            x = paddle.static.data(
                name='x', shape=[-1, 1, 1, 4], dtype='float32'
            )
            out = paddle.squeeze(x, axis=[2])
            return out

    def test_flatten(self):
        # TODO(minqiyang): dygraph do not support op without kernel now
        with self.static_graph():
            x = paddle.static.data(
                name='x',
                shape=[4, 4, 3],
                dtype="float32",
            )
            out = paddle.flatten(x, 1, -1, name="flatten")
            return out

    def test_linspace(self):
        program = base.Program()
        with base.program_guard(program):
            out = paddle.linspace(20, 10, 5, 'float64')
            self.assertIsNotNone(out)
        print(str(program))

    def test_unfold(self):
        with self.static_graph():
            x = paddle.static.data(
                name='x', shape=[-1, 3, 20, 20], dtype='float32'
            )
            out = paddle.nn.functional.unfold(x, [3, 3], 1, 1, 1)
            return out

    def test_addmm(self):
        with base.program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            input = paddle.static.data(
                name='input_data',
                shape=[3, 3],
                dtype='float32',
            )
            x = paddle.static.data(name='x', shape=[3, 2], dtype='float32')
            y = paddle.static.data(name='y', shape=[2, 3], dtype='float32')

            out = paddle.addmm(input=input, x=x, y=y)
            return out

    def test_warpctc_with_padding(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            input_length = paddle.static.data(
                name='logits_length', shape=[11], dtype='int64'
            )
            label_length = paddle.static.data(
                name='labels_length', shape=[12], dtype='int64'
            )
            label = paddle.static.data(
                name='label', shape=[12, 1], dtype='int32'
            )
            predict = paddle.static.data(
                name='predict', shape=[4, 4, 8], dtype='float32'
            )
            output = paddle.nn.functional.ctc_loss(
                log_probs=predict,
                labels=label,
                input_lengths=input_length,
                label_lengths=label_length,
                reduction='none',
            )
            return output


class ExampleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[1, 1], attr=paddle.ParamAttr(trainable=False)
        )

    def forward(self):
        # only for test parameter trainable attr
        pass


class TestLayerParameterTrainableSet(unittest.TestCase):
    def test_layer_parameter_set(self):
        with base.dygraph.guard():
            net = ExampleNet()
            self.assertFalse(net.weight.trainable)


class TestLayerTrainingAttribute(unittest.TestCase):
    def test_set_train_eval_in_dynamic_mode(self):
        with base.dygraph.guard():
            net = paddle.nn.Dropout()
            net.train()
            self.assertTrue(net.training)
            net.eval()
            self.assertFalse(net.training)

    def test_set_train_eval_in_static_mode(self):
        net = paddle.nn.Dropout()
        net.train()
        self.assertTrue(net.training)
        net.eval()
        self.assertFalse(net.training)


class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear = paddle.nn.Linear(1, 1)
        self._dropout = paddle.nn.Dropout(p=0.5)

    def forward(self, input):
        temp = self._linear(input)
        temp = self._dropout(temp)
        return temp


class MySuperLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._mylayer = MyLayer()

    def forward(self, input):
        temp = self._mylayer(input)
        return temp


class TestSubLayerCount(unittest.TestCase):
    def test_sublayer(self):
        with base.dygraph.guard():
            mySuperlayer = MySuperLayer()
            self.assertTrue(len(mySuperlayer.sublayers()) == 3)
            self.assertTrue(len(mySuperlayer.sublayers(include_self=True)) == 4)


class TestExcludedLayersSupportBool(unittest.TestCase):
    def test_support_tuple(self):
        with base.dygraph.guard():
            model = MyLayer()
            model.float16(excluded_layers=[paddle.nn.Linear])
            self.assertTrue(model._linear.weight.dtype == paddle.float32)
            model.bfloat16(excluded_layers=(paddle.nn.Linear))
            self.assertTrue(model._linear.weight.dtype == paddle.float32)


class TestLayerClearGradientSetToZero(unittest.TestCase):
    def test_layer_clear_gradient_set_to_zero_true(self):
        with base.dygraph.guard():
            net = MyLayer()
            inputs = paddle.randn([10, 1])
            outputs = net(inputs)
            outputs.backward()
            net.clear_gradients()
            self.assertTrue(
                net._linear.weight.grad.numpy() == np.array([[0.0]])
            )

    def test_layer_clear_gradient_set_to_zero_false(self):
        with base.dygraph.guard():
            net = MyLayer()
            inputs = paddle.randn([10, 1])
            outputs = net(inputs)
            outputs.backward()
            net.clear_gradients(set_to_zero=False)
            self.assertTrue(net._linear.weight.grad is None)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()

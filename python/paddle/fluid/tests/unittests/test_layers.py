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

from __future__ import print_function
import unittest

import contextlib
import numpy as np
import decorators

import paddle
import paddle.fluid as fluid
from paddle.fluid.layers.device import get_places
import paddle.fluid.nets as nets
from paddle.fluid.framework import Program, program_guard, default_main_program
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid import core
from paddle.fluid.initializer import Constant
import paddle.fluid.layers as layers
from test_imperative_base import new_program_scope
from paddle.fluid.dygraph import nn
from paddle.fluid.dygraph import base


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
        with new_program_scope():
            fluid.default_startup_program().random_seed = self.seed
            fluid.default_main_program().random_seed = self.seed
            yield

    def get_static_graph_result(self, feed, fetch_list, with_lod=False):
        exe = fluid.Executor(self._get_place())
        exe.run(fluid.default_startup_program())
        return exe.run(fluid.default_main_program(),
                       feed=feed,
                       fetch_list=fetch_list,
                       return_numpy=(not with_lod))

    @contextlib.contextmanager
    def dynamic_graph(self, force_to_use_cpu=False):
        with fluid.dygraph.guard(
                self._get_place(force_to_use_cpu=force_to_use_cpu)):
            fluid.default_startup_program().random_seed = self.seed
            fluid.default_main_program().random_seed = self.seed
            yield


class TestLayer(LayerTest):
    def test_layer_norm(self):
        inp = np.ones([3, 32, 32], dtype='float32')
        with self.static_graph():
            t = layers.data(
                name='data',
                shape=[3, 32, 32],
                dtype='float32',
                append_batch_size=False)
            ret = layers.layer_norm(t)
            static_ret = self.get_static_graph_result(
                feed={'data': inp}, fetch_list=[ret])[0]
        with self.static_graph():
            t = layers.data(
                name='data',
                shape=[3, 32, 32],
                dtype='float32',
                append_batch_size=False)
            lm = nn.LayerNorm('layer_norm')
            ret = lm(t)
            static_ret2 = self.get_static_graph_result(
                feed={'data': inp}, fetch_list=[ret])[0]
        with self.dynamic_graph():
            lm = nn.LayerNorm('layer_norm')
            dy_ret = lm(base.to_variable(inp))

        self.assertTrue(np.allclose(static_ret, static_ret2))
        self.assertTrue(np.allclose(dy_ret.numpy(), static_ret2))

    def test_relu(self):
        with self.static_graph():
            t = layers.data(name='t', shape=[3, 3], dtype='float32')
            ret = layers.relu(t)
            static_ret = self.get_static_graph_result(
                feed={'t': np.ones(
                    [3, 3], dtype='float32')}, fetch_list=[ret])[0]

        with self.dynamic_graph():
            t = np.ones([3, 3], dtype='float32')
            dy_ret = layers.relu(base.to_variable(t))

        self.assertTrue(np.allclose(static_ret, dy_ret.numpy()))

    def test_matmul(self):
        with self.static_graph():
            t = layers.data(name='t', shape=[3, 3], dtype='float32')
            t2 = layers.data(name='t2', shape=[3, 3], dtype='float32')
            ret = layers.matmul(t, t2)
            static_ret = self.get_static_graph_result(
                feed={
                    't': np.ones(
                        [3, 3], dtype='float32'),
                    't2': np.ones(
                        [3, 3], dtype='float32')
                },
                fetch_list=[ret])[0]

        with self.dynamic_graph():
            t = np.ones([3, 3], dtype='float32')
            t2 = np.ones([3, 3], dtype='float32')
            dy_ret = layers.matmul(base.to_variable(t), base.to_variable(t2))

        self.assertTrue(np.allclose(static_ret, dy_ret.numpy()))

    def test_conv2d(self):
        with self.static_graph():
            images = layers.data(name='pixel', shape=[3, 5, 5], dtype='float32')
            ret = layers.conv2d(input=images, num_filters=3, filter_size=[2, 2])
            static_ret = self.get_static_graph_result(
                feed={'pixel': np.ones(
                    [2, 3, 5, 5], dtype='float32')},
                fetch_list=[ret])[0]

        with self.static_graph():
            images = layers.data(name='pixel', shape=[3, 5, 5], dtype='float32')
            conv2d = nn.Conv2D(
                'conv2d', num_channels=3, num_filters=3, filter_size=[2, 2])
            ret = conv2d(images)
            static_ret2 = self.get_static_graph_result(
                feed={'pixel': np.ones(
                    [2, 3, 5, 5], dtype='float32')},
                fetch_list=[ret])[0]

        with self.dynamic_graph():
            images = np.ones([2, 3, 5, 5], dtype='float32')
            conv2d = nn.Conv2D(
                'conv2d', num_channels=3, num_filters=3, filter_size=[2, 2])
            dy_ret = conv2d(base.to_variable(images))

        self.assertTrue(np.allclose(static_ret, dy_ret.numpy()))
        self.assertTrue(np.allclose(static_ret, static_ret2))

    def test_gru_unit(self):
        lod = [[2, 4, 3]]
        D = 5
        T = sum(lod[0])
        N = len(lod[0])

        input = np.random.rand(T, 3 * D).astype('float32')
        hidden_input = np.random.rand(T, D).astype('float32')

        with self.static_graph():
            x = layers.data(name='x', shape=[-1, D * 3], dtype='float32')
            hidden = layers.data(name='hidden', shape=[-1, D], dtype='float32')
            updated_hidden, reset_hidden_pre, gate = layers.gru_unit(
                input=x, hidden=hidden, size=D * 3)
            static_ret = self.get_static_graph_result(
                feed={'x': input,
                      'hidden': hidden_input},
                fetch_list=[updated_hidden, reset_hidden_pre, gate])

        with self.static_graph():
            x = layers.data(name='x', shape=[-1, D * 3], dtype='float32')
            hidden = layers.data(name='hidden', shape=[-1, D], dtype='float32')
            updated_hidden, reset_hidden_pre, gate = layers.gru_unit(
                input=x, hidden=hidden, size=D * 3)
            gru = nn.GRUUnit('gru', size=D * 3)
            updated_hidden, reset_hidden_pre, gate = gru(x, hidden)

            static_ret2 = self.get_static_graph_result(
                feed={'x': input,
                      'hidden': hidden_input},
                fetch_list=[updated_hidden, reset_hidden_pre, gate])

        with self.dynamic_graph():
            gru = nn.GRUUnit('gru', size=D * 3)
            dy_ret = gru(
                base.to_variable(input), base.to_variable(hidden_input))

        for i in range(len(static_ret)):
            self.assertTrue(np.allclose(static_ret[i], static_ret2[i]))
            self.assertTrue(np.allclose(static_ret[i], dy_ret[i].numpy()))

    def test_elementwise_math(self):
        n = np.ones([3, 3], dtype='float32')
        n2 = np.ones([3, 3], dtype='float32') * 1.1
        n3 = np.ones([3, 3], dtype='float32') * 2
        n4 = np.ones([3, 3], dtype='float32') * 3
        n5 = np.ones([3, 3], dtype='float32') * 4
        n6 = np.ones([3, 3], dtype='float32') * 5

        with self.static_graph():
            t = layers.data(name='t', shape=[3, 3], dtype='float32')
            t2 = layers.data(name='t2', shape=[3, 3], dtype='float32')
            t3 = layers.data(name='t3', shape=[3, 3], dtype='float32')
            t4 = layers.data(name='t4', shape=[3, 3], dtype='float32')
            t5 = layers.data(name='t5', shape=[3, 3], dtype='float32')
            t6 = layers.data(name='t6', shape=[3, 3], dtype='float32')

            ret = layers.elementwise_add(t, t2)
            ret = layers.elementwise_pow(ret, t3)
            ret = layers.elementwise_div(ret, t4)
            ret = layers.elementwise_sub(ret, t5)
            ret = layers.elementwise_mul(ret, t6)

            static_ret = self.get_static_graph_result(
                feed={
                    't': n,
                    't2': n2,
                    't3': n3,
                    't4': n4,
                    't5': n5,
                    't6': n6
                },
                fetch_list=[ret])[0]

        with self.dynamic_graph():
            ret = layers.elementwise_add(n, n2)
            ret = layers.elementwise_pow(ret, n3)
            ret = layers.elementwise_div(ret, n4)
            ret = layers.elementwise_sub(ret, n5)
            dy_ret = layers.elementwise_mul(ret, n6)
        self.assertTrue(
            np.allclose(static_ret, dy_ret.numpy()),
            '%s vs %s' % (static_ret, dy_ret.numpy()))

    def test_elementwise_minmax(self):
        n = np.ones([3, 3], dtype='float32')
        n2 = np.ones([3, 3], dtype='float32') * 2

        with self.dynamic_graph():
            min_ret = layers.elementwise_min(n, n2)
            max_ret = layers.elementwise_max(n, n2)

        self.assertTrue(np.allclose(n, min_ret.numpy()))
        self.assertTrue(np.allclose(n2, max_ret.numpy()))

    def test_sequence_conv(self):
        inp_np = np.arange(12).reshape([3, 4]).astype('float32')
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        with self.static_graph():
            seq = layers.data(
                name='seq_in',
                shape=[3, 4],
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            out = layers.sequence_conv(seq, 2)
            static_rlt = self.get_static_graph_result(
                feed={
                    "seq_in": fluid.create_lod_tensor(
                        data=inp_np,
                        recursive_seq_lens=[[1, 1, 1]],
                        place=place)
                },
                fetch_list=[out],
                with_lod=True)[0]

        with self.static_graph():
            seq = layers.data(
                name='seq_in',
                shape=[3, 4],
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            seq_conv = nn.SequenceConv('seq_conv', num_filters=2)
            out = seq_conv(seq)
            static_rlt2 = self.get_static_graph_result(
                feed={
                    "seq_in": fluid.create_lod_tensor(
                        data=inp_np,
                        recursive_seq_lens=[[1, 1, 1]],
                        place=place)
                },
                fetch_list=[out],
                with_lod=True)[0]
        self.assertTrue(
            np.allclose(np.array(static_rlt), np.array(static_rlt2)))

    def test_conv2d_transpose(self):
        inp_np = np.arange(0, 24).reshape([2, 3, 2, 2]).astype('float32')
        with self.static_graph():
            img = layers.data(name='pixel', shape=[3, 2, 2], dtype='float32')
            out = layers.conv2d_transpose(
                input=img, num_filters=10, output_size=28)
            static_rlt = self.get_static_graph_result(
                feed={'pixel': inp_np}, fetch_list=[out])[0]
        with self.static_graph():
            img = layers.data(name='pixel', shape=[3, 2, 2], dtype='float32')
            conv2d_transpose = nn.Conv2DTranspose(
                'conv2d_transpose', num_filters=10, output_size=28)
            out = conv2d_transpose(img)
            static_rlt2 = self.get_static_graph_result(
                feed={'pixel': inp_np}, fetch_list=[out])[0]
        with self.dynamic_graph():
            conv2d_transpose = nn.Conv2DTranspose(
                'conv2d_transpose', num_filters=10, output_size=28)
            dy_rlt = conv2d_transpose(base.to_variable(inp_np))
        self.assertTrue(np.allclose(static_rlt2, static_rlt))
        self.assertTrue(np.allclose(dy_rlt.numpy(), static_rlt))

    def test_bilinear_tensor_product(self):
        inp_np_x = np.array([[1, 2, 3]]).astype('float32')
        inp_np_y = np.array([[4, 5, 6]]).astype('float32')

        with self.static_graph():
            data_x = layers.data(
                name='x',
                shape=[1, 3],
                dtype="float32",
                append_batch_size=False)
            data_y = layers.data(
                name='y',
                shape=[1, 3],
                dtype="float32",
                append_batch_size=False)
            out = layers.bilinear_tensor_product(data_x, data_y, 6)

            static_rlt = self.get_static_graph_result(
                feed={'x': inp_np_x,
                      'y': inp_np_y}, fetch_list=[out])[0]
        with self.static_graph():
            data_x = layers.data(
                name='x',
                shape=[1, 3],
                dtype="float32",
                append_batch_size=False)
            data_y = layers.data(
                name='y',
                shape=[1, 3],
                dtype="float32",
                append_batch_size=False)
            btp = nn.BilinearTensorProduct('btp', 6)
            out = btp(data_x, data_y)
            static_rlt2 = self.get_static_graph_result(
                feed={'x': inp_np_x,
                      'y': inp_np_y}, fetch_list=[out])[0]
        with self.dynamic_graph():
            btp = nn.BilinearTensorProduct('btp', 6)
            dy_rlt = btp(base.to_variable(inp_np_x), base.to_variable(inp_np_y))

        self.assertTrue(np.allclose(static_rlt2, static_rlt))
        self.assertTrue(np.allclose(dy_rlt.numpy(), static_rlt))

    def test_prelu(self):
        inp_np = np.ones([5, 200, 100, 100]).astype('float32')

        with self.static_graph():
            data_t = layers.data(
                name="input",
                shape=[5, 200, 100, 100],
                dtype="float32",
                append_batch_size=False)
            mode = 'channel'
            out = layers.prelu(
                data_t, mode, param_attr=ParamAttr(initializer=Constant(1.0)))
            static_rlt = self.get_static_graph_result(
                feed={"input": inp_np}, fetch_list=[out])[0]

        with self.static_graph():
            data_t = layers.data(
                name="input",
                shape=[5, 200, 100, 100],
                dtype="float32",
                append_batch_size=False)
            mode = 'channel'
            prelu = nn.PRelu(
                'prelu',
                mode=mode,
                param_attr=ParamAttr(initializer=Constant(1.0)))
            out = prelu(data_t)
            static_rlt2 = self.get_static_graph_result(
                feed={"input": inp_np}, fetch_list=[out])[0]

        with self.dynamic_graph():
            mode = 'channel'
            prelu = nn.PRelu(
                'prelu',
                mode=mode,
                param_attr=ParamAttr(initializer=Constant(1.0)))
            dy_rlt = prelu(base.to_variable(inp_np))

        self.assertTrue(np.allclose(static_rlt2, static_rlt))
        self.assertTrue(np.allclose(dy_rlt.numpy(), static_rlt))

    def test_embeding(self):
        inp_word = np.array([[[1]]]).astype('int64')
        dict_size = 20
        with self.static_graph():
            data_t = layers.data(name='word', shape=[1], dtype='int64')
            emb = layers.embedding(
                input=data_t,
                size=[dict_size, 32],
                param_attr='emb.w',
                is_sparse=False)
            static_rlt = self.get_static_graph_result(
                feed={'word': inp_word}, fetch_list=[emb])[0]
        with self.static_graph():
            data_t = layers.data(name='word', shape=[1], dtype='int64')
            emb2 = nn.Embedding(
                name_scope='embedding',
                size=[dict_size, 32],
                param_attr='emb.w',
                is_sparse=False)
            emb_rlt = emb2(data_t)
            static_rlt2 = self.get_static_graph_result(
                feed={'word': inp_word}, fetch_list=[emb_rlt])[0]
        with self.dynamic_graph():
            emb2 = nn.Embedding(
                name_scope='embedding',
                size=[dict_size, 32],
                param_attr='emb.w',
                is_sparse=False)
            static_rlt3 = emb2(base.to_variable(inp_word))

        self.assertTrue(np.allclose(static_rlt2, static_rlt))
        self.assertTrue(np.allclose(static_rlt3.numpy(), static_rlt))

    def test_nce(self):
        window_size = 5
        dict_size = 20
        label_word = int(window_size // 2) + 1
        inp_word = np.array([[[1]], [[2]], [[3]], [[4]], [[5]]]).astype('int64')
        nid_freq_arr = np.random.dirichlet(np.ones(20) * 1000).astype('float32')
        seed = 1
        with self.static_graph():
            words = []
            for i in range(window_size):
                words.append(
                    layers.data(
                        name='word_{0}'.format(i), shape=[1], dtype='int64'))

            embs = []
            for i in range(window_size):
                if i == label_word:
                    continue

                emb = layers.embedding(
                    input=words[i],
                    size=[dict_size, 32],
                    param_attr='emb.w',
                    is_sparse=False)
                embs.append(emb)

            embs = layers.concat(input=embs, axis=1)
            nce_loss = layers.nce(input=embs,
                                  label=words[label_word],
                                  num_total_classes=dict_size,
                                  num_neg_samples=2,
                                  sampler="custom_dist",
                                  custom_dist=nid_freq_arr.tolist(),
                                  seed=seed,
                                  param_attr='nce.w',
                                  bias_attr='nce.b')
            feed_dict = dict()
            for i in range(window_size):
                feed_dict['word_{0}'.format(i)] = inp_word[i]
            static_rlt = self.get_static_graph_result(
                feed=feed_dict, fetch_list=[nce_loss])[0]
        with self.static_graph():
            words = []
            for i in range(window_size):
                words.append(
                    layers.data(
                        name='word_{0}'.format(i), shape=[1], dtype='int64'))

            emb = nn.Embedding(
                'embedding',
                size=[dict_size, 32],
                param_attr='emb.w',
                is_sparse=False)

            embs2 = []
            for i in range(window_size):
                if i == label_word:
                    continue

                emb_rlt = emb(words[i])
                embs2.append(emb_rlt)

            embs2 = layers.concat(input=embs2, axis=1)
            nce = nn.NCE('nce',
                         num_total_classes=dict_size,
                         num_neg_samples=2,
                         sampler="custom_dist",
                         custom_dist=nid_freq_arr.tolist(),
                         seed=seed,
                         param_attr='nce.w',
                         bias_attr='nce.b')

            nce_loss2 = nce(embs2, words[label_word])
            feed_dict = dict()
            for i in range(len(words)):
                feed_dict['word_{0}'.format(i)] = inp_word[i]

            static_rlt2 = self.get_static_graph_result(
                feed=feed_dict, fetch_list=[nce_loss2])[0]

        with self.dynamic_graph(force_to_use_cpu=True):
            words = []
            for i in range(window_size):
                words.append(base.to_variable(inp_word[i]))

            emb = nn.Embedding(
                'embedding',
                size=[dict_size, 32],
                param_attr='emb.w',
                is_sparse=False)

            embs3 = []
            for i in range(window_size):
                if i == label_word:
                    continue

                emb_rlt = emb(words[i])
                embs3.append(emb_rlt)

            embs3 = layers.concat(input=embs3, axis=1)
            nce = nn.NCE('nce',
                         num_total_classes=dict_size,
                         num_neg_samples=2,
                         sampler="custom_dist",
                         custom_dist=nid_freq_arr.tolist(),
                         seed=seed,
                         param_attr='nce.w',
                         bias_attr='nce.b')

            nce_loss3 = nce(embs3, words[label_word])

        self.assertTrue(np.allclose(static_rlt2, static_rlt))
        self.assertTrue(np.allclose(nce_loss3.numpy(), static_rlt))

    def test_conv3d(self):
        with self.static_graph():
            images = layers.data(
                name='pixel', shape=[3, 6, 6, 6], dtype='float32')
            ret = layers.conv3d(input=images, num_filters=3, filter_size=2)
            static_ret = self.get_static_graph_result(
                feed={'pixel': np.ones(
                    [2, 3, 6, 6, 6], dtype='float32')},
                fetch_list=[ret])[0]

        with self.static_graph():
            images = layers.data(
                name='pixel', shape=[3, 6, 6, 6], dtype='float32')
            conv3d = nn.Conv3D('conv3d', num_filters=3, filter_size=2)
            ret = conv3d(images)
            static_ret2 = self.get_static_graph_result(
                feed={'pixel': np.ones(
                    [2, 3, 6, 6, 6], dtype='float32')},
                fetch_list=[ret])[0]

        with self.dynamic_graph():
            images = np.ones([2, 3, 6, 6, 6], dtype='float32')
            conv3d = nn.Conv3D('conv3d', num_filters=3, filter_size=2)
            dy_ret = conv3d(base.to_variable(images))

        self.assertTrue(np.allclose(static_ret, dy_ret.numpy()))
        self.assertTrue(np.allclose(static_ret, static_ret2))

    def test_row_conv(self):
        input = np.arange(15).reshape([3, 5]).astype('float32')
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        with self.static_graph():
            x = layers.data(
                name='X',
                shape=[3, 5],
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            ret = layers.row_conv(input=x, future_context_size=2)
            static_ret = self.get_static_graph_result(
                feed={
                    'X': fluid.create_lod_tensor(
                        data=input, recursive_seq_lens=[[1, 1, 1]], place=place)
                },
                fetch_list=[ret],
                with_lod=True)[0]

        with self.static_graph():
            x = layers.data(
                name='X',
                shape=[3, 5],
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            rowConv = nn.RowConv('RowConv', future_context_size=2)
            ret = rowConv(x)
            static_ret2 = self.get_static_graph_result(
                feed={
                    'X': fluid.create_lod_tensor(
                        data=input, recursive_seq_lens=[[1, 1, 1]], place=place)
                },
                fetch_list=[ret],
                with_lod=True)[0]

        # TODO: dygraph can't support LODTensor

        self.assertTrue(np.allclose(static_ret, static_ret2))

    def test_group_norm(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        shape = (2, 4, 3, 3)

        input = np.random.random(shape).astype('float32')

        with self.static_graph():
            X = fluid.layers.data(
                name='X',
                shape=shape,
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            ret = layers.group_norm(input=X, groups=2)
            static_ret = self.get_static_graph_result(
                feed={
                    'X': fluid.create_lod_tensor(
                        data=input, recursive_seq_lens=[[1, 1]], place=place)
                },
                fetch_list=[ret],
                with_lod=True)[0]

        with self.static_graph():
            X = fluid.layers.data(
                name='X',
                shape=shape,
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            groupNorm = nn.GroupNorm('GroupNorm', groups=2)
            ret = groupNorm(X)
            static_ret2 = self.get_static_graph_result(
                feed={
                    'X': fluid.create_lod_tensor(
                        data=input, recursive_seq_lens=[[1, 1]], place=place)
                },
                fetch_list=[ret],
                with_lod=True)[0]

        with self.dynamic_graph():
            groupNorm = nn.GroupNorm('GroupNorm', groups=2)
            dy_ret = groupNorm(base.to_variable(input))

        self.assertTrue(np.allclose(static_ret, dy_ret.numpy()))
        self.assertTrue(np.allclose(static_ret, static_ret2))

    def test_spectral_norm(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        shape = (2, 4, 3, 3)

        input = np.random.random(shape).astype('float32')

        with self.static_graph():
            Weight = fluid.layers.data(
                name='Weight',
                shape=shape,
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            ret = layers.spectral_norm(weight=Weight, dim=1, power_iters=2)
            static_ret = self.get_static_graph_result(
                feed={
                    'Weight': fluid.create_lod_tensor(
                        data=input, recursive_seq_lens=[[1, 1]], place=place),
                },
                fetch_list=[ret],
                with_lod=True)[0]

        with self.static_graph():
            Weight = fluid.layers.data(
                name='Weight',
                shape=shape,
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            spectralNorm = nn.SpectralNorm('SpectralNorm', dim=1, power_iters=2)
            ret = spectralNorm(Weight)
            static_ret2 = self.get_static_graph_result(
                feed={
                    'Weight': fluid.create_lod_tensor(
                        data=input, recursive_seq_lens=[[1, 1]], place=place)
                },
                fetch_list=[ret],
                with_lod=True)[0]

        with self.dynamic_graph():
            spectralNorm = nn.SpectralNorm('SpectralNorm', dim=1, power_iters=2)
            dy_ret = spectralNorm(base.to_variable(input))

        self.assertTrue(np.allclose(static_ret, dy_ret.numpy()))
        self.assertTrue(np.allclose(static_ret, static_ret2))

    def test_tree_conv(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        adj_array = [1, 2, 1, 3, 1, 4, 1, 5, 2, 6, 2, 7, 2, 8, 4, 9, 4, 10]
        adj = np.array(adj_array).reshape((1, 9, 2)).astype('int32')
        adj = np.tile(adj, (1, 1, 1))
        vectors = np.random.random((1, 10, 5)).astype('float32')
        with self.static_graph():
            NodesVector = fluid.layers.data(
                name='NodesVector',
                shape=(1, 10, 5),
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            EdgeSet = fluid.layers.data(
                name='EdgeSet',
                shape=(1, 9, 2),
                dtype='int32',
                lod_level=1,
                append_batch_size=False)
            ret = layers.tree_conv(
                nodes_vector=NodesVector,
                edge_set=EdgeSet,
                output_size=6,
                num_filters=1,
                max_depth=2)
            static_ret = self.get_static_graph_result(
                feed={
                    'NodesVector': fluid.create_lod_tensor(
                        data=vectors, recursive_seq_lens=[[1]], place=place),
                    'EdgeSet': fluid.create_lod_tensor(
                        data=adj, recursive_seq_lens=[[1]], place=place)
                },
                fetch_list=[ret],
                with_lod=False)[0]

        with self.static_graph():
            NodesVector = fluid.layers.data(
                name='NodesVector',
                shape=(1, 10, 5),
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            EdgeSet = fluid.layers.data(
                name='EdgeSet',
                shape=(1, 9, 2),
                dtype='int32',
                lod_level=1,
                append_batch_size=False)
            treeConv = nn.TreeConv(
                'TreeConv', output_size=6, num_filters=1, max_depth=2)
            ret = treeConv(NodesVector, EdgeSet)
            static_ret2 = self.get_static_graph_result(
                feed={
                    'NodesVector': fluid.create_lod_tensor(
                        data=vectors, recursive_seq_lens=[[1]], place=place),
                    'EdgeSet': fluid.create_lod_tensor(
                        data=adj, recursive_seq_lens=[[1]], place=place)
                },
                fetch_list=[ret],
                with_lod=False)[0]

        with self.dynamic_graph():
            treeConv = nn.TreeConv(
                'SpectralNorm', output_size=6, num_filters=1, max_depth=2)
            dy_ret = treeConv(base.to_variable(vectors), base.to_variable(adj))

        self.assertTrue(np.allclose(static_ret, static_ret2))
        self.assertTrue(np.allclose(static_ret, dy_ret.numpy()))

    def test_conv3d_transpose(self):
        input_array = np.arange(0, 48).reshape(
            [2, 3, 2, 2, 2]).astype('float32')

        with self.static_graph():
            img = layers.data(name='pixel', shape=[3, 2, 2, 2], dtype='float32')
            out = layers.conv3d_transpose(
                input=img, num_filters=12, filter_size=12, use_cudnn=False)
            static_rlt = self.get_static_graph_result(
                feed={'pixel': input_array}, fetch_list=[out])[0]
        with self.static_graph():
            img = layers.data(name='pixel', shape=[3, 2, 2, 2], dtype='float32')
            conv3d_transpose = nn.Conv3DTranspose(
                'Conv3DTranspose',
                num_filters=12,
                filter_size=12,
                use_cudnn=False)
            out = conv3d_transpose(img)
            static_rlt2 = self.get_static_graph_result(
                feed={'pixel': input_array}, fetch_list=[out])[0]
        with self.dynamic_graph():
            conv3d_transpose = nn.Conv3DTranspose(
                'Conv3DTranspose',
                num_filters=12,
                filter_size=12,
                use_cudnn=False)
            dy_rlt = conv3d_transpose(base.to_variable(input_array))
        self.assertTrue(np.allclose(static_rlt2, static_rlt))
        self.assertTrue(np.allclose(dy_rlt.numpy(), static_rlt))


class TestBook(unittest.TestCase):
    def test_fit_a_line(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            x = layers.data(name='x', shape=[13], dtype='float32')
            y_predict = layers.fc(input=x, size=1, act=None)
            y = layers.data(name='y', shape=[1], dtype='float32')
            cost = layers.square_error_cost(input=y_predict, label=y)
            avg_cost = layers.mean(cost)
            self.assertIsNotNone(avg_cost)

        print(str(program))

    def test_recognize_digits_mlp(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            # Change g_program, so the rest layers use `g_program`
            images = layers.data(name='pixel', shape=[784], dtype='float32')
            label = layers.data(name='label', shape=[1], dtype='int32')
            hidden1 = layers.fc(input=images, size=128, act='relu')
            hidden2 = layers.fc(input=hidden1, size=64, act='relu')
            predict = layers.fc(input=[hidden2, hidden1],
                                size=10,
                                act='softmax',
                                param_attr=["sftmax.w1", "sftmax.w2"])
            cost = layers.cross_entropy(input=predict, label=label)
            avg_cost = layers.mean(cost)
            self.assertIsNotNone(avg_cost)

        print(str(program))

    def test_simple_conv2d(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            images = layers.data(
                name='pixel', shape=[3, 48, 48], dtype='float32')
            layers.conv2d(input=images, num_filters=3, filter_size=[4, 4])

        print(str(program))

    def test_conv2d_transpose(self):
        program = Program()
        with program_guard(program):
            img = layers.data(name='pixel', shape=[3, 2, 2], dtype='float32')
            layers.conv2d_transpose(input=img, num_filters=10, output_size=28)
        print(str(program))

    def test_recognize_digits_conv(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            images = layers.data(
                name='pixel', shape=[1, 28, 28], dtype='float32')
            label = layers.data(name='label', shape=[1], dtype='int32')
            conv_pool_1 = nets.simple_img_conv_pool(
                input=images,
                filter_size=5,
                num_filters=2,
                pool_size=2,
                pool_stride=2,
                act="relu")
            conv_pool_2 = nets.simple_img_conv_pool(
                input=conv_pool_1,
                filter_size=5,
                num_filters=4,
                pool_size=2,
                pool_stride=2,
                act="relu")

            predict = layers.fc(input=conv_pool_2, size=10, act="softmax")
            cost = layers.cross_entropy(input=predict, label=label)
            avg_cost = layers.mean(cost)

        print(str(program))

    def test_word_embedding(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            dict_size = 10000
            embed_size = 32
            first_word = layers.data(name='firstw', shape=[1], dtype='int64')
            second_word = layers.data(name='secondw', shape=[1], dtype='int64')
            third_word = layers.data(name='thirdw', shape=[1], dtype='int64')
            forth_word = layers.data(name='forthw', shape=[1], dtype='int64')
            next_word = layers.data(name='nextw', shape=[1], dtype='int64')

            embed_first = layers.embedding(
                input=first_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w')
            embed_second = layers.embedding(
                input=second_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w')

            embed_third = layers.embedding(
                input=third_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w')
            embed_forth = layers.embedding(
                input=forth_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w')

            concat_embed = layers.concat(
                input=[embed_first, embed_second, embed_third, embed_forth],
                axis=1)

            hidden1 = layers.fc(input=concat_embed, size=256, act='sigmoid')
            predict_word = layers.fc(input=hidden1,
                                     size=dict_size,
                                     act='softmax')
            cost = layers.cross_entropy(input=predict_word, label=next_word)
            avg_cost = layers.mean(cost)
            self.assertIsNotNone(avg_cost)

        print(str(program))

    def test_linear_chain_crf(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            label_dict_len = 10
            images = layers.data(name='pixel', shape=[784], dtype='float32')
            label = layers.data(name='label', shape=[1], dtype='int32')
            hidden = layers.fc(input=images, size=128)
            crf = layers.linear_chain_crf(
                input=hidden, label=label, param_attr=ParamAttr(name="crfw"))
            crf_decode = layers.crf_decoding(
                input=hidden, param_attr=ParamAttr(name="crfw"))
            layers.chunk_eval(
                input=crf_decode,
                label=label,
                chunk_scheme="IOB",
                num_chunk_types=(label_dict_len - 1) // 2)
            self.assertFalse(crf is None)
            self.assertFalse(crf_decode is None)

        print(str(program))

    def test_sigmoid_cross_entropy(self):
        program = Program()
        with program_guard(program):
            dat = layers.data(name='data', shape=[10], dtype='float32')
            lbl = layers.data(name='label', shape=[10], dtype='float32')
            ignore_index = -1
            self.assertIsNotNone(
                layers.sigmoid_cross_entropy_with_logits(
                    x=dat, label=lbl, ignore_index=ignore_index))
        print(str(program))

    def test_hsigmoid(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[2], dtype='float32')
            y = layers.data(name='y', shape=[2], dtype='int64')
            self.assertIsNotNone(
                layers.hsigmoid(
                    input=x, label=y, num_classes=2))
        print(str(program))

        # test hsigmod with custom tree structure
        program2 = Program()
        with program_guard(program2):
            x2 = layers.data(name='x2', shape=[4, 8], dtype='float32')
            y2 = layers.data(name='y2', shape=[4], dtype='int64')
            path_table = layers.data(
                name='path_table', shape=[4, 6], dtype='int64')
            path_code = layers.data(
                name='path_code', shape=[4, 6], dtype='int64')
            self.assertIsNotNone(
                layers.hsigmoid(
                    input=x2,
                    label=y2,
                    num_classes=6,
                    path_table=path_table,
                    path_code=path_code,
                    is_custom=True))
            print(str(program2))

    def test_sequence_expand(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[10], dtype='float32')
            y = layers.data(
                name='y', shape=[10, 20], dtype='float32', lod_level=2)
            self.assertIsNotNone(layers.sequence_expand(x=x, y=y, ref_level=1))
        print(str(program))

    def test_sequence_unpad(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[10, 5], dtype='float32')
            length = layers.data(name='length', shape=[1], dtype='int64')
            self.assertIsNotNone(layers.sequence_unpad(x=x, length=length))
        print(str(program))

    def test_pool2d(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[3, 224, 224], dtype='float32')
            self.assertIsNotNone(
                layers.pool2d(
                    x,
                    pool_size=[5, 3],
                    pool_stride=[1, 2],
                    pool_padding=(2, 1)))

    def test_adaptive_pool2d(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[3, 224, 224], dtype='float32')
            self.assertIsNotNone(
                layers.adaptive_pool2d(
                    x, [3, 3], pool_type='avg'))
            pool, mask = layers.adaptive_pool2d(x, [3, 3], require_index=True)
            self.assertIsNotNone(pool)
            self.assertIsNotNone(mask)
            self.assertIsNotNone(layers.adaptive_pool2d(x, 3, pool_type='avg'))
            pool, mask = layers.adaptive_pool2d(x, 3, require_index=True)
            self.assertIsNotNone(pool)
            self.assertIsNotNone(mask)

    def test_adaptive_pool3d(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[3, 244, 224, 224], dtype='float32')
            self.assertIsNotNone(
                layers.adaptive_pool3d(
                    x, [3, 3, 3], pool_type='avg'))
            pool, mask = layers.adaptive_pool3d(
                x, [3, 3, 3], require_index=True)
            self.assertIsNotNone(pool)
            self.assertIsNotNone(mask)
            self.assertIsNotNone(layers.adaptive_pool3d(x, 3, pool_type='avg'))
            pool, mask = layers.adaptive_pool3d(x, 3, require_index=True)
            self.assertIsNotNone(pool)
            self.assertIsNotNone(mask)

    def test_lstm_unit(self):
        program = Program()
        with program_guard(program):
            x_t_data = layers.data(
                name='x_t_data', shape=[10, 10], dtype='float32')
            x_t = layers.fc(input=x_t_data, size=10)
            prev_hidden_data = layers.data(
                name='prev_hidden_data', shape=[10, 30], dtype='float32')
            prev_hidden = layers.fc(input=prev_hidden_data, size=30)
            prev_cell_data = layers.data(
                name='prev_cell', shape=[10, 30], dtype='float32')
            prev_cell = layers.fc(input=prev_cell_data, size=30)
            self.assertIsNotNone(
                layers.lstm_unit(
                    x_t=x_t, hidden_t_prev=prev_hidden, cell_t_prev=prev_cell))
        print(str(program))

    def test_dynamic_lstmp(self):
        program = Program()
        with program_guard(program):
            hidden_dim, proj_dim = 16, 8
            seq_data = layers.data(
                name='seq_data', shape=[10, 10], dtype='float32', lod_level=1)
            fc_out = layers.fc(input=seq_data, size=4 * hidden_dim)
            self.assertIsNotNone(
                layers.dynamic_lstmp(
                    input=fc_out, size=4 * hidden_dim, proj_size=proj_dim))
        print(str(program))

    def test_sequence_softmax(self):
        program = Program()
        with program_guard(program):
            seq_data = layers.data(
                name='seq_data', shape=[10, 10], dtype='float32', lod_level=1)
            seq = layers.fc(input=seq_data, size=20)
            self.assertIsNotNone(layers.sequence_softmax(seq))
        print(str(program))

    def test_softmax(self):
        program = Program()
        with program_guard(program):
            data = layers.data(name='data', shape=[10], dtype='float32')
            hid = layers.fc(input=data, size=20)
            self.assertIsNotNone(layers.softmax(hid, axis=1))
        print(str(program))

    def test_space_to_depth(self):
        program = Program()
        with program_guard(program):
            data = layers.data(
                name='data',
                shape=[32, 9, 6, 6],
                append_batch_size=False,
                dtype='float32')
            self.assertIsNotNone(layers.space_to_depth(data, 3))
        print(str(program))

    def test_sequence_unsqueeze(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[8, 2], dtype='float32')
            out = layers.unsqueeze(input=x, axes=[1])
            self.assertIsNotNone(out)
        print(str(program))

    def test_squeeze(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[1, 1, 4], dtype='float32')
            out = layers.squeeze(input=x, axes=[2])
            self.assertIsNotNone(out)
        print(str(program))

    def test_lrn(self):
        program = Program()
        with program_guard(program):
            data = layers.data(name='data', shape=[6, 2, 2], dtype='float32')
            self.assertIsNotNone(layers.lrn(data))
        print(str(program))

    def test_get_places(self):
        program = Program()
        with program_guard(program):
            x = get_places(device_count=4)
            self.assertIsNotNone(x)
        print(str(program))

    def test_sequence_reshape(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[8], dtype='float32', lod_level=1)
            out = layers.sequence_reshape(input=x, new_dim=16)
            self.assertIsNotNone(out)
        print(str(program))

    def test_im2sequence(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[3, 128, 128], dtype='float32')
            y = layers.data(name='y', shape=[], dtype='float32')
            output = layers.im2sequence(
                input=x,
                input_image_size=y,
                stride=[1, 1],
                filter_size=[2, 2],
                out_stride=[1, 1])
            self.assertIsNotNone(output)
        print(str(program))

    def test_sampled_softmax_with_cross_entropy(self):
        program = Program()
        with program_guard(program):
            logits = layers.data(name='Logits', shape=[256], dtype='float64')
            label = layers.data(name='Label', shape=[1], dtype='int64')
            num_samples = 25
            output = layers.sampled_softmax_with_cross_entropy(logits, label,
                                                               num_samples)
            self.assertIsNotNone(output)
        print(str(program))

    @decorators.prog_scope()
    def test_nce(self):
        window_size = 5
        words = []
        for i in range(window_size):
            words.append(
                layers.data(
                    name='word_{0}'.format(i), shape=[1], dtype='int64'))

        dict_size = 10000
        label_word = int(window_size // 2) + 1

        embs = []
        for i in range(window_size):
            if i == label_word:
                continue

            emb = layers.embedding(
                input=words[i],
                size=[dict_size, 32],
                param_attr='emb.w',
                is_sparse=True)

            embs.append(emb)

        embs = layers.concat(input=embs, axis=1)
        loss = layers.nce(input=embs,
                          label=words[label_word],
                          num_total_classes=dict_size,
                          param_attr='nce.w',
                          bias_attr='nce.b')
        avg_loss = layers.mean(loss)
        self.assertIsNotNone(avg_loss)
        print(str(default_main_program()))

    def test_row_conv(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[16], dtype='float32', lod_level=1)
            out = layers.row_conv(input=x, future_context_size=2)
            self.assertIsNotNone(out)
        print(str(program))

    def test_multiplex(self):
        program = Program()
        with program_guard(program):
            x1 = layers.data(name='x1', shape=[4], dtype='float32')
            x2 = layers.data(name='x2', shape=[4], dtype='float32')
            index = layers.data(name='index', shape=[1], dtype='int32')
            out = layers.multiplex(inputs=[x1, x2], index=index)
            self.assertIsNotNone(out)
        print(str(program))

    def test_softmax_with_cross_entropy(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[16], dtype='float32')
            y = layers.data(name='label', shape=[1], dtype='int64')
            loss, softmax = layers.softmax_with_cross_entropy(
                x, y, return_softmax=True)
            self.assertIsNotNone(loss)
            self.assertIsNotNone(softmax)
            loss = layers.softmax_with_cross_entropy(x, y)
            self.assertIsNotNone(loss)
        print(str(program))

    def test_smooth_l1(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[4], dtype='float32')
            y = layers.data(name='label', shape=[4], dtype='float32')
            loss = layers.smooth_l1(x, y)
            self.assertIsNotNone(loss)
        print(str(program))

    def test_scatter(self):
        program = Program()
        with program_guard(program):
            x = layers.data(
                name='x',
                shape=[3, 3],
                append_batch_size=False,
                dtype='float32')
            idx = layers.data(
                name='idx', shape=[2], append_batch_size=False, dtype='int32')
            updates = layers.data(
                name='updates',
                shape=[2, 3],
                append_batch_size=False,
                dtype='float32')
            out = layers.scatter(input=x, index=idx, updates=updates)
            self.assertIsNotNone(out)
        print(str(program))

    def test_sequence_scatter(self):
        program = Program()
        with program_guard(program):
            x = layers.data(
                name='x',
                shape=[3, 6],
                append_batch_size=False,
                dtype='float32')
            idx = layers.data(
                name='idx',
                shape=[12, 1],
                append_batch_size=False,
                dtype='int32',
                lod_level=1)
            updates = layers.data(
                name='updates',
                shape=[12, 1],
                append_batch_size=False,
                dtype='float32',
                lod_level=1)
            out = layers.sequence_scatter(input=x, index=idx, updates=updates)
            self.assertIsNotNone(out)
        print(str(program))

    def test_sequence_slice(self):
        program = Program()
        with program_guard(program):
            import numpy as np
            seqs = layers.data(
                name='x', shape=[10, 5], dtype='float32', lod_level=1)
            offset = layers.assign(input=np.array([[0, 1]]).astype('int32'))
            length = layers.assign(input=np.array([[2, 1]]).astype('int32'))
            out = layers.sequence_slice(
                input=seqs, offset=offset, length=length)
            self.assertIsNotNone(out)
        print(str(program))

    def test_lod_reset(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[10], dtype='float32')
            y = layers.data(
                name='y', shape=[10, 20], dtype='float32', lod_level=2)
            z = layers.lod_reset(x=x, y=y)
            self.assertTrue(z.lod_level == 2)
            # case 2
            lod_tensor_in = layers.data(name='lod_in', shape=[1], dtype='int64')
            z = layers.lod_reset(x=x, y=lod_tensor_in)
            self.assertTrue(z.lod_level == 1)
            # case 3
            z = layers.lod_reset(x=x, target_lod=[1, 2, 3])
            self.assertTrue(z.lod_level == 1)
        print(str(program))

    def test_label_smooth(self):
        program = Program()
        with program_guard(program):
            label = layers.data(name="label", shape=[1], dtype="float32")
            one_hot_label = layers.one_hot(input=label, depth=10)
            smooth_label = layers.label_smooth(
                label=one_hot_label, epsilon=0.1, dtype="float32")
            self.assertIsNotNone(smooth_label)
        print(str(program))

    def test_topk(self):
        program = Program()
        with program_guard(program):
            data = layers.data(name="label", shape=[200], dtype="float32")
            values, indices = layers.topk(data, k=5)
            self.assertIsNotNone(values)
            self.assertIsNotNone(indices)
        print(str(program))

    def test_roi_pool(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name="x", shape=[256, 30, 30], dtype="float32")
            rois = layers.data(
                name="rois", shape=[4], dtype="float32", lod_level=1)
            output = layers.roi_pool(x, rois, 7, 7, 0.6)
            self.assertIsNotNone(output)
        print(str(program))

    def test_psroi_pool(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name="x", shape=[245, 30, 30], dtype="float32")
            rois = layers.data(
                name="rois", shape=[4], dtype="float32", lod_level=1)
            output = layers.psroi_pool(x, rois, 5, 0.25, 7, 7)
            self.assertIsNotNone(output)
        print(str(program))

    def test_roi_align(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name="x", shape=[256, 30, 30], dtype="float32")
            rois = layers.data(
                name="rois", shape=[4], dtype="float32", lod_level=1)
            output = layers.roi_align(x, rois, 14, 14, 0.5, 2)
            self.assertIsNotNone(output)
        print(str(program))

    def test_resize_bilinear(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[3, 9, 6], dtype="float32")
            output = layers.resize_bilinear(x, out_shape=[12, 12])
            self.assertIsNotNone(output)
            output = layers.resize_bilinear(x, scale=3)
            self.assertIsNotNone(output)
        print(str(program))

    def test_resize_nearest(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[3, 9, 6], dtype="float32")
            output = layers.resize_nearest(x, out_shape=[12, 12])
            self.assertIsNotNone(output)
            output = layers.resize_nearest(x, scale=3)
            self.assertIsNotNone(output)
        print(str(program))

    def test_polygon_box_transform(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[8, 4, 4], dtype="float32")
            output = layers.polygon_box_transform(input=x)
            self.assertIsNotNone(output)
        print(str(program))

    def test_l2_normalize(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[8, 7, 10], dtype="float32")
            output = layers.l2_normalize(x, axis=1)

    def test_maxout(self):
        program = Program()
        with program_guard(program):
            data = layers.data(name='x', shape=[8, 6, 6], dtype="float32")
            output = layers.maxout(x=data, groups=2)
            self.assertIsNotNone(output)
        print(str(program))

    def test_crop(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[3, 5], dtype="float32")
            y = layers.data(name='y', shape=[2, 3], dtype="float32")
            output = layers.crop(x, shape=y)
            self.assertIsNotNone(output)
        print(str(program))

    def test_mean_iou(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[16], dtype='float32')
            y = layers.data(name='label', shape=[1], dtype='int64')
            iou = layers.mean_iou(x, y, 2)
            self.assertIsNotNone(iou)
        print(str(program))

    def test_argsort(self):
        program = Program()
        with program_guard(program):
            data = layers.data(name='x', shape=[2, 3, 3], dtype="float32")
            out, ids = layers.argsort(input=data, axis=1)
            self.assertIsNotNone(out)
            self.assertIsNotNone(ids)
        print(str(program))

    def test_rank_loss(self):
        program = Program()
        with program_guard(program):
            label = layers.data(
                name='label',
                append_batch_size=False,
                shape=[16, 1],
                dtype="float32")
            left = layers.data(
                name='left',
                append_batch_size=False,
                shape=[16, 1],
                dtype="float32")
            right = layers.data(
                name='right',
                append_batch_size=False,
                shape=[16, 1],
                dtype="float32")
            out = layers.rank_loss(label, left, right, name="rank_loss")
            self.assertIsNotNone(out)
        print(str(program))

    def test_flatten(self):
        program = Program()
        with program_guard(program):
            x = layers.data(
                name='x',
                append_batch_size=False,
                shape=[4, 4, 3],
                dtype="float32")
            out = layers.flatten(x, axis=1, name="flatten")
            self.assertIsNotNone(out)

    def test_shape(self):
        program = Program()
        with program_guard(program):
            input = layers.data(
                name="input", shape=[3, 100, 100], dtype="float32")
            out = layers.shape(input)
            self.assertIsNotNone(out)
        print(str(program))

    def test_pad2d(self):
        program = Program()
        with program_guard(program):
            input = layers.data(
                name="input", shape=[3, 100, 100], dtype="float32")
            paddings = layers.fill_constant(shape=[4], dtype='int32', value=1)
            out = layers.pad2d(
                input,
                paddings=[1, 2, 3, 4],
                mode='reflect',
                data_format='NCHW',
                name="shape")
            out_1 = layers.pad2d(
                input,
                paddings=paddings,
                mode='reflect',
                data_format='NCHW',
                name="shape")
            self.assertIsNotNone(out)
            self.assertIsNotNone(out_1)
        print(str(program))

    def test_prelu(self):
        program = Program()
        with program_guard(program):
            input = layers.data(
                name="input", shape=[5, 200, 100, 100], dtype="float32")
            mode = 'channel'
            out = layers.prelu(
                input,
                mode,
                param_attr=ParamAttr(initializer=Constant(1.0)),
                name='prelu')
            self.assertIsNotNone(out)
        print(str(program))

    def test_brelu(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.brelu(input, t_min=1.0, t_max=20.0, name='brelu')
            self.assertIsNotNone(out)
        print(str(program))

    def test_leaky_relu(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.leaky_relu(input, alpha=0.1, name='leaky_relu')
            self.assertIsNotNone(out)
        print(str(program))

    def test_soft_relu(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.soft_relu(input, threshold=30.0, name='soft_relu')
            self.assertIsNotNone(out)
        print(str(program))

    def test_sigmoid(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.sigmoid(input, name='sigmoid')
            self.assertIsNotNone(out)
        print(str(program))

    def test_logsigmoid(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.logsigmoid(input, name='logsigmoid')
            self.assertIsNotNone(out)
        print(str(program))

    def test_exp(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.exp(input, name='exp')
            self.assertIsNotNone(out)
        print(str(program))

    def test_tanh(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.tanh(input, name='tanh')
            self.assertIsNotNone(out)
        print(str(program))

    def test_tanh_shrink(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.tanh_shrink(input, name='tanh_shrink')
            self.assertIsNotNone(out)
        print(str(program))

    def test_sqrt(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.sqrt(input, name='sqrt')
            self.assertIsNotNone(out)
        print(str(program))

    def test_abs(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.abs(input, name='abs')
            self.assertIsNotNone(out)
        print(str(program))

    def test_ceil(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.ceil(input, name='ceil')
            self.assertIsNotNone(out)
        print(str(program))

    def test_floor(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.floor(input, name='floor')
            self.assertIsNotNone(out)
        print(str(program))

    def test_cos(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.cos(input, name='cos')
            self.assertIsNotNone(out)
        print(str(program))

    def test_sin(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.sin(input, name='sin')
            self.assertIsNotNone(out)
        print(str(program))

    def test_round(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.round(input, name='round')
            self.assertIsNotNone(out)
        print(str(program))

    def test_reciprocal(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.reciprocal(input, name='reciprocal')
            self.assertIsNotNone(out)
        print(str(program))

    def test_square(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.square(input, name='square')
            self.assertIsNotNone(out)
        print(str(program))

    def test_softplus(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.softplus(input, name='softplus')
            self.assertIsNotNone(out)
        print(str(program))

    def test_softsign(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.softsign(input, name='softsign')
            self.assertIsNotNone(out)
        print(str(program))

    def test_roi_perspective_transform(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name="x", shape=[256, 30, 30], dtype="float32")
            rois = layers.data(
                name="rois", shape=[8], dtype="float32", lod_level=1)
            output = layers.roi_perspective_transform(x, rois, 7, 7, 0.6)
            self.assertIsNotNone(output)
        print(str(program))

    def test_sequence_enumerate(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name="input", shape=[1], dtype='int32', lod_level=1)
            out = layers.sequence_enumerate(input=x, win_size=2, pad_value=0)
        print(str(program))

    def test_cross_entropy(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name="x", shape=[30, 10], dtype="float32")
            label = layers.data(name="label", shape=[30, 1], dtype="int32")
            mode = 'channel'
            out = layers.cross_entropy(x, label, False, 4)
            self.assertIsNotNone(out)

    def test_bpr_loss(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name="x", shape=[30, 10], dtype="float32")
            label = layers.data(name="label", shape=[30, 1], dtype="int32")
            out = layers.bpr_loss(x, label)
            self.assertIsNotNone(out)
        print(str(program))

    def test_expand(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name="input", shape=[10], dtype='int32')
            out = layers.expand(x, [1, 2])
        print(str(program))

    def test_uniform_random_batch_size_like(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[13, 11], dtype='float32')
            out = layers.uniform_random_batch_size_like(input, [-1, 11])
            self.assertIsNotNone(out)
        print(str(program))

    def test_gaussian_random(self):
        program = Program()
        with program_guard(program):
            out = layers.gaussian_random(shape=[20, 30])
            self.assertIsNotNone(out)
        print(str(program))

    def test_sampling_id(self):
        program = Program()
        with program_guard(program):
            x = layers.data(
                name="X",
                shape=[13, 11],
                dtype='float32',
                append_batch_size=False)

            out = layers.sampling_id(x)
            self.assertIsNotNone(out)
        print(str(program))

    def test_gaussian_random_batch_size_like(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[13, 11], dtype='float32')

            out = layers.gaussian_random_batch_size_like(
                input, shape=[-1, 11], mean=1.0, std=2.0)
            self.assertIsNotNone(out)
        print(str(program))

    def test_sum(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[13, 11], dtype='float32')

            out = layers.sum(input)
            self.assertIsNotNone(out)
        print(str(program))

    def test_slice(self):
        starts = [1, 0, 2]
        ends = [3, 3, 4]
        axes = [0, 1, 2]

        program = Program()
        with program_guard(program):
            input = layers.data(
                name="input", shape=[3, 4, 5, 6], dtype='float32')

            out = layers.slice(input, axes=axes, starts=starts, ends=ends)

    def test_softshrink(self):
        program = Program()
        with program_guard(program):
            input = layers.data(name="input", shape=[16], dtype="float32")
            out = layers.softshrink(input, name='softshrink')
            self.assertIsNotNone(out)
        print(str(program))

    def iou_similarity(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name="x", shape=[16], dtype="float32")
            y = layers.data(name="y", shape=[16], dtype="float32")
            out = layers.iou_similarity(x, y, name='iou_similarity')
            self.assertIsNotNone(out)
        print(str(program))

    def test_grid_sampler(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[3, 5, 7], dtype='float32')
            grid = layers.data(name='grid', shape=[5, 7, 2], dtype='float32')
            out = layers.grid_sampler(x, grid)
            self.assertIsNotNone(out)
        print(str(program))

    def test_affine_grid(self):
        program = Program()
        with program_guard(program):
            data = layers.data(name='data', shape=[2, 3, 3], dtype="float32")
            out, ids = layers.argsort(input=data, axis=1)

            theta = layers.data(name="theta", shape=[2, 3], dtype="float32")
            out_shape = layers.data(
                name="out_shape", shape=[-1], dtype="float32")
            data_0 = layers.affine_grid(theta, out_shape)
            data_1 = layers.affine_grid(theta, [5, 3, 28, 28])

            self.assertIsNotNone(data_0)
            self.assertIsNotNone(data_1)
        print(str(program))

    def test_bilinear_tensor_product_layer(self):
        program = Program()
        with program_guard(program):
            data = layers.data(name='data', shape=[4], dtype="float32")

            theta = layers.data(name="theta", shape=[5], dtype="float32")
            out = layers.bilinear_tensor_product(data, theta, 6)

        print(str(program))

    def test_batch_norm(self):
        program = Program()
        with program_guard(program):
            data = layers.data(
                name='data', shape=[32, 128, 128], dtype="float32")
            out = layers.batch_norm(data)

        print(str(program))

    def test_range(self):
        program = Program()
        with program_guard(program):
            layers.range(0, 10, 2, 'int32')
            layers.range(0.1, 10.0, 0.2, 'float32')

        print(str(program))

    def test_spectral_norm(self):
        program = Program()
        with program_guard(program):
            weight = layers.data(
                name='weight',
                shape=[2, 3, 32, 32],
                dtype="float32",
                append_batch_size=False)
            out = layers.spectral_norm(weight, dim=1, power_iters=1)
            self.assertIsNotNone(out)

    def test_kldiv_loss(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[32, 128, 128], dtype="float32")
            target = layers.data(
                name='target', shape=[32, 128, 128], dtype="float32")
            loss = layers.kldiv_loss(x=x, target=target, reduction='batchmean')
            self.assertIsNotNone(loss)

        print(str(program))

    def test_temporal_shift(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name="X", shape=[16, 4, 4], dtype="float32")
            out = layers.temporal_shift(x, seg_num=4, shift_ratio=0.2)
            self.assertIsNotNone(out)
        print(str(program))

    def test_shuffle_channel(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name="X", shape=[16, 4, 4], dtype="float32")
            out = layers.shuffle_channel(x, group=4)
            self.assertIsNotNone(out)
        print(str(program))

    def test_pixel_shuffle(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name="X", shape=[9, 4, 4], dtype="float32")
            out = layers.pixel_shuffle(x, upscale_factor=3)
            self.assertIsNotNone(out)
        print(str(program))

    def test_fsp(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name="X", shape=[16, 4, 4], dtype="float32")
            y = layers.data(name="Y", shape=[8, 4, 4], dtype="float32")
            out = layers.fsp_matrix(x, y)
            self.assertIsNotNone(out)
        print(str(program))


if __name__ == '__main__':
    unittest.main()

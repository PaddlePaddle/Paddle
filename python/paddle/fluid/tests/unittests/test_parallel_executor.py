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

import numpy
import unittest

import paddle.fluid as fluid
import paddle
import paddle.dataset.mnist as mnist
import paddle.dataset.wmt16 as wmt16


def simple_fc_net():
    reader = fluid.layers.open_recordio_file(
        filename='./mnist.recordio',
        shapes=[[-1, 784], [-1, 1]],
        lod_levels=[0, 0],
        dtypes=['float32', 'int64'])
    img, label = fluid.layers.read_file(reader)
    hidden = img
    for _ in xrange(4):
        hidden = fluid.layers.fc(
            hidden,
            size=200,
            act='tanh',
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=1.0)))
    prediction = fluid.layers.fc(hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = fluid.layers.mean(loss)
    return loss


def fc_with_batchnorm():
    reader = fluid.layers.open_recordio_file(
        filename='./mnist.recordio',
        shapes=[[-1, 784], [-1, 1]],
        lod_levels=[0, 0],
        dtypes=['float32', 'int64'])
    img, label = fluid.layers.read_file(reader)
    hidden = img
    for _ in xrange(1):
        hidden = fluid.layers.fc(
            hidden,
            size=200,
            act='tanh',
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=1.0)))

        hidden = fluid.layers.batch_norm(input=hidden)

    prediction = fluid.layers.fc(hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = fluid.layers.mean(loss)
    return loss


def squeeze_excitation(input, num_channels, reduction_ratio):
    # pool = fluid.layers.pool2d(
    #    input=input, pool_size=0, pool_type='avg', global_pooling=True)
    conv = input
    shape = conv.shape
    reshape = fluid.layers.reshape(
        x=conv, shape=[-1, shape[1], shape[2] * shape[3]])
    pool = fluid.layers.reduce_mean(input=reshape, dim=2)

    squeeze = fluid.layers.fc(input=pool,
                              size=num_channels / reduction_ratio,
                              act='relu')
    excitation = fluid.layers.fc(input=squeeze,
                                 size=num_channels,
                                 act='sigmoid')
    scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
    return scale


def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1,
                  act=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=(filter_size - 1) / 2,
        groups=groups,
        act=None,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv, act=act, momentum=0.1)


def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]
    if ch_in != ch_out:
        if stride == 1:
            filter_size = 1
        else:
            filter_size = 3
        return conv_bn_layer(input, ch_out, filter_size, stride)
    else:
        return input


def bottleneck_block(input, num_filters, stride, cardinality, reduction_ratio):
    # The number of first 1x1 convolutional channels for each bottleneck build block
    # was halved to reduce the compution cost.
    conv0 = conv_bn_layer(
        input=input, num_filters=num_filters, filter_size=1, act='relu')
    conv1 = conv_bn_layer(
        input=conv0,
        num_filters=num_filters * 2,
        filter_size=3,
        stride=stride,
        groups=cardinality,
        act='relu')
    conv2 = conv_bn_layer(
        input=conv1, num_filters=num_filters * 2, filter_size=1, act=None)
    scale = squeeze_excitation(
        input=conv2,
        num_channels=num_filters * 2,
        reduction_ratio=reduction_ratio)

    short = shortcut(input, num_filters * 2, stride)

    return fluid.layers.elementwise_add(x=short, y=scale, act='relu')


def SE_ResNeXt152(batch_size=4):
    img = fluid.layers.fill_constant(
        shape=[batch_size, 3, 224, 224], dtype='float32', value=0.0)
    label = fluid.layers.fill_constant(
        shape=[batch_size, 1], dtype='int64', value=0.0)

    conv = conv_bn_layer(
        input=img, num_filters=64, filter_size=3, stride=2, act='relu')
    conv = conv_bn_layer(
        input=conv, num_filters=64, filter_size=3, stride=1, act='relu')
    conv = conv_bn_layer(
        input=conv, num_filters=128, filter_size=3, stride=1, act='relu')
    conv = fluid.layers.pool2d(
        input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

    cardinality = 64
    reduction_ratio = 16
    depth = [3, 8, 36, 3]
    num_filters = [128, 256, 512, 1024]

    for block in range(len(depth)):
        for i in range(depth[block]):
            conv = bottleneck_block(
                input=conv,
                num_filters=num_filters[block],
                stride=2 if i == 0 and block != 0 else 1,
                cardinality=cardinality,
                reduction_ratio=reduction_ratio)

    shape = conv.shape
    reshape = fluid.layers.reshape(
        x=conv, shape=[-1, shape[1], shape[2] * shape[3]])
    pool = fluid.layers.reduce_mean(input=reshape, dim=2)
    dropout = fluid.layers.dropout(x=pool, dropout_prob=0.2)
    # Classifier layer:
    prediction = fluid.layers.fc(input=dropout, size=1000, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = fluid.layers.mean(loss)
    return loss


import time


class TestParallelExecutorBase(unittest.TestCase):
    def check_network_convergence(self,
                                  method,
                                  memory_opt=True,
                                  iter=10,
                                  batch_size=None,
                                  allow_op_delay=False):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = method()
            adam = fluid.optimizer.Adam()
            adam.minimize(loss)
            if memory_opt:
                fluid.memory_optimize(main)

            exe = fluid.ParallelExecutor(
                loss_name=loss.name,
                use_cuda=True,
                allow_op_delay=allow_op_delay)
            if batch_size is not None:
                batch_size *= fluid.core.get_cuda_device_count()
            begin = time.time()
            first_loss, = exe.run([loss.name])
            first_loss = numpy.array(first_loss)

            for i in xrange(iter):
                exe.run([])

            last_loss, = exe.run([loss.name])
            end = time.time()

            if batch_size is not None:
                print "%.4f Instance per second" % (
                    (batch_size * iter + 2) / (end - begin))

            last_loss = numpy.array(last_loss)

            print first_loss, last_loss
            # self.assertGreater(first_loss[0], last_loss[0])


class TestMNIST(TestParallelExecutorBase):
    @classmethod
    def setUpClass(cls):
        # Convert mnist to recordio file
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            reader = paddle.batch(mnist.train(), batch_size=32)
            feeder = fluid.DataFeeder(
                feed_list=[  # order is image and label
                    fluid.layers.data(
                        name='image', shape=[784]),
                    fluid.layers.data(
                        name='label', shape=[1], dtype='int64'),
                ],
                place=fluid.CPUPlace())
            fluid.recordio_writer.convert_reader_to_recordio_file(
                './mnist.recordio', reader, feeder)

    def test_simple_fc(self):
        self.check_network_convergence(simple_fc_net)
        self.check_network_convergence(simple_fc_net, allow_op_delay=True)

    def test_batchnorm_fc(self):
        self.check_network_convergence(fc_with_batchnorm)
        self.check_network_convergence(fc_with_batchnorm, allow_op_delay=True)


class TestResnet(TestParallelExecutorBase):
    # @classmethod
    # def setUpClass(cls):
    #     # import os
    #     # if os.path.exists('./flowers.recordio'):
    #     #     return
    #     with fluid.program_guard(fluid.Program(), fluid.Program()):
    #         reader = paddle.batch(flowers.train(), batch_size=4)
    #         feeder = fluid.DataFeeder(
    #             feed_list=[
    #                 fluid.layers.data(
    #                     name='image', shape=[3, 224, 224]),
    #                 fluid.layers.data(
    #                     name='label', shape=[1], dtype='int64'),
    #             ],
    #             place=fluid.CPUPlace())
    #         fluid.recordio_writer.convert_reader_to_recordio_file(
    #             "./flowers.recordio", reader, feeder, compressor=fluid.core.RecordIOWriter.Compressor.NoCompress)

    def test_resnet(self):
        import functools
        batch_size = 4
        self.check_network_convergence(
            functools.partial(
                SE_ResNeXt152, batch_size=batch_size),
            iter=20,
            batch_size=batch_size)
        self.check_network_convergence(
            functools.partial(
                SE_ResNeXt152, batch_size=batch_size),
            iter=20,
            batch_size=batch_size,
            allow_op_delay=True)


class ModelHyperParams(object):
    # Dictionary size for source and target language. This model directly uses
    # paddle.dataset.wmt16 in which <bos>, <eos> and <unk> token has
    # alreay been added, but the <pad> token is not added. Transformer requires
    # sequences in a mini-batch are padded to have the same length. A <pad> token is
    # added into the original dictionary in paddle.dateset.wmt16.

    # size of source word dictionary.
    src_vocab_size = 10000
    # index for <pad> token in source language.
    src_pad_idx = src_vocab_size

    # size of target word dictionay
    trg_vocab_size = 10000
    # index for <pad> token in target language.
    trg_pad_idx = trg_vocab_size

    # position value corresponding to the <pad> token.
    pos_pad_idx = 0

    # max length of sequences. It should plus 1 to include position
    # padding token for position encoding.
    max_length = 50

    # the dimension for word embeddings, which is also the last dimension of
    # the input and output of multi-head attention, position-wise feed-forward
    # networks, encoder and decoder.

    d_model = 512
    # size of the hidden layer in position-wise feed-forward networks.
    d_inner_hid = 1024
    # the dimension that keys are projected to for dot-product attention.
    d_key = 64
    # the dimension that values are projected to for dot-product attention.
    d_value = 64
    # number of head used in multi-head attention.
    n_head = 8
    # number of sub-layers to be stacked in the encoder and decoder.
    n_layer = 6
    # dropout rate used by all dropout layers.
    dropout = 0.1


import numpy as np


def prepare_batch_input(insts, src_pad_idx, trg_pad_idx, n_head):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias. Then, convert the numpy
    data to tensors and return a dict mapping names to tensors.
    """

    def __pad_batch_data(insts,
                         pad_idx,
                         is_target=False,
                         return_pos=True,
                         return_attn_bias=True,
                         return_max_len=True):
        """
        Pad the instances to the max sequence length in batch, and generate the
        corresponding position data and attention bias.
        """
        return_list = []
        max_len = max(len(inst) for inst in insts)
        inst_data = np.array(
            [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
        return_list += [inst_data.astype("int64").reshape([-1, 1])]
        if return_pos:
            inst_pos = np.array([[
                pos_i + 1 if w_i != pad_idx else 0
                for pos_i, w_i in enumerate(inst)
            ] for inst in inst_data])

            return_list += [inst_pos.astype("int64").reshape([-1, 1])]
        if return_attn_bias:
            if is_target:
                # This is used to avoid attention on paddings and subsequent
                # words.
                slf_attn_bias_data = np.ones((inst_data.shape[0], max_len,
                                              max_len))
                slf_attn_bias_data = np.triu(slf_attn_bias_data, 1).reshape(
                    [-1, 1, max_len, max_len])
                slf_attn_bias_data = np.tile(slf_attn_bias_data,
                                             [1, n_head, 1, 1]) * [-1e9]
            else:
                # This is used to avoid attention on paddings.
                slf_attn_bias_data = np.array([[0] * len(inst) + [-1e9] *
                                               (max_len - len(inst))
                                               for inst in insts])
                slf_attn_bias_data = np.tile(
                    slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                    [1, n_head, max_len, 1])
            return_list += [slf_attn_bias_data.astype("float32")]
        if return_max_len:
            return_list += [max_len]
        return return_list if len(return_list) > 1 else return_list[0]

    def data_to_tensor(data_list, name_list, input_dict, place):
        assert len(data_list) == len(name_list)
        for i in range(len(name_list)):
            tensor = fluid.LoDTensor()
            tensor.set(data_list[i], place)
            input_dict[name_list[i]] = tensor

    src_word, src_pos, src_slf_attn_bias, src_max_len = __pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, is_target=False)
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = __pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, is_target=True)
    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, trg_max_len, 1]).astype("float32")
    lbl_word = __pad_batch_data([inst[2] for inst in insts], trg_pad_idx, False,
                                False, False, False)
    lbl_weight = (lbl_word != trg_pad_idx).astype("float32").reshape([-1, 1])

    return [
        src_word, src_pos, trg_word, trg_pos, src_slf_attn_bias,
        trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight
    ]


import transformer_model


def transformer():
    return transformer_model.transformer(
        ModelHyperParams.src_vocab_size + 1,
        ModelHyperParams.trg_vocab_size + 1, ModelHyperParams.max_length + 1,
        ModelHyperParams.n_layer, ModelHyperParams.n_head,
        ModelHyperParams.d_key, ModelHyperParams.d_value,
        ModelHyperParams.d_model, ModelHyperParams.d_inner_hid,
        ModelHyperParams.dropout, ModelHyperParams.src_pad_idx,
        ModelHyperParams.trg_pad_idx, ModelHyperParams.pos_pad_idx)


class TestTransformer(TestParallelExecutorBase):
    @classmethod
    def setUpClass(cls):
        reader = paddle.batch(
            wmt16.train(ModelHyperParams.src_vocab_size,
                        ModelHyperParams.trg_vocab_size),
            batch_size=transformer_model.batch_size)

        with fluid.recordio_writer.create_recordio_writer(
                "./wmt16.recordio") as writer:
            for batch in reader():
                for tensor in prepare_batch_input(
                        batch, ModelHyperParams.src_pad_idx,
                        ModelHyperParams.trg_pad_idx, ModelHyperParams.n_head):
                    t = fluid.LoDTensor()
                    t.set(tensor, fluid.CPUPlace())
                    writer.append_tensor(t)
                writer.complete_append_tensor()

    @unittest.skip("transformer is buggy in multi gpu")
    def test_main(self):
        self.check_network_convergence(transformer)

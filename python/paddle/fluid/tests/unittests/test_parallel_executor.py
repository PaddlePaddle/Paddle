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

import numpy as np
import unittest

import paddle.fluid as fluid
import paddle
import paddle.dataset.mnist as mnist
import paddle.dataset.wmt16 as wmt16


def simple_fc_net(use_feed):
    if use_feed:
        img = fluid.layers.data(name='image', shape=[784], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    else:
        reader = fluid.layers.open_files(
            filenames=['./mnist.recordio'],
            shapes=[[-1, 784], [-1, 1]],
            lod_levels=[0, 0],
            dtypes=['float32', 'int64'],
            thread_num=1,
            for_parallel=True)
        reader = fluid.layers.io.double_buffer(reader)
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


def fc_with_batchnorm(use_feed):
    if use_feed:
        img = fluid.layers.data(name='image', shape=[784], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    else:
        reader = fluid.layers.open_files(
            filenames=['mnist.recordio'],
            shapes=[[-1, 784], [-1, 1]],
            lod_levels=[0, 0],
            dtypes=['float32', 'int64'],
            thread_num=1,
            for_parallel=True)
        reader = fluid.layers.io.double_buffer(reader)
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


def SE_ResNeXt50Small(batch_size=2, use_feed=False):
    assert not use_feed, "SE_ResNeXt doesn't support feed yet"

    img = fluid.layers.fill_constant(
        shape=[batch_size, 3, 224, 224], dtype='float32', value=0.0)
    label = fluid.layers.fill_constant(
        shape=[batch_size, 1], dtype='int64', value=0.0)

    conv = conv_bn_layer(
        input=img, num_filters=16, filter_size=3, stride=2, act='relu')
    conv = conv_bn_layer(
        input=conv, num_filters=16, filter_size=3, stride=1, act='relu')
    conv = conv_bn_layer(
        input=conv, num_filters=16, filter_size=3, stride=1, act='relu')
    conv = fluid.layers.pool2d(
        input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

    cardinality = 32
    reduction_ratio = 16
    depth = [3, 4, 6, 3]
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
                                  iter=50,
                                  batch_size=None,
                                  allow_op_delay=False,
                                  feed_dict=None,
                                  seed=None,
                                  use_parallel_executor=True,
                                  balance_parameter_opt_between_cards=False):
        def run_executor(exe, feed, fetch_list, program=None):
            if isinstance(exe, fluid.ParallelExecutor):
                res = exe.run(fetch_list=fetch_list, feed=feed)
            elif isinstance(exe, fluid.Executor):
                if program is None:
                    program = fluid.default_main_program()
                res = exe.run(program=program, feed=feed, fetch_list=fetch_list)
            else:
                raise ValueError('Unkown type exe')
            return res

        main = fluid.Program()
        startup = fluid.Program()
        startup.random_seed = 1  # Fix random seed
        with fluid.program_guard(main, startup):
            if seed is not None:
                startup.random_seed = seed
            loss = method(use_feed=feed_dict is not None)
            adam = fluid.optimizer.Adam()
            adam.minimize(loss)
            if memory_opt:
                fluid.memory_optimize(main)
            place = fluid.CUDAPlace(0)
            startup_exe = fluid.Executor(place)
            startup_exe.run(startup)

            if use_parallel_executor:
                exe = fluid.ParallelExecutor(
                    True,
                    loss_name=loss.name,
                    allow_op_delay=allow_op_delay,
                    balance_parameter_opt_between_cards=balance_parameter_opt_between_cards
                )
            else:
                exe = fluid.Executor(place=place)

            if batch_size is not None:
                batch_size *= fluid.core.get_cuda_device_count()
            begin = time.time()
            first_loss, = run_executor(
                exe=exe, feed=feed_dict, fetch_list=[loss.name])
            first_loss = np.array(first_loss)

            for i in xrange(iter):
                run_executor(exe=exe, feed=feed_dict, fetch_list=[])

            last_loss, = run_executor(
                exe=exe, feed=feed_dict, fetch_list=[loss.name])
            end = time.time()

            if batch_size is not None:
                print "%.4f Instance per second" % (
                    (batch_size * iter + 2) / (end - begin))

            last_loss = np.array(last_loss)

            print first_loss, last_loss
            # self.assertGreater(first_loss[0], last_loss[0])
            return first_loss, last_loss


class TestMNIST(TestParallelExecutorBase):
    @classmethod
    def setUpClass(cls):
        # Convert mnist to recordio file
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            reader = paddle.batch(mnist.train(), batch_size=4)
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

    def check_simple_fc_convergence(self, balance_parameter_opt_between_cards):
        self.check_network_convergence(simple_fc_net)
        self.check_network_convergence(simple_fc_net, allow_op_delay=True)

        img = np.zeros(shape=[32, 784], dtype='float32')
        label = np.ones(shape=[32, 1], dtype='int64')
        self.check_network_convergence(
            simple_fc_net,
            feed_dict={"image": img,
                       "label": label},
            balance_parameter_opt_between_cards=balance_parameter_opt_between_cards
        )

    def test_simple_fc(self):
        self.check_simple_fc_convergence(False)

    def test_simple_fc_with_new_strategy(self):
        self.check_simple_fc_convergence(True)

    def check_simple_fc_parallel_accuracy(self,
                                          balance_parameter_opt_between_cards):
        img = np.zeros(shape=[32, 784], dtype='float32')
        label = np.ones(shape=[32, 1], dtype='int64')
        single_first_loss, single_last_loss = self.check_network_convergence(
            method=simple_fc_net,
            seed=1000,
            feed_dict={"image": img,
                       "label": label},
            use_parallel_executor=False)
        parallel_first_loss, parallel_last_loss = self.check_network_convergence(
            method=simple_fc_net,
            seed=1000,
            feed_dict={"image": img,
                       "label": label},
            use_parallel_executor=True,
            balance_parameter_opt_between_cards=balance_parameter_opt_between_cards
        )

        for p_f in parallel_first_loss:
            self.assertAlmostEquals(p_f, single_first_loss[0], delta=1e-6)
        for p_l in parallel_last_loss:
            self.assertAlmostEquals(p_l, single_last_loss[0], delta=1e-6)

    def test_simple_fc_parallel_accuracy(self):
        self.check_simple_fc_parallel_accuracy(False)

    def test_simple_fc_parallel_accuracy_with_new_strategy(self):
        self.check_simple_fc_parallel_accuracy(True)

    def check_batchnorm_fc_convergence(self,
                                       balance_parameter_opt_between_cards):
        self.check_network_convergence(fc_with_batchnorm)
        img = np.zeros(shape=[32, 784], dtype='float32')
        label = np.ones(shape=[32, 1], dtype='int64')
        self.check_network_convergence(
            fc_with_batchnorm,
            feed_dict={"image": img,
                       "label": label},
            balance_parameter_opt_between_cards=balance_parameter_opt_between_cards
        )

    def test_batchnorm_fc(self):
        self.check_batchnorm_fc_convergence(False)

    def test_batchnorm_fc_with_new_strategy(self):
        self.check_batchnorm_fc_convergence(True)


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

    def check_resnet_convergence(self, balance_parameter_opt_between_cards):
        import functools
        batch_size = 2
        self.check_network_convergence(
            functools.partial(
                SE_ResNeXt50Small, batch_size=batch_size),
            iter=20,
            batch_size=batch_size,
            balance_parameter_opt_between_cards=balance_parameter_opt_between_cards
        )

    def test_resnet(self):
        self.check_resnet_convergence(False)

    def test_resnet_with_new_strategy(self):
        self.check_resnet_convergence(True)


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


def transformer(use_feed):
    assert not use_feed, "transfomer doesn't support feed yet"
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


class ParallelExecutorTestingDuringTraining(unittest.TestCase):
    def check_network_convergence(self, balance_parameter_opt_between_cards):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = simple_fc_net(True)
            test_program = main.clone(for_test=True)

            opt = fluid.optimizer.SGD(learning_rate=0.001)
            opt.minimize(loss)

            batch_size = 32
            image = np.random.normal(size=(batch_size, 784)).astype('float32')
            label = np.random.randint(0, 10, (batch_size, 1), dtype="int64")

            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup)
            feed_dict = {'image': image, 'label': label}

            train_exe = fluid.ParallelExecutor(
                use_cuda=True,
                loss_name=loss.name,
                main_program=main,
                balance_parameter_opt_between_cards=balance_parameter_opt_between_cards
            )

            test_exe = fluid.ParallelExecutor(
                use_cuda=True,
                main_program=test_program,
                share_vars_from=train_exe,
                balance_parameter_opt_between_cards=balance_parameter_opt_between_cards
            )

            for i in xrange(5):
                test_loss, = test_exe.run([loss.name], feed=feed_dict)
                test_loss = np.array(test_loss)

                train_loss, = train_exe.run([loss.name], feed=feed_dict)
                train_loss = np.array(train_loss)
                self.assertTrue(
                    np.allclose(
                        train_loss, test_loss, atol=1e-8),
                    "Train loss: " + str(train_loss) + "\n Test loss:" +
                    str(test_loss))

    def test_parallel_testing(self):
        self.check_network_convergence(False)

    def test_parallel_testing_with_new_strategy(self):
        self.check_network_convergence(True)


import paddle.dataset.conll05 as conll05
import paddle.fluid as fluid

word_dict, verb_dict, label_dict = conll05.get_dict()
word_dict_len = len(word_dict)
label_dict_len = len(label_dict)
pred_dict_len = len(verb_dict)
mark_dict_len = 2
word_dim = 32
mark_dim = 5
hidden_dim = 512
depth = 8
mix_hidden_lr = 1e-3
embedding_name = 'emb'


def db_lstm(word, predicate, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2, mark,
            is_sparse, balance_parameter_opt_between_cards, **ignored):
    # 8 features
    predicate_embedding = fluid.layers.embedding(
        input=predicate,
        is_sparse=is_sparse,
        size=[pred_dict_len, word_dim],
        dtype='float32',
        param_attr='vemb')

    mark_embedding = fluid.layers.embedding(
        input=mark,
        is_sparse=is_sparse,
        size=[mark_dict_len, mark_dim],
        dtype='float32')

    word_input = [word, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2]
    emb_layers = [
        fluid.layers.embedding(
            size=[word_dict_len, word_dim],
            is_sparse=is_sparse,
            input=x,
            param_attr=fluid.ParamAttr(
                name=embedding_name, trainable=False)) for x in word_input
    ]
    emb_layers.append(predicate_embedding)
    emb_layers.append(mark_embedding)

    hidden_0_layers = [
        fluid.layers.fc(input=emb, size=hidden_dim, act='tanh')
        for emb in emb_layers
    ]

    hidden_0 = fluid.layers.sums(input=hidden_0_layers)

    lstm_0 = fluid.layers.dynamic_lstm(
        input=hidden_0,
        size=hidden_dim,
        candidate_activation='relu',
        gate_activation='sigmoid',
        cell_activation='sigmoid')

    # stack L-LSTM and R-LSTM with direct edges
    input_tmp = [hidden_0, lstm_0]

    for i in range(1, depth):
        mix_hidden = fluid.layers.sums(input=[
            fluid.layers.fc(input=input_tmp[0], size=hidden_dim, act='tanh'),
            fluid.layers.fc(input=input_tmp[1], size=hidden_dim, act='tanh')
        ])

        lstm = fluid.layers.dynamic_lstm(
            input=mix_hidden,
            size=hidden_dim,
            candidate_activation='relu',
            gate_activation='sigmoid',
            cell_activation='sigmoid',
            is_reverse=((i % 2) == 1))

        input_tmp = [mix_hidden, lstm]

    feature_out = fluid.layers.sums(input=[
        fluid.layers.fc(input=input_tmp[0], size=label_dict_len, act='tanh'),
        fluid.layers.fc(input=input_tmp[1], size=label_dict_len, act='tanh')
    ])

    return feature_out


class TestCRFModel(unittest.TestCase):
    def check_network_convergence(self,
                                  is_sparse,
                                  balance_parameter_opt_between_cards=False):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            word = fluid.layers.data(
                name='word_data', shape=[1], dtype='int64', lod_level=1)
            predicate = fluid.layers.data(
                name='verb_data', shape=[1], dtype='int64', lod_level=1)
            ctx_n2 = fluid.layers.data(
                name='ctx_n2_data', shape=[1], dtype='int64', lod_level=1)
            ctx_n1 = fluid.layers.data(
                name='ctx_n1_data', shape=[1], dtype='int64', lod_level=1)
            ctx_0 = fluid.layers.data(
                name='ctx_0_data', shape=[1], dtype='int64', lod_level=1)
            ctx_p1 = fluid.layers.data(
                name='ctx_p1_data', shape=[1], dtype='int64', lod_level=1)
            ctx_p2 = fluid.layers.data(
                name='ctx_p2_data', shape=[1], dtype='int64', lod_level=1)
            mark = fluid.layers.data(
                name='mark_data', shape=[1], dtype='int64', lod_level=1)

            feature_out = db_lstm(**locals())
            target = fluid.layers.data(
                name='target', shape=[1], dtype='int64', lod_level=1)
            crf_cost = fluid.layers.linear_chain_crf(
                input=feature_out,
                label=target,
                param_attr=fluid.ParamAttr(
                    name='crfw', learning_rate=1e-1))
            avg_cost = fluid.layers.mean(crf_cost)

            sgd_optimizer = fluid.optimizer.SGD(
                learning_rate=fluid.layers.exponential_decay(
                    learning_rate=0.01,
                    decay_steps=100000,
                    decay_rate=0.5,
                    staircase=True))
            sgd_optimizer.minimize(avg_cost)

            train_data = paddle.batch(
                paddle.reader.shuffle(
                    paddle.dataset.conll05.test(), buf_size=8192),
                batch_size=16)

            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup)

            pe = fluid.ParallelExecutor(
                use_cuda=True,
                loss_name=avg_cost.name,
                balance_parameter_opt_between_cards=balance_parameter_opt_between_cards
            )

            feeder = fluid.DataFeeder(
                feed_list=[
                    word, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2, predicate,
                    mark, target
                ],
                place=fluid.CPUPlace())

            data = train_data()
            for i in xrange(10):
                cur_batch = next(data)
                print map(np.array,
                          pe.run(feed=feeder.feed(cur_batch),
                                 fetch_list=[avg_cost.name]))[0]

    def test_update_sparse_parameter(self):
        self.check_network_convergence(is_sparse=True)

    def test_update_dense_parameter(self):
        self.check_network_convergence(is_sparse=False)

    def test_update_sparse_parameter_with_new_strategy(self):
        self.check_network_convergence(
            is_sparse=False, balance_parameter_opt_between_cards=True)

    def test_update_dense_parameter_with_new_strategy(self):
        self.check_network_convergence(
            is_sparse=False, balance_parameter_opt_between_cards=True)


# test fetch all the variables of global_block

import paddle.dataset.flowers as flowers
import math


def Lenet(data, class_dim):
    conv1 = fluid.layers.conv2d(data, 32, 5, 1, act=None)
    bn1 = fluid.layers.batch_norm(conv1, act='relu')
    pool1 = fluid.layers.pool2d(bn1, 2, 'max', 2)
    conv2 = fluid.layers.conv2d(pool1, 50, 5, 1, act=None)
    bn2 = fluid.layers.batch_norm(conv2, act='relu')
    pool2 = fluid.layers.pool2d(bn2, 2, 'max', 2)

    fc1 = fluid.layers.fc(pool2, size=500, act='relu')
    fc2 = fluid.layers.fc(fc1, size=class_dim, act='softmax')

    return fc2


class TestFetchOp(unittest.TestCase):
    def parallel_exe(self, train_inputs, seed):
        main = fluid.Program()
        startup = fluid.Program()
        startup.random_seed = seed
        with fluid.program_guard(main, startup):
            data = fluid.layers.data(
                name='image', shape=[3, 224, 224], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            out = Lenet(data, class_dim=102)
            loss = fluid.layers.cross_entropy(input=out, label=label)
            loss = fluid.layers.mean(loss)

            opt = fluid.optimizer.Momentum(
                learning_rate=0.1,
                momentum=0.9,
                regularization=fluid.regularizer.L2Decay(1e-4))

            opt.minimize(loss)

            # TODO(zcd): I found that onece the memory optimizer is open,
            # parallel_exe doesn't fetch some variable, such as conv2d_0.b_0@GRAD,
            # conv2d_1.b_0@GRAD. Those variables should not be pruned.
            # fluid.memory_optimize(main)

            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup)

            feeder = fluid.DataFeeder(place=place, feed_list=[data, label])
            pe = fluid.ParallelExecutor(
                use_cuda=True, loss_name=loss.name, main_program=main)

            fetch_list = []
            all_vars = main.global_block().vars
            for k, v in all_vars.iteritems():
                if 'tmp' not in k and k[0] is not '_' or v.persistable:
                    fetch_list.append(k)

            for data in train_inputs:
                ret = pe.run(fetch_list, feed=feeder.feed(data))
                for i in range(len(fetch_list)):
                    assert not math.isnan(np.sum(ret[i])) and \
                           not math.isinf(np.sum(ret[i]))

    def test_update_sparse_parameter(self):
        tst_reader = paddle.batch(flowers.test(use_xmap=False), batch_size=16)
        tst_reader_iter = tst_reader()

        iters = 3
        train_inputs = []
        for i in range(iters):
            train_inputs.append(tst_reader_iter.next())

        self.parallel_exe(train_inputs, seed=1)


class TestFeedParallel(unittest.TestCase):
    def test_main(self):
        main = fluid.Program()
        startup = fluid.Program()
        startup.random_seed = 1
        with fluid.scope_guard(fluid.core.Scope()):
            with fluid.program_guard(main, startup):
                data = fluid.layers.data(
                    name='image', shape=[3, 224, 224], dtype='float32')
                label = fluid.layers.data(
                    name='label', shape=[1], dtype='int64')
                out = Lenet(data, class_dim=102)
                loss = fluid.layers.cross_entropy(input=out, label=label)
                loss = fluid.layers.mean(loss)
                opt = fluid.optimizer.Momentum(
                    learning_rate=0.1,
                    momentum=0.9,
                    regularization=fluid.regularizer.L2Decay(1e-4))

                opt.minimize(loss)
        place = fluid.CUDAPlace(0)
        feeder = fluid.DataFeeder(place=place, feed_list=[data, label])
        reader = feeder.decorate_reader(
            paddle.batch(
                flowers.train(), batch_size=16), multi_devices=True)
        exe = fluid.Executor(place)
        exe.run(startup)
        pe = fluid.ParallelExecutor(
            use_cuda=True, loss_name=loss.name, main_program=main)

        for batch_id, data in enumerate(reader()):
            loss_np = np.array(pe.run(feed=data, fetch_list=[loss.name])[0])
            print batch_id, loss_np
            if batch_id == 2:
                break


if __name__ == '__main__':
    unittest.main()

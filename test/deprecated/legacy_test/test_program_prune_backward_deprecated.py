#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import os
import unittest

import numpy as np
import seresnext_net
import transformer_model
from feed_data_reader import FeedDataReader
from simple_nets import fc_with_batchnorm, init_data, simple_fc_net

import paddle
from paddle import base
from paddle.base import core
from paddle.dataset import wmt16

paddle.enable_static()

DeviceType = core.DeviceType


class ModelHyperParams:
    # Dictionary size for source and target language. This model directly uses
    # paddle.dataset.wmt16 in which <bos>, <eos> and <unk> token has
    # already been added, but the <pad> token is not added. Transformer requires
    # sequences in a mini-batch are padded to have the same length. A <pad> token is
    # added into the original dictionary in paddle.dataset.wmt16.

    # size of source word dictionary.
    src_vocab_size = 10000
    # index for <pad> token in source language.
    src_pad_idx = src_vocab_size

    # size of target word dictionary
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
    # NOTE(zcd): the origin number of layer is 6, to make this unit test faster,
    # we should reduce the layer number to 4.
    n_layer = 4
    # dropout rate used by all dropout layers.
    dropout = 0.1


def prepare_batch_input(insts, src_pad_idx, trg_pad_idx, n_head):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias. Then, convert the numpy
    data to tensors and return a dict mapping names to tensors.
    """

    def __pad_batch_data(
        insts,
        pad_idx,
        is_target=False,
        return_pos=True,
        return_attn_bias=True,
        return_max_len=True,
    ):
        """
        Pad the instances to the max sequence length in batch, and generate the
        corresponding position data and attention bias.
        """
        return_list = []
        max_len = max(len(inst) for inst in insts)
        inst_data = np.array(
            [inst + [pad_idx] * (max_len - len(inst)) for inst in insts]
        )
        return_list += [inst_data.astype("int64").reshape([-1, 1])]
        if return_pos:
            inst_pos = np.array(
                [
                    [
                        pos_i + 1 if w_i != pad_idx else 0
                        for pos_i, w_i in enumerate(inst)
                    ]
                    for inst in inst_data
                ]
            )

            return_list += [inst_pos.astype("int64").reshape([-1, 1])]
        if return_attn_bias:
            if is_target:
                # This is used to avoid attention on paddings and subsequent
                # words.
                slf_attn_bias_data = np.ones(
                    (inst_data.shape[0], max_len, max_len)
                )
                slf_attn_bias_data = np.triu(slf_attn_bias_data, 1).reshape(
                    [-1, 1, max_len, max_len]
                )
                slf_attn_bias_data = np.tile(
                    slf_attn_bias_data, [1, n_head, 1, 1]
                ) * [-1e9]
            else:
                # This is used to avoid attention on paddings.
                slf_attn_bias_data = np.array(
                    [
                        [0] * len(inst) + [-1e9] * (max_len - len(inst))
                        for inst in insts
                    ]
                )
                slf_attn_bias_data = np.tile(
                    slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                    [1, n_head, max_len, 1],
                )
            return_list += [slf_attn_bias_data.astype("float32")]
        if return_max_len:
            return_list += [max_len]
        return return_list if len(return_list) > 1 else return_list[0]

    src_word, src_pos, src_slf_attn_bias, src_max_len = __pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, is_target=False
    )
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = __pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, is_target=True
    )
    trg_src_attn_bias = np.tile(
        src_slf_attn_bias[:, :, ::src_max_len, :], [1, 1, trg_max_len, 1]
    ).astype("float32")
    lbl_word = __pad_batch_data(
        [inst[2] for inst in insts], trg_pad_idx, False, False, False, False
    )
    lbl_weight = (lbl_word != trg_pad_idx).astype("float32").reshape([-1, 1])

    return [
        src_word,
        src_pos,
        trg_word,
        trg_pos,
        src_slf_attn_bias,
        trg_slf_attn_bias,
        trg_src_attn_bias,
        lbl_word,
        lbl_weight,
    ]


feed_data_reader = None


def transformer(use_feed):
    assert not use_feed, "transformer doesn't support feed yet"
    return transformer_model.transformer(
        ModelHyperParams.src_vocab_size + 1,
        ModelHyperParams.trg_vocab_size + 1,
        ModelHyperParams.max_length + 1,
        ModelHyperParams.n_layer,
        ModelHyperParams.n_head,
        ModelHyperParams.d_key,
        ModelHyperParams.d_value,
        ModelHyperParams.d_model,
        ModelHyperParams.d_inner_hid,
        ModelHyperParams.dropout,
        ModelHyperParams.src_pad_idx,
        ModelHyperParams.trg_pad_idx,
        ModelHyperParams.pos_pad_idx,
    )


def get_feed_data_reader():
    global feed_data_reader
    if feed_data_reader is not None:
        return feed_data_reader

    reader = paddle.batch(
        wmt16.train(
            ModelHyperParams.src_vocab_size, ModelHyperParams.trg_vocab_size
        ),
        batch_size=transformer_model.batch_size,
    )
    all_batch_tensors = []
    for batch in reader():
        tensors = []
        for tensor in prepare_batch_input(
            batch,
            ModelHyperParams.src_pad_idx,
            ModelHyperParams.trg_pad_idx,
            ModelHyperParams.n_head,
        ):
            tensors.append(np.array(tensor))
        all_batch_tensors.append(tensors)

    def __reader__():
        yield from all_batch_tensors

    feed_data_reader = FeedDataReader(
        feed_list=transformer_model.build_inputs(
            ModelHyperParams.max_length + 1, ModelHyperParams.n_head
        ),
        reader=__reader__,
    )

    return feed_data_reader


def simple_fc_net_with_accuracy(use_feed):
    img = paddle.static.data(name='image', shape=[-1, 784], dtype='float32')
    label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')

    hidden = img
    for _ in range(4):
        hidden = paddle.static.nn.fc(
            hidden,
            size=200,
            activation='relu',
            bias_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)
            ),
        )
    prediction = paddle.static.nn.fc(hidden, size=10, activation='softmax')
    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    loss = paddle.mean(loss)
    accuracy_out = paddle.static.accuracy(input=prediction, label=label, k=5)
    return loss


def cond_net(use_feed=None):
    x = paddle.static.data(name="x", shape=[-1, 4], dtype='float32')
    label = paddle.static.data('label', shape=[-1, 1], dtype='int64')
    prediction = paddle.static.nn.fc(x, size=1, activation=None)

    def loss1(pred, label):
        x = paddle.static.data(name="x", shape=[-1, 4], dtype='float32')
        loss = paddle.nn.functional.cross_entropy(
            input=pred, label=label, reduction='none', use_softmax=False
        )
        avg_loss = paddle.mean(loss, name='mean_cross_entropy_loss')
        return avg_loss

    def loss2(pred, label):
        loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=pred, label=label
        )
        avg_loss = paddle.mean(loss, name='mean_softmax_loss')
        return avg_loss

    two = paddle.tensor.fill_constant([1], 'int32', 2)
    pred = two == 0
    avg_loss = paddle.static.nn.case(
        [(pred, lambda: loss1(prediction, label))],
        lambda: loss2(prediction, label),
    )
    return avg_loss


def pylayer_net(use_feed=None):
    x = paddle.static.data(name="x", shape=[-1, 4], dtype='float32')
    label = paddle.static.data('label', shape=[-1, 1], dtype='int64')

    def forward_fn(x):
        y = 3 * x
        return y

    def backward_fn(dy):
        grad = paddle.exp(dy)
        return grad

    y = paddle.static.nn.static_pylayer(forward_fn, [x], backward_fn)
    hidden = paddle.static.nn.fc(x=[y], size=4, activation="softmax")
    loss = paddle.nn.functional.cross_entropy(
        input=hidden, label=label, reduction='none', use_softmax=False
    )
    loss = paddle.mean(loss, name='mean_softmax_loss')
    return loss


def optimization_in_cond_net(with_optimize=False):
    x = paddle.static.data(name="x", shape=[-1, 4], dtype='float32')
    label = paddle.static.data('label', shape=[-1, 1], dtype='int64')
    prediction = paddle.static.nn.fc(x, size=1, activation=None)

    def loss1(opt, pred, label, with_optimize):
        x = paddle.static.data(name="x", shape=[-1, 4], dtype='float32')
        loss = paddle.nn.functional.cross_entropy(
            input=pred, label=label, reduction='none', use_softmax=False
        )
        avg_loss = paddle.mean(loss, name='mean_cross_entropy_loss')
        if with_optimize:
            opt.minimize(avg_loss)
        return avg_loss

    def loss2(opt, pred, label, with_optimize):
        loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=pred, label=label
        )
        avg_loss = paddle.mean(loss, name='mean_softmax_loss')
        if with_optimize:
            opt.minimize(avg_loss)
        return avg_loss

    sgd = paddle.optimizer.SGD(learning_rate=0.1)
    two = paddle.tensor.fill_constant([1], 'int32', 2)
    pred = two == 0
    avg_loss = paddle.static.nn.case(
        [(pred, lambda: loss1(sgd, prediction, label, with_optimize))],
        lambda: loss2(sgd, prediction, label, with_optimize),
    )
    return avg_loss


def optimization_in_pylayer_net(with_optimize=False):
    x = paddle.static.data(name="x", shape=[-1, 4], dtype='float32')
    label = paddle.static.data('label', shape=[-1, 1], dtype='int64')

    def forward_fn(x):
        y = 3 * x
        return y

    def backward_fn(dy):
        grad = paddle.exp(dy)
        return grad

    y = paddle.static.nn.static_pylayer(forward_fn, [x], backward_fn)
    hidden = 3 * y
    loss = paddle.nn.functional.softmax_with_cross_entropy(
        logits=hidden, label=label
    )
    loss = paddle.mean(loss, name='mean_softmax_loss')
    sgd = paddle.optimizer.SGD(learning_rate=0.1)
    if with_optimize:
        sgd.minimize(loss)

    return loss


class TestProgramPruneBackward(unittest.TestCase):
    def program_compare(self, program_a, program_b):
        assert isinstance(
            program_a, base.framework.Program
        ), "The first argument should be base.framework.Program."
        assert isinstance(
            program_b, base.framework.Program
        ), "The second argument should be base.framework Program."

        self.assertEqual(len(program_a.blocks), len(program_b.blocks))
        for idx in range(len(program_a.blocks)):
            block_a = program_a.blocks[idx]
            block_b = program_b.blocks[idx]
            self.assertEqual(len(block_a.ops), len(block_b.ops))
            self.assertEqual(len(block_a.vars), len(block_b.vars))
            for op_idx in range(len(block_a.ops)):
                self.assertEqual(
                    block_a.ops[op_idx].type, block_b.ops[op_idx].type
                )
            for var_key in list(block_a.vars.keys()):
                self.assertTrue(block_b.has_var(var_key))

    def check_prune_correctness(self, method, feed_dict, optimizer):
        loss = method(use_feed=False)

        main_program = base.default_main_program()
        test_prog_orig = main_program.clone(for_test=True)
        optimizer().minimize(loss)
        test_prog_prune = main_program.clone(for_test=True)

        self.program_compare(test_prog_orig, test_prog_prune)

        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(core.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for place in places:
            exe = base.Executor(place)
            exe.run(base.default_startup_program())

            (loss_data_prune,) = exe.run(
                test_prog_prune, feed=feed_dict, fetch_list=[loss]
            )
            (loss_data_orig,) = exe.run(
                test_prog_orig, feed=feed_dict, fetch_list=[loss]
            )
            self.assertEqual(loss_data_orig, loss_data_prune)

    def test_simple_fc_net(self):
        def optimizer():
            optimizer = paddle.optimizer.SGD(
                learning_rate=0.001,
                weight_decay=paddle.regularizer.L2Decay(1e-4),
            )
            return optimizer

        with self.program_scope_guard():
            img, label = init_data()
            self.check_prune_correctness(
                method=simple_fc_net,
                feed_dict={"image": img, "label": label},
                optimizer=optimizer,
            )

    def test_simple_fc_net_with_accuracy(self):
        def optimizer():
            optimizer = paddle.optimizer.SGD(
                learning_rate=0.001,
                weight_decay=paddle.regularizer.L2Decay(1e-4),
            )
            return optimizer

        with self.program_scope_guard():
            img, label = init_data()
            self.check_prune_correctness(
                method=simple_fc_net_with_accuracy,
                feed_dict={"image": img, "label": label},
                optimizer=optimizer,
            )

    def test_batchnorm_fc(self):
        def optimizer():
            optimizer = paddle.optimizer.SGD(
                learning_rate=0.001,
                weight_decay=paddle.regularizer.L2Decay(1e-4),
            )
            return optimizer

        with self.program_scope_guard():
            img, label = init_data()
            self.check_prune_correctness(
                method=fc_with_batchnorm,
                feed_dict={"image": img, "label": label},
                optimizer=optimizer,
            )

    def test_seresnet(self):
        with self.program_scope_guard():
            self.check_prune_correctness(
                method=seresnext_net.model,
                feed_dict=seresnext_net.feed_dict(use_device=DeviceType.CPU),
                optimizer=seresnext_net.optimizer,
            )

    def test_transformer(self):
        def optimizer():
            optimizer = paddle.optimizer.Adam(
                learning_rate=0.001,
                weight_decay=paddle.regularizer.L2Decay(1e-4),
            )
            return optimizer

        with self.program_scope_guard():
            # the program argument is used to distinguish Program and CompiledProgram
            feed_dict = get_feed_data_reader().get_next(
                base.Executor(core.CPUPlace()), base.default_main_program()
            )
            self.check_prune_correctness(
                method=transformer, feed_dict=feed_dict, optimizer=optimizer
            )

    def test_cond(self):
        def optimizer():
            optimizer = paddle.optimizer.SGD(learning_rate=0.01)
            return optimizer

        with self.program_scope_guard():
            x_in = np.random.random(size=(10, 4)).astype('float32')
            label_in = np.random.randint(1, size=(10, 1)).astype('int64')
            feed_dict = {'x': x_in, 'label': label_in}
            self.check_prune_correctness(
                method=cond_net, feed_dict=feed_dict, optimizer=optimizer
            )

    def test_pylayer(self):
        def optimizer():
            optimizer = paddle.optimizer.SGD(learning_rate=0.01)
            return optimizer

        with self.program_scope_guard():
            x_in = np.random.random(size=(10, 4)).astype('float32')
            label_in = np.random.randint(1, size=(10, 1)).astype('int64')
            feed_dict = {'x': x_in, 'label': label_in}
            self.check_prune_correctness(
                method=pylayer_net, feed_dict=feed_dict, optimizer=optimizer
            )

    def test_optimization_in_cond(self):
        x_in = np.random.random(size=(10, 4)).astype('float32')
        label_in = np.random.randint(1, size=(10, 1)).astype('int64')
        feed_dict = {'x': x_in, 'label': label_in}
        with self.program_scope_guard():
            loss = optimization_in_cond_net(False)
            main_program = base.default_main_program()
            test_prog_orig = main_program.clone(for_test=True)
            place = core.CPUPlace()
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            (loss_data_orig,) = exe.run(
                test_prog_orig, feed=feed_dict, fetch_list=[loss]
            )

        with self.program_scope_guard():
            loss = optimization_in_cond_net(True)
            main_program = base.default_main_program()
            test_prog_prune = main_program.clone(for_test=True)

            place = core.CPUPlace()
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            (loss_data_prune,) = exe.run(
                test_prog_prune, feed=feed_dict, fetch_list=[loss]
            )

        self.program_compare(test_prog_orig, test_prog_prune)
        self.assertEqual(loss_data_orig, loss_data_prune)

    def test_optimization_in_pylayer(self):
        x_in = np.random.random(size=(10, 4)).astype('float32')
        label_in = np.random.randint(1, size=(10, 1)).astype('int64')
        feed_dict = {'x': x_in, 'label': label_in}
        with self.program_scope_guard():
            loss = optimization_in_pylayer_net(False)
            main_program = base.default_main_program()
            test_prog_orig = main_program.clone(for_test=True)
            place = core.CPUPlace()
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            (loss_data_orig,) = exe.run(
                test_prog_orig, feed=feed_dict, fetch_list=[loss]
            )

        with self.program_scope_guard():
            loss = optimization_in_pylayer_net(True)
            main_program = base.default_main_program()
            test_prog_prune = main_program.clone(for_test=True)

            place = core.CPUPlace()
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            (loss_data_prune,) = exe.run(
                test_prog_prune, feed=feed_dict, fetch_list=[loss]
            )

        self.program_compare(test_prog_orig, test_prog_prune)
        self.assertEqual(loss_data_orig, loss_data_prune)

    @contextlib.contextmanager
    def program_scope_guard(self):
        prog = base.Program()
        startup_prog = base.Program()
        scope = base.core.Scope()
        with base.scope_guard(scope):
            with base.program_guard(prog, startup_prog):
                with base.unique_name.guard():
                    yield


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()

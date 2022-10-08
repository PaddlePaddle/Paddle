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

import unittest

import contextlib
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
from simple_nets import init_data, simple_fc_net, fc_with_batchnorm
import seresnext_net
from test_parallel_executor_transformer import transformer, get_feed_data_reader, DeviceType
from fake_reader import fake_imdb_reader
import paddle


def lstm_net(use_feed):
    dict_dim = 5147
    emb_dim = 128
    hid_dim = 128
    hid_dim2 = 96
    class_dim = 2
    emb_lr = 30.0
    data = fluid.layers.data(name="words",
                             shape=[1],
                             dtype="int64",
                             lod_level=1)
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    emb = fluid.layers.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(learning_rate=emb_lr))
    fc0 = fluid.layers.fc(input=emb, size=hid_dim * 4)
    lstm_h, c = fluid.layers.dynamic_lstm(input=fc0,
                                          size=hid_dim * 4,
                                          is_reverse=False)
    lstm_max = fluid.layers.sequence_pool(input=lstm_h, pool_type='max')
    lstm_max_tanh = fluid.layers.tanh(lstm_max)
    fc1 = fluid.layers.fc(input=lstm_max_tanh, size=hid_dim2, act='tanh')
    prediction = fluid.layers.fc(input=fc1, size=class_dim, act='softmax')
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = paddle.mean(x=cost)
    return avg_cost


def simple_fc_net_with_accuracy(use_feed):
    img = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    hidden = img
    for _ in range(4):
        hidden = fluid.layers.fc(
            hidden,
            size=200,
            act='relu',
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=1.0)))
    prediction = fluid.layers.fc(hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = paddle.mean(loss)
    accuracy_out = fluid.layers.accuracy(input=prediction, label=label, k=5)
    return loss


def cond_net(use_feed=None):
    x = fluid.layers.data(name="x", shape=[4], dtype='float32')
    label = fluid.layers.data('label', shape=[1], dtype='int64')
    prediction = fluid.layers.fc(input=x, size=1, act=None)

    def loss1(pred, label):
        x = fluid.layers.data(name="x", shape=[4], dtype='float32')
        loss = fluid.layers.cross_entropy(input=pred, label=label)
        avg_loss = paddle.mean(loss, name='mean_cross_entropy_loss')
        return avg_loss

    def loss2(pred, label):
        loss = fluid.layers.softmax_with_cross_entropy(logits=pred, label=label)
        avg_loss = paddle.mean(loss, name='mean_softmax_loss')
        return avg_loss

    two = fluid.layers.fill_constant([1], 'int32', 2)
    pred = (two == 0)
    avg_loss = fluid.layers.case([(pred, lambda: loss1(prediction, label))],
                                 lambda: loss2(prediction, label))
    return avg_loss


def optimization_in_cond_net(with_optimize=False):
    x = fluid.layers.data(name="x", shape=[4], dtype='float32')
    label = fluid.layers.data('label', shape=[1], dtype='int64')
    prediction = fluid.layers.fc(input=x, size=1, act=None)

    def loss1(opt, pred, label, with_optimize):
        x = fluid.layers.data(name="x", shape=[4], dtype='float32')
        loss = fluid.layers.cross_entropy(input=pred, label=label)
        avg_loss = paddle.mean(loss, name='mean_cross_entropy_loss')
        if with_optimize:
            opt.minimize(avg_loss)
        return avg_loss

    def loss2(opt, pred, label, with_optimize):
        loss = fluid.layers.softmax_with_cross_entropy(logits=pred, label=label)
        avg_loss = paddle.mean(loss, name='mean_softmax_loss')
        if with_optimize:
            opt.minimize(avg_loss)
        return avg_loss

    sgd = fluid.optimizer.SGD(learning_rate=0.1)
    two = fluid.layers.fill_constant([1], 'int32', 2)
    pred = (two == 0)
    avg_loss = fluid.layers.case(
        [(pred, lambda: loss1(sgd, prediction, label, with_optimize))],
        lambda: loss2(sgd, prediction, label, with_optimize))
    return avg_loss


class TestProgramPruneBackward(unittest.TestCase):

    def program_compare(self, program_a, program_b):
        assert isinstance(
            program_a, fluid.framework.Program
        ), "The first argument should be fluid.framework.Program."
        assert isinstance(
            program_b, fluid.framework.Program
        ), "The second argument should be fluid.framework Program."

        self.assertEqual(len(program_a.blocks), len(program_b.blocks))
        for idx in range(len(program_a.blocks)):
            block_a = program_a.blocks[idx]
            block_b = program_b.blocks[idx]
            self.assertEqual(len(block_a.ops), len(block_b.ops))
            self.assertEqual(len(block_a.vars), len(block_b.vars))
            for op_idx in range(len(block_a.ops)):
                self.assertEqual(block_a.ops[op_idx].type,
                                 block_b.ops[op_idx].type)
            for var_key in list(block_a.vars.keys()):
                self.assertTrue(block_b.has_var(var_key))

    def check_prune_correctness(self, method, feed_dict, optimizer):
        loss = method(use_feed=False)

        main_program = fluid.default_main_program()
        test_prog_orig = main_program.clone(for_test=True)
        optimizer().minimize(loss)
        test_prog_prune = main_program.clone(for_test=True)

        self.program_compare(test_prog_orig, test_prog_prune)

        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for place in places:
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            loss_data_prune, = exe.run(test_prog_prune,
                                       feed=feed_dict,
                                       fetch_list=[loss.name])
            loss_data_orig, = exe.run(test_prog_orig,
                                      feed=feed_dict,
                                      fetch_list=[loss.name])
            self.assertEqual(loss_data_orig, loss_data_prune)

    def test_simple_fc_net(self):

        def optimizer():
            optimizer = fluid.optimizer.SGD(
                learning_rate=0.001,
                regularization=fluid.regularizer.L2Decay(1e-4))
            return optimizer

        with self.program_scope_guard():
            img, label = init_data()
            self.check_prune_correctness(method=simple_fc_net,
                                         feed_dict={
                                             "image": img,
                                             "label": label
                                         },
                                         optimizer=optimizer)

    def test_simple_fc_net_with_accuracy(self):

        def optimizer():
            optimizer = fluid.optimizer.SGD(
                learning_rate=0.001,
                regularization=fluid.regularizer.L2Decay(1e-4))
            return optimizer

        with self.program_scope_guard():
            img, label = init_data()
            self.check_prune_correctness(method=simple_fc_net_with_accuracy,
                                         feed_dict={
                                             "image": img,
                                             "label": label
                                         },
                                         optimizer=optimizer)

    def test_batchnorm_fc(self):

        def optimizer():
            optimizer = fluid.optimizer.SGD(
                learning_rate=0.001,
                regularization=fluid.regularizer.L2Decay(1e-4))
            return optimizer

        with self.program_scope_guard():
            img, label = init_data()
            self.check_prune_correctness(method=fc_with_batchnorm,
                                         feed_dict={
                                             "image": img,
                                             "label": label
                                         },
                                         optimizer=optimizer)

    def test_seresnet(self):
        with self.program_scope_guard():
            self.check_prune_correctness(
                method=seresnext_net.model,
                feed_dict=seresnext_net.feed_dict(use_device=DeviceType.CPU),
                optimizer=seresnext_net.optimizer)

    def test_transformer(self):

        def optimizer():
            optimizer = fluid.optimizer.Adam(
                learning_rate=0.001,
                regularization=fluid.regularizer.L2Decay(1e-4))
            return optimizer

        with self.program_scope_guard():
            # the program argument is used to distinguish Program and CompiledProgram
            feed_dict = get_feed_data_reader().get_next(
                fluid.Executor(core.CPUPlace()), fluid.default_main_program())
            self.check_prune_correctness(method=transformer,
                                         feed_dict=feed_dict,
                                         optimizer=optimizer)

    def test_lstm(self):

        def optimizer():
            optimizer = fluid.optimizer.Adagrad(
                learning_rate=0.001,
                regularization=fluid.regularizer.L2Decay(1e-4))
            return optimizer

        with self.program_scope_guard():
            word_dict_size = 5147
            reader = fake_imdb_reader(word_dict_size, 1)
            data = fluid.layers.data(name="words",
                                     shape=[1],
                                     dtype="int64",
                                     lod_level=1)
            label = fluid.layers.data(name="label", shape=[1], dtype="int64")
            feeder = fluid.DataFeeder(feed_list=[data, label],
                                      place=core.CPUPlace())
            feed_data = feeder.feed(reader())
            self.check_prune_correctness(method=lstm_net,
                                         feed_dict=feed_data,
                                         optimizer=optimizer)

    def test_cond(self):

        def optimizer():
            optimizer = fluid.optimizer.SGD(learning_rate=0.01)
            return optimizer

        with self.program_scope_guard():
            x_in = np.random.random(size=(10, 4)).astype('float32')
            label_in = np.random.randint(1, size=(10, 1)).astype('int64')
            feed_dict = {'x': x_in, 'label': label_in}
            self.check_prune_correctness(method=cond_net,
                                         feed_dict=feed_dict,
                                         optimizer=optimizer)

    def test_optimization_in_cond(self):
        x_in = np.random.random(size=(10, 4)).astype('float32')
        label_in = np.random.randint(1, size=(10, 1)).astype('int64')
        feed_dict = {'x': x_in, 'label': label_in}
        with self.program_scope_guard():
            loss = optimization_in_cond_net(False)
            main_program = fluid.default_main_program()
            test_prog_orig = main_program.clone(for_test=True)
            place = core.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            loss_data_orig, = exe.run(test_prog_orig,
                                      feed=feed_dict,
                                      fetch_list=[loss.name])

        with self.program_scope_guard():
            loss = optimization_in_cond_net(True)
            main_program = fluid.default_main_program()
            test_prog_prune = main_program.clone(for_test=True)

            place = core.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            loss_data_prune, = exe.run(test_prog_prune,
                                       feed=feed_dict,
                                       fetch_list=[loss.name])

        self.program_compare(test_prog_orig, test_prog_prune)
        self.assertEqual(loss_data_orig, loss_data_prune)

    @contextlib.contextmanager
    def program_scope_guard(self):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                with fluid.unique_name.guard():
                    yield


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()

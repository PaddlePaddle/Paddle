# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import paddle
import numpy as np
import unittest


def simple_fc_net(image, label, image_shape, class_num, hidden_sizes=[100]):
    hidden = image
    for hidden_size in hidden_sizes:
        hidden = fluid.layers.fc(
            hidden,
            size=hidden_size,
            act='tanh',
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=1.0)))
    hidden = fluid.layers.dropout(x=hidden, dropout_prob=0.1)
    predict_label = fluid.layers.fc(hidden,
                                    size=class_num,
                                    act='softmax',
                                    bias_attr=False)
    loss = fluid.layers.mean(
        fluid.layers.cross_entropy(
            input=predict_label, label=label))
    return loss, predict_label


class TestPyReaderUsingDataLayerTrainAndTest(unittest.TestCase):
    def initParameters(self):
        return True, True, True

    def setUp(self):
        self.image_shape = [3, 224, 224]
        self.class_num = 128
        self.train_batch_size = 128
        self.test_batch_size = 32
        self.save_dirname = '/tmp/py_reader_inference_model'
        self.use_double_buffer, self.use_parallel_executor, self.use_py_reader_infer = self.initParameters(
        )
        print('use_double_buffer', self.use_double_buffer,
              'use_parallel_executor', self.use_parallel_executor,
              'use_py_reader_infer', self.use_py_reader_infer)

    def random_train_reader(self):
        for _ in range(self.train_batch_size * 4):
            yield np.random.uniform(
                low=1, high=2,
                size=self.image_shape), np.random.random_integers(
                    low=0, high=self.class_num / 2 - 1, size=[1])

    def random_test_reader(self):
        for _ in range(self.test_batch_size * 4):
            yield np.random.uniform(
                low=-2, high=-1,
                size=self.image_shape), np.random.random_integers(
                    low=self.class_num / 2, high=self.class_num - 1, size=[1])

    def validate(self, ret, is_train=True):
        if is_train:
            self.assertTrue(np.all(ret[0] >= 1) and np.all(ret[0] <= 2))
            self.assertTrue(
                np.all(ret[1] >= 0) and
                np.all(ret[1] <= self.class_num / 2 - 1))
        else:
            self.assertTrue(np.all(ret[0] >= -2) and np.all(ret[0] <= -1))
            self.assertTrue(
                np.all(ret[1] >= self.class_num / 2) and
                np.all(ret[1] <= self.class_num - 1))

    def test_main(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            with fluid.scope_guard(fluid.core.Scope()):
                self.main()

    def main(self):
        image = fluid.layers.data(
            name='image', dtype='float32', shape=[-1] + self.image_shape)
        label = fluid.layers.data(name='label', dtype='int64', shape=[-1, 1])
        train_reader = fluid.layers.create_py_reader_by_data(
            feed_list=[image, label],
            capacity=16,
            use_double_buffer=self.use_double_buffer)
        loss, predict_label = simple_fc_net(image, label, self.image_shape,
                                            self.class_num)
        test_program = fluid.default_main_program(
        ).prune_read_op_and_convert_for_test()
        for op in test_program.current_block().ops:
            self.assertTrue(
                op.type != 'create_double_buffer_reader' and op.type != 'read',
                "Find read_op or double_buffer_op in cloned program")
        test_startup = fluid.Program()
        with fluid.program_guard(test_program, test_startup):
            test_reader = fluid.layers.create_py_reader_by_data(
                feed_list=[image, label],
                capacity=16,
                use_double_buffer=self.use_double_buffer,
                prepend=True)

        self.assertNotEqual(train_reader.name, test_reader.name)

        optim = fluid.optimizer.SGD(learning_rate=1e-3)
        optim.minimize(loss)

        use_cuda = fluid.core.is_compiled_with_cuda()
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)

        exe.run(fluid.default_startup_program())
        exe.run(test_startup)

        train_reader.decorate_paddle_reader(
            paddle.batch(
                self.random_train_reader, batch_size=self.train_batch_size))
        test_reader.decorate_paddle_reader(
            paddle.batch(
                self.random_test_reader, batch_size=self.test_batch_size))
        if self.use_parallel_executor:
            train_exe = fluid.ParallelExecutor(
                use_cuda=use_cuda, loss_name=loss.name)
            test_exe = fluid.ParallelExecutor(
                use_cuda=use_cuda,
                share_vars_from=train_exe,
                main_program=test_program)
            fetch_list = [image.name, label.name]
        else:
            fetch_list = [image, label]
            train_exe = exe
            test_exe = exe

        for pass_id in range(2):
            train_reader.start()
            batch_id = 0
            while True:
                try:
                    train_ret = train_exe.run(fetch_list=fetch_list)
                    self.validate(train_ret, True)
                    batch_id += 1
                    if batch_id % 2 == 0:
                        for test_pass_id in range(2):
                            test_reader.start()
                            while True:
                                try:
                                    if self.use_parallel_executor:
                                        test_ret = test_exe.run(
                                            fetch_list=fetch_list)
                                    else:
                                        test_ret = test_exe.run(
                                            test_program, fetch_list=fetch_list)
                                    self.validate(test_ret, False)
                                except fluid.core.EOFException:
                                    test_reader.reset()
                                    break
                except fluid.core.EOFException as ex:
                    train_reader.reset()
                    break

        self.target_var_names = [loss.name, predict_label.name]
        fluid.io.save_inference_model(
            dirname=self.save_dirname,
            feeded_var_names=[image.name, label.name],
            target_vars=[loss, predict_label],
            executor=exe)

        with fluid.program_guard(fluid.Program(), fluid.Program()):
            with fluid.scope_guard(fluid.core.Scope()):
                self.infer_main()

    def infer_main(self):
        use_cuda = fluid.core.is_compiled_with_cuda()
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        infer_prog = fluid.io.load_inference_model(
            dirname=self.save_dirname, executor=exe)[0]
        feed_list, target_var_lists = fluid.io.remove_inference_program_feed_fetch_ops(
            infer_prog)
        self.assertEqual(len(feed_list), 2)
        self.assertEqual(feed_list[0].name, 'image')
        self.assertEqual(feed_list[1].name, 'label')
        self.assertEqual(len(target_var_lists), 2)
        self.assertEqual(self.target_var_names[0], target_var_lists[0].name)
        self.assertEqual(self.target_var_names[1], target_var_lists[1].name)

        paddle_reader = paddle.batch(
            self.random_test_reader, batch_size=self.test_batch_size)

        if self.use_py_reader_infer:
            infer_startup_prog = fluid.Program()
            with fluid.program_guard(infer_prog, infer_startup_prog):
                train_reader = fluid.layers.create_py_reader_by_data(
                    feed_list=feed_list,
                    capacity=16,
                    use_double_buffer=self.use_double_buffer,
                    prepend=True)
                train_reader.decorate_paddle_reader(paddle_reader)
            exe.run(infer_startup_prog)
        else:
            feeder = fluid.DataFeeder(
                feed_list=feed_list, place=place, program=infer_prog)

        if self.use_parallel_executor:
            test_exe = fluid.ParallelExecutor(
                use_cuda=use_cuda, main_program=infer_prog)
            fetch_list = [d.name for d in feed_list]
        else:
            test_exe = exe
            fetch_list = feed_list

        with fluid.program_guard(infer_prog, fluid.Program()):
            for _ in range(2):
                if self.use_py_reader_infer:
                    train_reader.start()
                    while True:
                        try:
                            self.validate(
                                test_exe.run(fetch_list=fetch_list), False)
                        except fluid.core.EOFException:
                            train_reader.reset()
                            break
                else:
                    reader = paddle_reader()
                    while True:
                        next_data = next(reader, None)
                        if next_data is None:
                            break
                        self.validate(
                            test_exe.run(fetch_list=fetch_list,
                                         feed=feeder.feed(next_data)),
                            False)


class Test2(TestPyReaderUsingDataLayerTrainAndTest):
    def initParameters(self):
        return True, False, True


class Test3(TestPyReaderUsingDataLayerTrainAndTest):
    def initParameters(self):
        return False, True, True


class Test4(TestPyReaderUsingDataLayerTrainAndTest):
    def initParameters(self):
        return False, False, True


class Test5(TestPyReaderUsingDataLayerTrainAndTest):
    def initParameters(self):
        return True, True, False


class Test6(TestPyReaderUsingDataLayerTrainAndTest):
    def initParameters(self):
        return True, False, False


class Test7(TestPyReaderUsingDataLayerTrainAndTest):
    def initParameters(self):
        return False, True, False


class Test8(TestPyReaderUsingDataLayerTrainAndTest):
    def initParameters(self):
        return False, False, False


if __name__ == '__main__':
    unittest.main()

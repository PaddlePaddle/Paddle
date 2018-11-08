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
    return loss


class TestPyReaderUsingDataLayerTrainAndTest(unittest.TestCase):
    def initParameters(self):
        return True, True

    def setUp(self):
        self.image_shape = [3, 224, 224]
        self.class_num = 128
        self.train_batch_size = 128
        self.test_batch_size = 32
        self.use_double_buffer, self.use_parallel_executor = self.initParameters(
        )
        print('use_double_buffer', self.use_double_buffer,
              'use_parallel_executor', self.use_parallel_executor)

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
        train_reader = fluid.layers.build_py_reader_from_data(
            feed_list=[image, label],
            capacity=16,
            use_double_buffer=self.use_double_buffer)
        loss = simple_fc_net(image, label, self.image_shape, self.class_num)
        test_program = fluid.default_main_program().convert_to_test_program()
        for op in test_program.current_block().ops:
            self.assertTrue(
                op.type != 'create_double_buffer_reader' and op.type != 'read',
                "Find read_op or double_buffer_op in cloned program")
        test_startup = fluid.Program()
        with fluid.program_guard(test_program, test_startup):
            test_reader = fluid.layers.build_py_reader_from_data(
                feed_list=[image, label],
                capacity=16,
                use_double_buffer=self.use_double_buffer)

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


class Test2(TestPyReaderUsingDataLayerTrainAndTest):
    def initParameters(self):
        return True, False


class Test3(TestPyReaderUsingDataLayerTrainAndTest):
    def initParameters(self):
        return False, True


class Test4(TestPyReaderUsingDataLayerTrainAndTest):
    def initParameters(self):
        return False, False


if __name__ == '__main__':
    unittest.main()

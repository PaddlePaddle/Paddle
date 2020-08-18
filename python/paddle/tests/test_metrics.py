# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import print_function

import os
import unittest
import numpy as np

import paddle
import paddle.fluid as fluid

from paddle.incubate.hapi.utils import to_list


def accuracy(pred, label, topk=(1, )):
    maxk = max(topk)
    pred = np.argsort(pred)[:, ::-1][:, :maxk]
    correct = (pred == np.repeat(label, maxk, 1))

    batch_size = label.shape[0]
    res = []
    for k in topk:
        correct_k = correct[:, :k].sum()
        res.append(float(correct_k) / batch_size)
    return res


def convert_to_one_hot(y, C):
    oh = np.random.random((y.shape[0], C)).astype('float32') * .5
    for i in range(y.shape[0]):
        oh[i, int(y[i])] = 1.
    return oh


class TestAccuracy(unittest.TestCase):
    def test_acc(self):
        paddle.disable_static()

        x = paddle.to_tensor(
            np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.4, 0.3, 0.2],
                      [0.1, 0.2, 0.4, 0.3], [0.1, 0.2, 0.3, 0.4]]))
        y = paddle.to_tensor(np.array([[0], [1], [2], [3]]))

        m = paddle.metric.Accuracy(name='my_acc')

        # check name
        self.assertEqual(m.name(), ['my_acc'])

        correct = m.compute(x, y)
        # check results
        self.assertEqual(m.update(correct), 0.75)
        self.assertEqual(m.accumulate(), 0.75)

        x = paddle.to_tensor(
            np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.3, 0.4, 0.2],
                      [0.1, 0.2, 0.4, 0.3], [0.1, 0.2, 0.3, 0.4]]))
        y = paddle.to_tensor(np.array([[0], [1], [2], [3]]))
        correct = m.compute(x, y)
        # check results
        self.assertEqual(m.update(correct), 0.5)
        self.assertEqual(m.accumulate(), 0.625)

        # check reset
        m.reset()
        self.assertEqual(m.total[0], 0.0)
        self.assertEqual(m.count[0], 0.0)
        paddle.enable_static()


class TestAccuracyDynamic(unittest.TestCase):
    def setUp(self):
        self.topk = (1, )
        self.class_num = 5
        self.sample_num = 1000
        self.name = None

    def random_pred_label(self):
        label = np.random.randint(0, self.class_num,
                                  (self.sample_num, 1)).astype('int64')
        pred = np.random.randint(0, self.class_num,
                                 (self.sample_num, 1)).astype('int32')
        pred_one_hot = convert_to_one_hot(pred, self.class_num)
        pred_one_hot = pred_one_hot.astype('float32')

        return label, pred_one_hot

    def test_main(self):
        with fluid.dygraph.guard(fluid.CPUPlace()):
            acc = paddle.metric.Accuracy(topk=self.topk, name=self.name)
            for _ in range(10):
                label, pred = self.random_pred_label()
                label_var = paddle.to_tensor(label)
                pred_var = paddle.to_tensor(pred)
                state = to_list(acc.compute(pred_var, label_var))
                acc.update(* [s.numpy() for s in state])
                res_m = acc.accumulate()
                res_f = accuracy(pred, label, self.topk)
                assert np.all(np.isclose(np.array(res_m, dtype='float64'),
                              np.array(res_f, dtype='float64'), rtol=1e-3)), \
                    "Accuracy precision error: {} != {}".format(res_m, res_f)
                acc.reset()
                assert np.sum(acc.total) == 0
                assert np.sum(acc.count) == 0


class TestAccuracyDynamicMultiTopk(TestAccuracyDynamic):
    def setUp(self):
        self.topk = (1, 5)
        self.class_num = 10
        self.sample_num = 1000
        self.name = "accuracy"


class TestAccuracyStatic(TestAccuracyDynamic):
    def test_main(self):
        main_prog = fluid.Program()
        startup_prog = fluid.Program()
        main_prog.random_seed = 1024
        startup_prog.random_seed = 1024
        with fluid.program_guard(main_prog, startup_prog):
            pred = fluid.data(
                name='pred', shape=[None, self.class_num], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            acc = paddle.metric.Accuracy(topk=self.topk, name=self.name)
            state = acc.compute(pred, label)

        exe = fluid.Executor(fluid.CPUPlace())
        compiled_main_prog = fluid.CompiledProgram(main_prog)

        for _ in range(10):
            label, pred = self.random_pred_label()
            state_ret = exe.run(compiled_main_prog,
                                feed={'pred': pred,
                                      'label': label},
                                fetch_list=[s.name for s in to_list(state)],
                                return_numpy=True)
            acc.update(*state_ret)
            res_m = acc.accumulate()
            res_f = accuracy(pred, label, self.topk)
            assert np.all(np.isclose(np.array(res_m), np.array(res_f), rtol=1e-3)), \
                    "Accuracy precision error: {} != {}".format(res_m, res_f)
            acc.reset()
            assert np.sum(acc.total) == 0
            assert np.sum(acc.count) == 0


class TestAccuracyStaticMultiTopk(TestAccuracyStatic):
    def setUp(self):
        self.topk = (1, 5)
        self.class_num = 10
        self.sample_num = 100
        self.name = "accuracy"


class TestPrecision(unittest.TestCase):
    def test_1d(self):
        paddle.disable_static()

        x = np.array([0.1, 0.5, 0.6, 0.7])
        y = np.array([1, 0, 1, 1])

        m = paddle.metric.Recall()
        m.update(x, y)
        r = m.accumulate()
        self.assertAlmostEqual(r, 2. / 3.)

        x = np.array([0.1, 0.5, 0.6, 0.7, 0.2])
        y = np.array([1, 0, 1, 1, 1])
        m.update(x, y)
        r = m.accumulate()
        self.assertAlmostEqual(r, 4. / 7.)

        paddle.enable_static()

    def test_2d(self):
        paddle.disable_static()

        x = np.array([0.1, 0.5, 0.6, 0.7]).reshape(-1, 1)
        y = np.array([1, 0, 1, 1]).reshape(-1, 1)

        m = paddle.metric.Recall()
        m.update(x, y)
        r = m.accumulate()
        self.assertAlmostEqual(r, 2. / 3.)

        x = np.array([0.1, 0.5, 0.6, 0.7, 0.2]).reshape(-1, 1)
        y = np.array([1, 0, 1, 1, 1]).reshape(-1, 1)
        m.update(x, y)
        r = m.accumulate()
        self.assertAlmostEqual(r, 4. / 7.)

        paddle.enable_static()


class TestRecall(unittest.TestCase):
    def test_1d(self):
        paddle.disable_static()

        x = np.array([0.1, 0.5, 0.6, 0.7])
        y = np.array([1, 0, 1, 1])

        m = paddle.metric.Recall()
        m.update(x, y)
        r = m.accumulate()
        self.assertAlmostEqual(r, 2. / 3.)

        x = np.array([0.1, 0.5, 0.6, 0.7])
        y = np.array([1, 0, 0, 1])
        m.update(x, y)
        r = m.accumulate()
        self.assertAlmostEqual(r, 3. / 5.)

        paddle.enable_static()


class TestAuc(unittest.TestCase):
    def test_auc(self):
        paddle.disable_static()
        x = np.array([[0.78, 0.22], [0.62, 0.38], [0.55, 0.45], [0.30, 0.70],
                      [0.14, 0.86], [0.59, 0.41], [0.91, 0.08], [0.16, 0.84]])
        y = np.array([[0], [1], [1], [0], [1], [0], [0], [1]])
        m = paddle.metric.Auc()
        m.update(x, y)
        r = m.accumulate()
        self.assertAlmostEqual(r, 0.8125)
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()

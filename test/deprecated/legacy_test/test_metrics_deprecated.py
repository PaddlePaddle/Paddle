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

import unittest

import numpy as np

import paddle
from paddle.hapi.model import to_list


def one_hot(x, n_class):
    res = np.eye(n_class)[np.array(x).reshape(-1)]
    res = res.reshape(list(x.shape) + [n_class])
    return res


def accuracy(pred, label, topk=(1,)):
    maxk = max(topk)
    pred = np.argsort(pred)[..., ::-1][..., :maxk]
    if len(label.shape) == 1:
        label = label.reshape(-1, 1)
    elif label.shape[-1] != 1:
        label = np.argmax(label, axis=-1)
        label = label[..., np.newaxis]
    correct = pred == np.repeat(label, maxk, -1)

    total = np.prod(np.array(label.shape[:-1]))

    res = []
    for k in topk:
        correct_k = correct[..., :k].sum()
        res.append(float(correct_k) / total)
    return res


def convert_to_one_hot(y, C):
    oh = np.random.choice(np.arange(C), C, replace=False).astype('float32') / C
    oh = np.tile(oh[np.newaxis, :], (y.shape[0], 1))
    for i in range(y.shape[0]):
        oh[i, int(y[i])] = 1.0
    return oh


class TestAccuracyStatic(unittest.TestCase):
    def setUp(self):
        self.topk = (1,)
        self.class_num = 5
        self.sample_num = 1000
        self.name = None
        self.squeeze_label = True

    def random_pred_label(self):
        label = np.random.randint(
            0, self.class_num, (self.sample_num, 1)
        ).astype('int64')
        pred = np.random.randint(
            0, self.class_num, (self.sample_num, 1)
        ).astype('int32')
        if self.squeeze_label:
            label = label.squeeze()
        pred_one_hot = convert_to_one_hot(pred, self.class_num)
        pred_one_hot = pred_one_hot.astype('float32')

        return label, pred_one_hot

    def test_main(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        paddle.seed(1024)
        with paddle.static.program_guard(main_prog, startup_prog):
            pred = paddle.static.data(
                name='pred', shape=[None, self.class_num], dtype='float32'
            )
            label = paddle.static.data(
                name='label', shape=[None, 1], dtype='int64'
            )
            acc = paddle.metric.Accuracy(topk=self.topk, name=self.name)
            state = acc.compute(pred, label)

        exe = paddle.static.Executor(paddle.CPUPlace())
        compiled_main_prog = paddle.static.CompiledProgram(main_prog)

        for _ in range(10):
            label, pred = self.random_pred_label()
            state_ret = exe.run(
                compiled_main_prog,
                feed={'pred': pred, 'label': label},
                fetch_list=to_list(state),
                return_numpy=True,
            )
            acc.update(*state_ret)
            res_m = acc.accumulate()
            res_f = accuracy(pred, label, self.topk)
            assert np.all(
                np.isclose(np.array(res_m), np.array(res_f), rtol=1e-3)
            ), f"Accuracy precision error: {res_m} != {res_f}"
            acc.reset()
            assert np.sum(acc.total) == 0
            assert np.sum(acc.count) == 0

        paddle.disable_static()


class TestAccuracyStaticMultiTopk(TestAccuracyStatic):
    def setUp(self):
        self.topk = (1, 5)
        self.class_num = 10
        self.sample_num = 100
        self.name = "accuracy"
        self.squeeze_label = False


if __name__ == '__main__':
    unittest.main()

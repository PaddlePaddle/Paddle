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

import unittest
import numpy as np
from op_test import OpTest, randomize_probability
import paddle.fluid as fluid


class TestCrossEntropyOp1(OpTest):
    """Test cross-entropy with discrete one-hot labels.
    """

    def setUp(self):
        self.op_type = "cross_entropy"
        batch_size = 30
        class_num = 10

        X = randomize_probability(batch_size, class_num, dtype='float64')

        label = np.random.randint(0, class_num, (batch_size, 1), dtype="int64")
        cross_entropy = np.asmatrix(
            [[-np.log(X[i][label[i][0]])] for i in range(X.shape[0])],
            dtype="float64")

        self.inputs = {"X": X, "Label": label}
        self.outputs = {"Y": cross_entropy}
        self.attrs = {"soft_label": False}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Y", numeric_grad_delta=0.001)


class TestCrossEntropyOp2(OpTest):
    """Test cross-entropy with vectorized soft labels.
    """

    def setUp(self):
        self.op_type = "cross_entropy"
        batch_size = 5
        class_num = 37

        X = randomize_probability(batch_size, class_num)
        label = np.random.uniform(0.1, 1.0,
                                  [batch_size, class_num]).astype("float32")
        label /= label.sum(axis=1, keepdims=True)
        cross_entropy = (-label * np.log(X)).sum(
            axis=1, keepdims=True).astype("float32")

        self.inputs = {"X": X, "Label": label}
        self.outputs = {"Y": cross_entropy}
        self.attrs = {"soft_label": True}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ["X"], "Y", max_relative_error=0.05, numeric_grad_delta=0.001)


class TestCrossEntropyOp3(OpTest):
    """Test cross-entropy with vectorized one-hot representation of labels.
    """

    def setUp(self):
        self.op_type = "cross_entropy"
        batch_size = 5
        class_num = 17

        X = randomize_probability(batch_size, class_num)
        label_index = np.random.randint(
            0, class_num, (batch_size), dtype="int32")
        label = np.zeros(X.shape)
        label[np.arange(batch_size), label_index] = 1

        cross_entropy = np.asmatrix(
            [[-np.log(X[i][label_index[i]])] for i in range(X.shape[0])],
            dtype="float32")
        cross_entropy2 = (-label * np.log(X)).sum(
            axis=1, keepdims=True).astype("float32")

        self.inputs = {"X": X, "Label": label.astype(np.float32)}
        self.outputs = {"Y": cross_entropy}
        self.attrs = {"soft_label": True}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ["X"], "Y", max_relative_error=0.05, numeric_grad_delta=0.001)


class TestCrossEntropyStable(unittest.TestCase):
    def main(self, place):
        if isinstance(
                place,
                fluid.CUDAPlace) and not fluid.core.is_compiled_with_cuda():
            return

        class DataRandom(object):
            def __init__(self):
                self.random = np.random.RandomState(seed=1)

            def next(self):
                return {
                    'input': self.random.uniform(
                        low=-1, high=1, size=(64, 200)).astype('float32'),
                    'label': self.random.uniform(
                        low=0, high=10000, size=(64, 1)).astype('int64'),
                }

        losses = []
        for _ in xrange(2):
            startup = fluid.Program()
            startup.random_seed = 1
            main = fluid.Program()
            scope = fluid.core.Scope()
            with fluid.scope_guard(scope):
                with fluid.program_guard(main, startup):
                    img = fluid.layers.data('input', shape=[200])
                    label = fluid.layers.data('label', shape=[1], dtype='int64')
                    prediction = fluid.layers.fc(input=img,
                                                 size=10000,
                                                 act='softmax')
                    xe = fluid.layers.cross_entropy(
                        input=prediction, label=label)
                    loss = fluid.layers.mean(xe)
                    adam = fluid.optimizer.Adam()
                    adam.minimize(loss)

                    exe = fluid.Executor(place)
                    exe.run(startup)
                    data = DataRandom()
                    for i in xrange(1000):
                        exe.run(feed=next(data))
                    losses.append(
                        exe.run(feed=next(data), fetch_list=[loss])[0])
        print losses
        self.assertAlmostEqual(losses[0][0], losses[1][0])

    def test_cpu(self):
        self.main(fluid.CPUPlace())

    def test_cuda(self):
        self.main(fluid.CUDAPlace(0))


if __name__ == "__main__":
    unittest.main()

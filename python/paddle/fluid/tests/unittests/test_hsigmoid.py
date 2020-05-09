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

from paddle import fluid, nn
import paddle.fluid.dygraph as dg
import paddle.nn.functional as F
import paddle.fluid.initializer as I
import numpy as np
import unittest


class HSigmoidTestCase(unittest.TestCase):
    def __init__(self,
                 methodName="runTest",
                 batch_size=4,
                 feature_size=6,
                 num_classes=8,
                 labels=None,
                 path_code=None,
                 path_table=None,
                 is_sparse=False,
                 dtype="float32"):
        super(HSigmoidTestCase, self).__init__()
        self.batch_size = batch_size
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.dtype = dtype
        self.is_sparse = is_sparse

        self.labels = labels
        self.path_code = path_code
        self.path_table = path_table
        self.is_custom = path_code is not None and path_table is not None

    def setUp(self):
        input_shape = (self.batch_size, self.feature_size)
        self.input = np.random.uniform(
            -1, 1, size=input_shape).astype(self.dtype)
        if self.labels is None:
            self.labels = np.random.randint(
                0, self.num_classes, size=(self.batch_size, 1)).astype(np.int64)
        C = self.num_classes if self.is_custom else self.num_classes - 1
        self.weight_shape = (C, self.feature_size)
        self.weight = np.random.randn(*self.weight_shape).astype(self.dtype)
        self.bias_shape = (C, 1)
        self.bias = np.random.randn(*self.bias_shape).astype(self.dtype)

    def fluid_layer(self, place):
        main = fluid.Program()
        start = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, start):
                x = fluid.data(
                    "input", [-1, self.feature_size], dtype=self.dtype)
                label = fluid.data("labels", [-1, 1], dtype="int64")
                if self.is_custom:
                    path_table = fluid.data(
                        "path_table", [-1, -1], dtype="int64")
                    path_code = fluid.data("path_code", [-1, -1], dtype="int64")
                else:
                    path_table = path_code = None
                y = fluid.layers.hsigmoid(
                    x,
                    label,
                    self.num_classes,
                    param_attr=I.NumpyArrayInitializer(self.weight),
                    bias_attr=I.NumpyArrayInitializer(self.bias),
                    path_table=path_table,
                    path_code=path_code,
                    is_custom=self.is_custom,
                    is_sparse=self.is_sparse, )
        exe = fluid.Executor(place)
        exe.run(start)
        feed_dict = {"input": self.input, "labels": self.labels}
        if self.is_custom:
            feed_dict["path_code"] = self.path_code
            feed_dict["path_table"] = self.path_table
        y_np, = exe.run(main, feed=feed_dict, fetch_list=[y])
        return y_np

    def functional(self, place):
        main = fluid.Program()
        start = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, start):
                x = fluid.data(
                    "input", [-1, self.feature_size], dtype=self.dtype)
                label = fluid.data("labels", [-1, 1], dtype="int64")
                if self.is_custom:
                    path_table = fluid.data(
                        "path_table", [-1, -1], dtype="int64")
                    path_code = fluid.data("path_code", [-1, -1], dtype="int64")
                else:
                    path_table = path_code = None
                w = fluid.data("weight", self.weight_shape, dtype=self.dtype)
                b = fluid.data("bias", self.bias_shape, dtype=self.dtype)
                y = F.hsigmoid(
                    x,
                    label,
                    w,
                    b,
                    self.num_classes,
                    is_sparse=self.is_sparse,
                    path_table=path_table,
                    path_code=path_code)

        exe = fluid.Executor(place)
        exe.run(start)
        feed_dict = {
            "input": self.input,
            "labels": self.labels,
            "weight": self.weight,
            "bias": self.bias
        }
        if self.is_custom:
            feed_dict["path_code"] = self.path_code
            feed_dict["path_table"] = self.path_table
        y_np, = exe.run(main, feed=feed_dict, fetch_list=[y])
        return y_np

    def nn_layer(self, place):
        with dg.guard(place):
            x_var = dg.to_variable(self.input)
            label_var = dg.to_variable(self.labels)
            if self.is_custom:
                path_code_var = dg.to_variable(self.path_code)
                path_table_var = dg.to_variable(self.path_table)
            else:
                path_code_var = path_table_var = None
            hierarchical_softmax = nn.HSigmoid(
                self.feature_size,
                self.num_classes,
                is_custom=self.is_custom,
                is_sparse=self.is_sparse,
                param_attr=I.NumpyArrayInitializer(self.weight),
                bias_attr=I.NumpyArrayInitializer(self.bias),
                dtype=self.dtype)
            y_var = hierarchical_softmax(
                x_var,
                label_var,
                path_table=path_table_var,
                path_code=path_code_var)
            y_np = y_var.numpy()
        return y_np

    def _test_equivalence(self, place):
        result1 = self.fluid_layer(place)
        result2 = self.functional(place)
        result3 = self.nn_layer(place)
        np.testing.assert_array_almost_equal(result1, result2)
        np.testing.assert_array_almost_equal(result2, result3)

    def runTest(self):
        place = fluid.CPUPlace()
        self._test_equivalence(place)


class HSigmoidTestErrorCase(HSigmoidTestCase):
    def runTest(self):
        place = fluid.CPUPlace()
        with dg.guard(place):
            with self.assertRaises(ValueError):
                self.nn_layer()

    def nn_layer(self):
        x_var = dg.to_variable(self.input)
        label_var = dg.to_variable(self.labels)
        if self.is_custom:
            path_code_var = dg.to_variable(self.path_code)
            path_table_var = dg.to_variable(self.path_table)
        else:
            path_code_var = path_table_var = None
        hierarchical_softmax = nn.HSigmoid(
            self.feature_size,
            self.num_classes,
            is_custom=self.is_custom,
            param_attr=I.NumpyArrayInitializer(self.weight),
            bias_attr=I.NumpyArrayInitializer(self.bias),
            dtype=self.dtype)
        y_var = hierarchical_softmax(
            x_var,
            label_var,
            path_table=path_table_var,
            path_code=path_code_var)
        y_np = y_var.numpy()
        return y_np


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    suite.addTest(HSigmoidTestCase(methodName="runTest"))
    suite.addTest(
        HSigmoidTestCase(
            methodName="runTest",
            batch_size=4,
            feature_size=6,
            num_classes=8,
            labels=np.array([0, 1, 4, 5]).astype(np.int64),
            path_table=np.array([(0, 2, -1, -1, -1), (0, 1, 3, -1, -1), (
                0, 1, 4, -1, -1), (0, 2, -1, -1, -1)]).astype(np.int64),
            path_code=np.array([(0, 0, -1, -1, -1), (1, 1, 1, -1, -1), (
                1, 0, 0, -1, -1), (0, 1, -1, -1, -1)]).astype(np.int64)))
    suite.addTest(HSigmoidTestErrorCase(methodName="runTest", num_classes=1))
    return suite


if __name__ == "__main__":
    unittest.main()

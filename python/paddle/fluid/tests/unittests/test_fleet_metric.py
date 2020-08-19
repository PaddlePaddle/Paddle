#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Test fleet metric."""

from __future__ import print_function
import numpy as np
import paddle
import paddle.fluid as fluid
import os
import unittest
import paddle.distributed.fleet.metrics.metric as metric
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet


class TestFleetMetric(unittest.TestCase):
    """Test cases for fleet metric."""

    def setUp(self):
        """Set up, set envs."""

        class FakeFleet:
            """Fake fleet only for test."""

            def __init__(self):
                """Init."""
                self.gloo = fluid.core.Gloo()
                self.gloo.set_rank(0)
                self.gloo.set_size(1)
                self.gloo.set_prefix("123")
                self.gloo.set_iface("lo")
                self.gloo.set_hdfs_store("./tmp_test_metric", "", "")
                self.gloo.init()

            def _all_reduce(self, input, output, mode="sum"):
                """All reduce using gloo."""
                input_list = [i for i in input]
                ans = self.gloo.all_reduce(input_list, mode)
                for i in range(len(ans)):
                    output[i] = 1

            def _barrier_worker(self):
                """Fake barrier worker, do nothing."""
                pass

        self.fleet = FakeFleet()
        fleet._role_maker = self.fleet

    def test_metric_1(self):
        """Test cases for metrics."""
        train = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(train, startup):
            t = fluid.layers.create_global_var(
                shape=[1, 1],
                value=1,
                dtype='int64',
                persistable=True,
                force_cpu=True)
            t1 = fluid.layers.create_global_var(
                shape=[1, 1],
                value=1,
                dtype='int64',
                persistable=True,
                force_cpu=True)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup)
            metric.sum(t, scope)
            metric.max(t, scope)
            metric.min(t, scope)
            metric.auc(t, t1, scope)
            metric.mae(t1, 3, scope)
            metric.rmse(t1, 3, scope)
            metric.mse(t1, 3, scope)
            metric.acc(t, t1, scope)
            metric.sum(str(t.name), scope)
            metric.max(str(t.name), scope)
            metric.min(str(t.name), scope)
            metric.auc(str(t1.name), str(t.name), scope)
            metric.mae(str(t1.name), 3, scope)
            metric.rmse(str(t1.name), 3, scope)
            metric.mse(str(t1.name), 3, scope)
            metric.acc(str(t.name), str(t1.name), scope)
        arr = np.array([1, 2, 3, 4])
        metric.sum(arr)
        metric.max(arr)
        metric.min(arr)
        arr1 = np.array([[1, 2, 3, 4]])
        arr2 = np.array([[1, 2, 3, 4]])
        arr3 = np.array([1, 2, 3, 4])
        metric.auc(arr1, arr2)
        metric.mae(arr, 3)
        metric.rmse(arr, 3)
        metric.mse(arr, 3)
        metric.acc(arr, arr3)


if __name__ == "__main__":
    unittest.main()

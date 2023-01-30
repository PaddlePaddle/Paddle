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

<<<<<<< HEAD
import os
import unittest

import numpy as np

os.environ['FLAGS_enable_parallel_graph'] = str(1)
import os

from parallel_executor_test_base import DeviceType, TestParallelExecutorBase
from simple_nets import init_data, simple_fc_net

import paddle.fluid.core as core


class TestMNIST(TestParallelExecutorBase):
=======
from __future__ import print_function

import unittest

import numpy as np
import os

os.environ['FLAGS_enable_parallel_graph'] = str(1)
import paddle.fluid.core as core
import os
from parallel_executor_test_base import TestParallelExecutorBase, DeviceType
from simple_nets import simple_fc_net, init_data


class TestMNIST(TestParallelExecutorBase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)

    # simple_fc
    def check_simple_fc_convergence(self, use_device, use_reduce=False):
        if use_device == DeviceType.CUDA and not core.is_compiled_with_cuda():
            return

        img, label = init_data()
<<<<<<< HEAD
        self.check_network_convergence(
            simple_fc_net,
            feed_dict={"image": img, "label": label},
            use_device=use_device,
            use_reduce=use_reduce,
        )
=======
        self.check_network_convergence(simple_fc_net,
                                       feed_dict={
                                           "image": img,
                                           "label": label
                                       },
                                       use_device=use_device,
                                       use_reduce=use_reduce)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_simple_fc(self):
        # use_device
        self.check_simple_fc_convergence(True)

    def check_simple_fc_parallel_accuracy(self, use_device):
        if use_device and not core.is_compiled_with_cuda():
            return

        img, label = init_data()
        single_first_loss, single_last_loss, _ = self.check_network_convergence(
            method=simple_fc_net,
<<<<<<< HEAD
            feed_dict={"image": img, "label": label},
            use_device=use_device,
            use_parallel_executor=False,
        )
        (
            parallel_first_loss,
            parallel_last_loss,
            _,
        ) = self.check_network_convergence(
            method=simple_fc_net,
            feed_dict={"image": img, "label": label},
            use_device=use_device,
            use_parallel_executor=True,
        )

        self.assertAlmostEqual(
=======
            feed_dict={
                "image": img,
                "label": label
            },
            use_device=use_device,
            use_parallel_executor=False)
        parallel_first_loss, parallel_last_loss, _ = self.check_network_convergence(
            method=simple_fc_net,
            feed_dict={
                "image": img,
                "label": label
            },
            use_device=use_device,
            use_parallel_executor=True)

        self.assertAlmostEquals(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            np.mean(parallel_first_loss),
            single_first_loss,
            delta=1e-6,
        )
<<<<<<< HEAD
        self.assertAlmostEqual(
            np.mean(parallel_last_loss), single_last_loss, delta=1e-6
        )
=======
        self.assertAlmostEquals(np.mean(parallel_last_loss),
                                single_last_loss,
                                delta=1e-6)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_simple_fc_parallel_accuracy(self):
        self.check_simple_fc_parallel_accuracy(DeviceType.CUDA)


if __name__ == '__main__':
    unittest.main()

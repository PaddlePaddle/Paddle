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

from simple_nets import simple_fc_net, fc_with_batchnorm, init_data
from parallel_executor_test_base import TestParallelExecutorBase
import paddle.fluid as fluid
import paddle.fluid.core as core
import unittest
import os


class TestMNIST(TestParallelExecutorBase):
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)

    def _compare_fuse_elewise_add_act_ops(self, model, use_cuda):
        if use_cuda and not core.is_compiled_with_cuda():
            return
        img, label = init_data()

        def _optimizer(learning_rate=1e-6):
            optimizer = fluid.optimizer.SGD(
                learning_rate=learning_rate,
                regularization=fluid.regularizer.L2Decay(1e-6))
            return optimizer

        # NOTE(dzh):
        # need to make it compatible with elewise fuse act
        # FIXME (liuwei12)
        # the new memory optimize strategy will crash this unittest
        # add enable_inplace=False here to force pass the unittest
        not_fuse_op_first_loss, not_fuse_op_last_loss = self.check_network_convergence(
            model,
            feed_dict={"image": img,
                       "label": label},
            use_cuda=use_cuda,
            fuse_elewise_add_act_ops=False,
            memory_opt=False,
            use_ir_memory_optimize=False,
            enable_inplace=False,
            optimizer=_optimizer)
        fuse_op_first_loss, fuse_op_last_loss = self.check_network_convergence(
            model,
            feed_dict={"image": img,
                       "label": label},
            use_cuda=use_cuda,
            fuse_elewise_add_act_ops=True,
            memory_opt=False,
            use_ir_memory_optimize=False,
            enable_inplace=False,
            optimizer=_optimizer)

        for loss in zip(not_fuse_op_first_loss, fuse_op_first_loss):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-6)
        for loss in zip(not_fuse_op_last_loss, fuse_op_last_loss):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-6)

    def test_simple_fc_with_fuse_op(self):
        self._compare_fuse_elewise_add_act_ops(simple_fc_net, True)
        self._compare_fuse_elewise_add_act_ops(simple_fc_net, False)

    def test_batchnorm_fc_with_fuse_op(self):
        self._compare_fuse_elewise_add_act_ops(fc_with_batchnorm, True)
        self._compare_fuse_elewise_add_act_ops(fc_with_batchnorm, False)


if __name__ == '__main__':
    unittest.main()

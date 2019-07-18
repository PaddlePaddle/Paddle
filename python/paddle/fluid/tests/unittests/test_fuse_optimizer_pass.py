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


class TestFuseAdamOps(TestParallelExecutorBase):
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)

    def _compare_fused_optimizer_ops(self,
                                     model,
                                     use_cuda,
                                     optimizer=fluid.optimizer.Adam):
        if use_cuda and not core.is_compiled_with_cuda():
            return
        img, label = init_data()
        feed_dict = {"image": img, "label": label}
        not_fuse_op_first_loss, not_fuse_op_last_loss = self.check_network_convergence(
            model,
            feed_dict=feed_dict,
            use_cuda=use_cuda,
            fuse_all_optimizer_ops=False,
            memory_opt=False,  # avoid the gradient's name changed in Python side.
            optimizer=optimizer)
        fuse_op_first_loss, fuse_op_last_loss = self.check_network_convergence(
            model,
            feed_dict=feed_dict,
            use_cuda=use_cuda,
            fuse_all_optimizer_ops=True,
            memory_opt=False,  # avoid the gradient's name changed in Python side.
            optimizer=optimizer)

        for loss in zip(not_fuse_op_first_loss, fuse_op_first_loss):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-6)
        for loss in zip(not_fuse_op_last_loss, fuse_op_last_loss):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-6)

    def test_simple_fc_with_fuse_op(self):
        self._compare_fused_optimizer_ops(simple_fc_net, True)
        self._compare_fused_optimizer_ops(simple_fc_net, False)

    def test_batchnorm_fc_with_fuse_op(self):
        self._compare_fused_optimizer_ops(fc_with_batchnorm, True)
        self._compare_fused_optimizer_ops(fc_with_batchnorm, False)


class TestFuseSGDOps(TestFuseAdamOps):
    def sgd_optimizer(self, learning_rate=1e-3):
        return fluid.optimizer.SGD(learning_rate=learning_rate)

    def test_simple_fc_with_fuse_op(self):
        self._compare_fused_optimizer_ops(
            simple_fc_net, True, optimizer=self.sgd_optimizer)
        self._compare_fused_optimizer_ops(
            simple_fc_net, False, optimizer=self.sgd_optimizer)

    def test_batchnorm_fc_with_fuse_op(self):
        self._compare_fused_optimizer_ops(
            fc_with_batchnorm, True, optimizer=self.sgd_optimizer)
        self._compare_fused_optimizer_ops(
            fc_with_batchnorm, False, optimizer=self.sgd_optimizer)


class TestFuseMomentumOps(TestFuseAdamOps):
    def momentum_optimizer(self, learning_rate=1e-3):
        return fluid.optimizer.Momentum(
            learning_rate=learning_rate, momentum=0.1)

    def test_simple_fc_with_fuse_op(self):
        self._compare_fused_optimizer_ops(
            simple_fc_net, True, optimizer=self.momentum_optimizer)
        self._compare_fused_optimizer_ops(
            simple_fc_net, False, optimizer=self.momentum_optimizer)

    def test_batchnorm_fc_with_fuse_op(self):
        self._compare_fused_optimizer_ops(
            fc_with_batchnorm, True, optimizer=self.momentum_optimizer)
        self._compare_fused_optimizer_ops(
            fc_with_batchnorm, False, optimizer=self.momentum_optimizer)


if __name__ == '__main__':
    unittest.main()

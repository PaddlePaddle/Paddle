# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from parallel_executor_test_base import TestParallelExecutorBase, DeviceType
import seresnext_net
import paddle.fluid.core as core


class TestResnetWithReduceBase(TestParallelExecutorBase):

    def _compare_reduce_and_allreduce(self, use_device, delta2=1e-5):
        if use_device == DeviceType.CUDA and not core.is_compiled_with_cuda():
            return

        all_reduce_first_loss, all_reduce_last_loss, _ = self.check_network_convergence(
            seresnext_net.model,
            feed_dict=seresnext_net.feed_dict(use_device),
            iter=seresnext_net.iter(use_device),
            batch_size=seresnext_net.batch_size(use_device),
            use_device=use_device,
            use_reduce=False,
            optimizer=seresnext_net.optimizer)
        reduce_first_loss, reduce_last_loss, _ = self.check_network_convergence(
            seresnext_net.model,
            feed_dict=seresnext_net.feed_dict(use_device),
            iter=seresnext_net.iter(use_device),
            batch_size=seresnext_net.batch_size(use_device),
            use_device=use_device,
            use_reduce=True,
            optimizer=seresnext_net.optimizer)

        for loss in zip(all_reduce_first_loss, reduce_first_loss):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-5)
        for loss in zip(all_reduce_last_loss, reduce_last_loss):
            self.assertAlmostEquals(loss[0], loss[1], delta=loss[0] * delta2)

        if not use_device:
            return

        all_reduce_first_loss_seq, all_reduce_last_loss_seq, _ = self.check_network_convergence(
            seresnext_net.model,
            feed_dict=seresnext_net.feed_dict(use_device),
            iter=seresnext_net.iter(use_device),
            batch_size=seresnext_net.batch_size(use_device),
            use_device=use_device,
            use_reduce=False,
            optimizer=seresnext_net.optimizer,
            enable_sequential_execution=True)

        reduce_first_loss_seq, reduce_last_loss_seq, _ = self.check_network_convergence(
            seresnext_net.model,
            feed_dict=seresnext_net.feed_dict(use_device),
            iter=seresnext_net.iter(use_device),
            batch_size=seresnext_net.batch_size(use_device),
            use_device=use_device,
            use_reduce=True,
            optimizer=seresnext_net.optimizer,
            enable_sequential_execution=True)

        for loss in zip(all_reduce_first_loss, all_reduce_first_loss_seq):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-5)
        for loss in zip(all_reduce_last_loss, all_reduce_last_loss_seq):
            self.assertAlmostEquals(loss[0], loss[1], delta=loss[0] * delta2)

        for loss in zip(reduce_first_loss, reduce_first_loss_seq):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-5)
        for loss in zip(reduce_last_loss, reduce_last_loss_seq):
            self.assertAlmostEquals(loss[0], loss[1], delta=loss[0] * delta2)

        for loss in zip(all_reduce_first_loss_seq, reduce_first_loss_seq):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-5)
        for loss in zip(all_reduce_last_loss_seq, reduce_last_loss_seq):
            self.assertAlmostEquals(loss[0], loss[1], delta=loss[0] * delta2)


class TestResnetWithReduceCPU(TestResnetWithReduceBase):

    def test_seresnext_with_reduce(self):
        self._compare_reduce_and_allreduce(use_device=DeviceType.CPU,
                                           delta2=1e-3)


if __name__ == '__main__':
    unittest.main()

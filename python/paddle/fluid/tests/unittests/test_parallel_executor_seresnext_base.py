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

from __future__ import print_function

import paddle.fluid.core as core
from parallel_executor_test_base import TestParallelExecutorBase
import unittest
from seresnext_base_model import model, _feed_dict, _iter, _batch_size, optimizer


class TestResnet(TestParallelExecutorBase):
    def _compare_reduce_and_allreduce(self, use_cuda, delta2=1e-5):
        if use_cuda and not core.is_compiled_with_cuda():
            return
        all_reduce_first_loss, all_reduce_last_loss = self.check_network_convergence(
            model,
            feed_dict=_feed_dict(use_cuda),
            iter=_iter(use_cuda),
            batch_size=_batch_size(),
            use_cuda=use_cuda,
            use_reduce=False,
            optimizer=optimizer)
        reduce_first_loss, reduce_last_loss = self.check_network_convergence(
            model,
            feed_dict=_feed_dict(use_cuda),
            iter=_iter(use_cuda),
            batch_size=_batch_size(),
            use_cuda=use_cuda,
            use_reduce=True,
            optimizer=optimizer)

        for loss in zip(all_reduce_first_loss, reduce_first_loss):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-5)
        for loss in zip(all_reduce_last_loss, reduce_last_loss):
            self.assertAlmostEquals(loss[0], loss[1], delta=delta2)

        if not use_cuda:
            return

        all_reduce_first_loss_seq, all_reduce_last_loss_seq = self.check_network_convergence(
            model,
            feed_dict=_feed_dict(use_cuda),
            iter=_iter(use_cuda),
            batch_size=_batch_size(),
            use_cuda=use_cuda,
            use_reduce=False,
            optimizer=optimizer,
            enable_sequential_execution=True)

        reduce_first_loss_seq, reduce_last_loss_seq = self.check_network_convergence(
            model,
            feed_dict=_feed_dict(use_cuda),
            iter=_iter(use_cuda),
            batch_size=_batch_size(),
            use_cuda=use_cuda,
            use_reduce=True,
            optimizer=optimizer,
            enable_sequential_execution=True)

        for loss in zip(all_reduce_first_loss, all_reduce_first_loss_seq):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-5)
        for loss in zip(all_reduce_last_loss, all_reduce_last_loss_seq):
            self.assertAlmostEquals(loss[0], loss[1], delta=delta2)

        for loss in zip(reduce_first_loss, reduce_first_loss_seq):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-5)
        for loss in zip(reduce_last_loss, reduce_last_loss_seq):
            self.assertAlmostEquals(loss[0], loss[1], delta=delta2)

        for loss in zip(all_reduce_first_loss_seq, reduce_first_loss_seq):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-5)
        for loss in zip(all_reduce_last_loss_seq, reduce_last_loss_seq):
            self.assertAlmostEquals(loss[0], loss[1], delta=delta2)

    def test_seresnext_with_reduce(self):
        self._compare_reduce_and_allreduce(use_cuda=False, delta2=1e-3)
        self._compare_reduce_and_allreduce(use_cuda=True, delta2=1e-2)


if __name__ == '__main__':
    unittest.main()

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

import numpy as np
import unittest

import paddle

# used by model.run_trainer in test_dist_base
from test_dist_base import RUN_STEP
from paddle.fluid.framework import _test_eager_guard


# NOTE: compatible TestParallelDyGraphRunnerBase args
class SpawnAssistTestArgs(object):
    update_method = "local"
    trainer_id = 0
    find_unused_parameters = False


class TestDistSpawnRunner(unittest.TestCase):

    def setUp(self):
        # NOTE(chenweihang): keep consistent with
        # TestDistBase.check_with_place
        self.nprocs = 2

    def _run(self, model, args):
        args.update_method = "local"
        return model.run_trainer_with_spawn(args)

    def _run_parallel(self, model, args):
        args.update_method = "nccl2"
        context = paddle.distributed.spawn(func=model.run_trainer_with_spawn,
                                           args=(args, ),
                                           nprocs=self.nprocs,
                                           join=True)
        result_list = []
        for res_queue in context.return_queues:
            result_list.append(res_queue.get())
        return result_list

    def check_dist_result_with_spawn(self, test_class, delta=1e-3):
        with _test_eager_guard():
            self.check_dist_result_with_spawn_func(test_class=test_class,
                                                   delta=delta)
        self.check_dist_result_with_spawn_func(test_class=test_class,
                                               delta=delta)

    def check_dist_result_with_spawn_func(self, test_class, delta=1e-3):
        # 0. prepare model and args
        model = test_class()
        args = SpawnAssistTestArgs()

        # 1. calc signal card loss
        losses = self._run(model, args)

        # 2. calc multi card loss (nccl mode)
        dist_losses_list = self._run_parallel(model, args)

        # 3. compare losses
        for step_id in range(RUN_STEP):
            loss = losses[step_id]
            dist_loss_sum = None
            for dist_losses in dist_losses_list:
                if dist_loss_sum is None:
                    dist_loss_sum = np.array(dist_losses[step_id])
                else:
                    dist_loss_sum += np.array(dist_losses[step_id])
            dist_loss = dist_loss_sum / self.nprocs
            self.assertAlmostEqual(
                loss,
                dist_loss,
                delta=delta,
                msg=
                "The results of single-card execution and multi-card execution are inconsistent."
                "signal-card loss is:\n{}\nmulti-card average loss is:\n{}\n".
                format(loss, dist_loss))

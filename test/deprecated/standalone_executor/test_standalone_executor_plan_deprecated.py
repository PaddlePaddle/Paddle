# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import static
from paddle.base import core


class TestStandaloneExecutorPlan(unittest.TestCase):
    def test_standalone_executor_plan(self):
        micro_batch_id = 0
        forward_job = core.Job("forward")
        backward_job = core.Job("backward")
        optimizer_job = core.Job("optimizer")
        forward_job.set_micro_batch_id(micro_batch_id)
        backward_job.set_micro_batch_id(micro_batch_id)
        optimizer_job.set_micro_batch_id(micro_batch_id)
        self.assertEqual(forward_job.micro_batch_id(), micro_batch_id)
        self.assertEqual(forward_job.type(), "forward")

        forward_program = static.Program()
        backward_program = static.Program()
        optimizer_program = static.Program()
        job_list = [forward_job, backward_job, optimizer_job]
        type_to_program = {
            "forward": forward_program.desc,
            "backward": backward_program.desc,
            "optimizer": optimizer_program.desc,
        }
        plan = core.Plan(job_list, type_to_program)
        self.assertEqual(plan.job_list(), job_list)
        for type in type_to_program.keys():
            self.assertEqual(plan.program(type), type_to_program[type])


if __name__ == '__main__':
    unittest.main()

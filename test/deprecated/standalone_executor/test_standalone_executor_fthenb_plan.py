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
from paddle.distributed.passes import PassContext, new_pass


class TestStandaloneExecutorFThenBPlan(unittest.TestCase):
    def test_standalone_executor_fthenb_plan(self):
        config = {}
        config["num_micro_batches"] = 4
        pass_context = PassContext()

        startup_program = static.Program()
        main_program = static.Program()

        pipeline_fthenb_pass = new_pass("pipeline_scheduler_FThenB", config)
        pipeline_fthenb_pass.apply(
            [main_program], [startup_program], pass_context
        )
        plan = pass_context.get_attr("plan")
        job_type_list = []
        for job in plan.job_list():
            job_type_list.append(job.type())
        expect_job_type_list = [
            "forward",
            "forward",
            "forward",
            "forward",
            "backward",
            "backward",
            "backward",
            "backward",
            "optimizer",
        ]
        self.assertEqual(job_type_list, expect_job_type_list)


if __name__ == '__main__':
    unittest.main()

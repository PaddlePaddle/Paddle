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

from paddle import base, static
from paddle.distributed.passes import PassContext, new_pass


class TestStandaloneExecutor1F1BPlan(unittest.TestCase):
    def test_standalone_executor_1f1b_plan_stage0(self):
        base.set_flags({'FLAGS_enable_pir_api': 0})
        config = {"num_micro_batches": 8, "pp_stage": 0, "pp_degree": 4}
        pass_context = PassContext()

        startup_program = static.Program()
        main_program = static.Program()

        pipeline_1f1b_pass = new_pass("pipeline_scheduler_1F1B", config)
        pipeline_1f1b_pass.apply(
            [main_program], [startup_program], pass_context
        )
        plan = pass_context.get_attr("plan")
        job_type_list = []
        micro_batch_id_list = []
        for job in plan.job_list():
            job_type_list.append(job.type())
            micro_batch_id_list.append(job.micro_batch_id())
        expect_job_type_list = [
            "forward",
            "forward",
            "forward",
            "forward",
            "backward",
            "forward",
            "backward",
            "forward",
            "backward",
            "forward",
            "backward",
            "forward",
            "backward",
            "backward",
            "backward",
            "backward",
            "optimizer",
        ]
        expect_micro_batch_id_list = [
            0,
            1,
            2,
            3,
            0,
            4,
            1,
            5,
            2,
            6,
            3,
            7,
            4,
            5,
            6,
            7,
            0,
        ]
        self.assertEqual(job_type_list, expect_job_type_list)
        self.assertEqual(micro_batch_id_list, expect_micro_batch_id_list)

    def test_standalone_executor_1f1b_plan_stage1(self):
        base.set_flags({'FLAGS_enable_pir_api': 0})
        config = {"num_micro_batches": 8, "pp_stage": 1, "pp_degree": 4}
        pass_context = PassContext()

        startup_program = static.Program()
        main_program = static.Program()

        pipeline_1f1b_pass = new_pass("pipeline_scheduler_1F1B", config)
        pipeline_1f1b_pass.apply(
            [main_program], [startup_program], pass_context
        )
        plan = pass_context.get_attr("plan")
        job_type_list = []
        micro_batch_id_list = []
        for job in plan.job_list():
            job_type_list.append(job.type())
            micro_batch_id_list.append(job.micro_batch_id())
        expect_job_type_list = [
            "forward",
            "forward",
            "forward",
            "backward",
            "forward",
            "backward",
            "forward",
            "backward",
            "forward",
            "backward",
            "forward",
            "backward",
            "forward",
            "backward",
            "backward",
            "backward",
            "optimizer",
        ]
        expect_micro_batch_id_list = [
            0,
            1,
            2,
            0,
            3,
            1,
            4,
            2,
            5,
            3,
            6,
            4,
            7,
            5,
            6,
            7,
            0,
        ]
        self.assertEqual(job_type_list, expect_job_type_list)
        self.assertEqual(micro_batch_id_list, expect_micro_batch_id_list)

    def test_standalone_executor_1f1b_plan_stage2(self):
        base.set_flags({'FLAGS_enable_pir_api': 0})
        config = {"num_micro_batches": 8, "pp_stage": 2, "pp_degree": 4}
        pass_context = PassContext()

        startup_program = static.Program()
        main_program = static.Program()

        pipeline_1f1b_pass = new_pass("pipeline_scheduler_1F1B", config)
        pipeline_1f1b_pass.apply(
            [main_program], [startup_program], pass_context
        )
        plan = pass_context.get_attr("plan")
        job_type_list = []
        micro_batch_id_list = []
        for job in plan.job_list():
            job_type_list.append(job.type())
            micro_batch_id_list.append(job.micro_batch_id())
        expect_job_type_list = [
            "forward",
            "forward",
            "backward",
            "forward",
            "backward",
            "forward",
            "backward",
            "forward",
            "backward",
            "forward",
            "backward",
            "forward",
            "backward",
            "forward",
            "backward",
            "backward",
            "optimizer",
        ]
        expect_micro_batch_id_list = [
            0,
            1,
            0,
            2,
            1,
            3,
            2,
            4,
            3,
            5,
            4,
            6,
            5,
            7,
            6,
            7,
            0,
        ]
        self.assertEqual(job_type_list, expect_job_type_list)
        self.assertEqual(micro_batch_id_list, expect_micro_batch_id_list)

    def test_standalone_executor_1f1b_plan_stage3(self):
        base.set_flags({'FLAGS_enable_pir_api': 0})
        config = {"num_micro_batches": 8, "pp_stage": 3, "pp_degree": 4}
        pass_context = PassContext()

        startup_program = static.Program()
        main_program = static.Program()

        pipeline_1f1b_pass = new_pass("pipeline_scheduler_1F1B", config)
        pipeline_1f1b_pass.apply(
            [main_program], [startup_program], pass_context
        )
        plan = pass_context.get_attr("plan")
        job_type_list = []
        micro_batch_id_list = []
        for job in plan.job_list():
            job_type_list.append(job.type())
            micro_batch_id_list.append(job.micro_batch_id())
        expect_job_type_list = [
            "forward",
            "backward",
            "forward",
            "backward",
            "forward",
            "backward",
            "forward",
            "backward",
            "forward",
            "backward",
            "forward",
            "backward",
            "forward",
            "backward",
            "forward",
            "backward",
            "optimizer",
        ]
        expect_micro_batch_id_list = [
            0,
            0,
            1,
            1,
            2,
            2,
            3,
            3,
            4,
            4,
            5,
            5,
            6,
            6,
            7,
            7,
            0,
        ]
        self.assertEqual(job_type_list, expect_job_type_list)
        self.assertEqual(micro_batch_id_list, expect_micro_batch_id_list)


if __name__ == '__main__':
    unittest.main()

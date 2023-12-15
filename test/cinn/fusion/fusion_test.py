# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

import logging
import os
from test.cinn.passes.pass_test import PassTest

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO').upper())
logger = logging.getLogger(name="pass_test")


class FusionTest(PassTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_input_data(self):
        """Set feed data"""
        self.feed_data = {}
        logger.warn("No Input Data")

    def build_program(self, builder, target):
        """ """
        raise Exception("Not implemented.")

    def check_fusion_outputs(
        self,
        group_size,
        max_relative_error=1e-5,
        all_equal=False,
        equal_nan=False,
    ):
        base_passes = ["AutoCast", "Decomposer", "TransToCustomCallPass"]
        fusion_passes = ["OpFusionPass", "FusionMergePass"]

        real_group_size = self.get_pass_size(base_passes + fusion_passes)
        logger.debug(f"The model has been fused into {real_group_size} groups")
        self.assertEqual(
            real_group_size,
            group_size,
            msg="The model should be fused into {} groups, but actually fused {} groups".format(
                group_size, real_group_size
            ),
        )

        cinn_no_fusion_outputs = self.get_pass_outputs(base_passes)
        cinn_fusion_outputs = self.get_pass_outputs(base_passes + fusion_passes)

        logger.debug("============ Check Outputs ============")
        self.check_results(
            cinn_no_fusion_outputs,
            cinn_fusion_outputs,
            max_relative_error,
            all_equal,
            equal_nan,
        )

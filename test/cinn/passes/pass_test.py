# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
from test.cinn.ops.op_test import OpTest

from cinn.frontend import NetBuilder, Variable

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO').upper())
logger = logging.getLogger(name="pass_test")


class PassTest(OpTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_input_data()

    def init_input_data(self) -> dict:
        """Set feed data"""
        self.feed_data = {}
        logger.warn("No Input Data")

    def build_program(self, builder, target):
        """ """
        raise Exception("Not implemented.")

    def run_program(self):
        net_builder = NetBuilder("pass_test_netbuilder")

        inputs, outputs = self.build_program(net_builder, self.target)

        self.assertIsNotNone(
            outputs, msg="The program's output should not empty!"
        )
        self.assertGreater(
            len(outputs), 0, msg="The program's output should not empty!"
        )
        self.assertIsInstance(
            outputs[0],
            Variable,
            msg="The program's output should be list(cinn.frontend.Variable)",
        )

        pass_prog = net_builder.build()
        return pass_prog, inputs, outputs

    def get_pass_outputs(self, passes):
        pass_prog, inputs, outputs = self.run_program()

        feed_list = []
        for var in inputs:
            self.assertIn(
                var.name(),
                self.feed_data,
                msg=f"Cannot found input data {var.name()} in self.feed_data",
            )
            feed_list.append(self.feed_data[var.name()])

        return self.get_cinn_output(
            pass_prog, self.target, inputs, feed_list, outputs, passes
        )

    def get_pass_size(self, passes):
        pass_prog, _, outputs = self.run_program()
        fetch_ids = {str(out) for out in outputs}
        logger.debug(f"Before pass {passes}:\n{str(pass_prog)}")
        op_num = pass_prog.apply_pass(fetch_ids, self.target, passes)
        logger.debug(f"After pass {passes}:\n{str(pass_prog)}")
        return op_num

    def check_pass_outputs(
        self,
        pass_diff,
        test_passes,
        base_passes=["AutoCast", "Decomposer", "TransToCustomCallPass"],
        max_relative_error=1e-5,
        all_equal=False,
        equal_nan=False,
    ):
        base_pass_size = self.get_pass_size(base_passes)
        logger.debug(f"Pass after base pass optimize has {base_pass_size} ops")
        test_pass_size = self.get_pass_size(base_passes + test_passes)
        logger.debug(
            f"Pass after base and test pass optimize has {test_pass_size} ops"
        )
        self.assertEqual(
            base_pass_size - test_pass_size,
            pass_diff,
            "The pass not running as expected",
        )

        cinn_no_pass_outputs = self.get_pass_outputs(base_passes)
        cinn_pass_outputs = self.get_pass_outputs(base_passes + test_passes)

        logger.debug("============ Check Outputs ============")
        self.check_results(
            cinn_no_pass_outputs,
            cinn_pass_outputs,
            max_relative_error,
            all_equal,
            equal_nan,
        )

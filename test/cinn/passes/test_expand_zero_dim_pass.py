#!/usr/bin/env python3

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

import unittest

import numpy as np
from pass_test import PassTest


class TestExpandZeroDimPass(PassTest):
    def init_input_data(self):
        self.feed_data = {'x': np.random.randint(-10, 10, []).astype("float32")}

    def build_program(self, builder, target):
        x = builder.create_input(
            self.nptype2cinntype(self.feed_data['x'].dtype),
            self.feed_data['x'].shape,
            "x",
        )
        out = builder.exp(x)
        return [x], [out]

    def test_check_results(self):
        # ExpandZeroDim is the first pass in CINN, so that set base_passes = empty
        self.check_pass_outputs(
            pass_diff=0, test_passes=["ExpandZeroDim"], base_passes=[]
        )


if __name__ == "__main__":
    unittest.main()

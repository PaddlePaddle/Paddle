# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import unittest
<<<<<<< HEAD
from paddle.static.amp.fp16_lists import (
    AutoMixedPrecisionLists,
)


class TestAMPList(unittest.TestCase):
=======
from paddle.fluid.contrib.mixed_precision.fp16_lists import AutoMixedPrecisionLists


class TestAMPList(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_main(self):
        custom_white_list = [
            'lookup_table',
            'lookup_table_v2',
        ]
        amp_list = AutoMixedPrecisionLists(custom_white_list=custom_white_list)
        for op in custom_white_list:
            self.assertTrue(op in amp_list.white_list)
            self.assertTrue(op not in amp_list.black_list)
            self.assertTrue(op not in amp_list.unsupported_list)

        default_black_list = [
<<<<<<< HEAD
            'linear_interp_v2',
            'nearest_interp_v2',
            'bilinear_interp_v2',
            'bicubic_interp_v2',
            'trilinear_interp_v2',
=======
            'linear_interp_v2', 'nearest_interp_v2', 'bilinear_interp_v2',
            'bicubic_interp_v2', 'trilinear_interp_v2'
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ]
        for op in default_black_list:
            self.assertTrue(op in amp_list.black_list)


if __name__ == "__main__":
    unittest.main()

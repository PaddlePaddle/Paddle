#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
from paddle.fluid.framework import Program, program_guard


class TestMetricsDetectionMap(unittest.TestCase):
    def test_detection_map(self):
        program = fluid.Program()
        with program_guard(program):
            detect_res = fluid.layers.data(
                name='detect_res',
                shape=[10, 6],
                append_batch_size=False,
                dtype='float32')
            label = fluid.layers.data(
                name='label',
                shape=[10, 1],
                append_batch_size=False,
                dtype='float32')
            box = fluid.layers.data(
                name='bbox',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            map_eval = fluid.metrics.DetectionMAP(
                detect_res, label, box, class_num=21)
            cur_map, accm_map = map_eval.get_map_var()
            self.assertIsNotNone(cur_map)
            self.assertIsNotNone(accm_map)
        print(str(program))


if __name__ == '__main__':
    unittest.main()

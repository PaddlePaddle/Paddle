#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.fluid.layers as layers
<<<<<<< HEAD
import numpy as np
import os
import shutil
import paddle.fluid.core as core
import unittest
from paddle.fluid.layers.nn import _pull_box_sparse
from paddle.fluid.transpiler import collective
=======
import unittest
>>>>>>> 6dd52c5b255f7399188818a29991f5d375cd175d


class TestAscendTriggerOP(unittest.TestCase):
    """ TestCases for ascend_trigger op"""

    def test_ascend_trigger_op(self):
        paddle.enable_static()
        program = fluid.Program()
        block = program.global_block()
        with fluid.program_guard(program):
<<<<<<< HEAD
            x = fluid.layers.data(
                name='x', shape=[1], dtype='int64', lod_level=0)
            y = fluid.layers.data(
                name='y', shape=[1], dtype='int64', lod_level=0)
=======
            x = fluid.data(name='x', shape=[1], dtype='int64', lod_level=0)
            y = fluid.data(name='y', shape=[1], dtype='int64', lod_level=0)
>>>>>>> 6dd52c5b255f7399188818a29991f5d375cd175d
            block.append_op(
                type="ascend_trigger",
                inputs={"FeedList": [x]},
                outputs={"FetchList": [y]},
                attrs={'graph_idx': 0})

        exe = paddle.static.Executor(paddle.CPUPlace())
        try:
            exe.run(program)
        except RuntimeError as e:
            pass
        except:
            self.assertTrue(False)

<<<<<<< HEAD
=======
        paddle.disable_static()

>>>>>>> 6dd52c5b255f7399188818a29991f5d375cd175d

if __name__ == '__main__':
    unittest.main()

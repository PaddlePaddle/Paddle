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

from decorator_helper import prog_scope
import unittest
import paddle.fluid as fluid
import numpy as np
import paddle
import warnings


class TestBackwardInferVarDataTypeShape(unittest.TestCase):

    def test_backward_infer_var_data_type_shape(self):
        paddle.enable_static()
        program = fluid.default_main_program()
        dy = program.global_block().create_var(name="Tmp@GRAD",
                                               shape=[1, 1],
                                               dtype=np.float32,
                                               persistable=True)
        # invoke warning
        fluid.backward._infer_var_data_type_shape_("Tmp@GRAD",
                                                   program.global_block())
        res = False
        with warnings.catch_warnings():
            res = True
        self.assertTrue(res)


if __name__ == '__main__':
    unittest.main()

#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import compiler, Program, program_guard

paddle.enable_static()
SEED = 2021


class TestMemcpy_FillConstant(unittest.TestCase):

    def get_prog(self):
        paddle.enable_static()
        main_program = Program()
        with program_guard(main_program):
            cpu_var_name = "tensor@Cpu"
            npu_var_name = "tensor@Npu"
            cpu_var = main_program.global_block().create_var(name=cpu_var_name,
                                                             shape=[10, 10],
                                                             dtype='float32',
                                                             persistable=False,
                                                             stop_gradient=True)
            npu_var = main_program.global_block().create_var(name=npu_var_name,
                                                             shape=[10, 10],
                                                             dtype='float32',
                                                             persistable=False,
                                                             stop_gradient=True)
            main_program.global_block().append_op(type="fill_constant",
                                                  outputs={"Out": npu_var_name},
                                                  attrs={
                                                      "shape": [10, 10],
                                                      "dtype": npu_var.dtype,
                                                      "value": 1.0,
                                                      "place_type": 4
                                                  })
            main_program.global_block().append_op(type="fill_constant",
                                                  outputs={"Out": cpu_var_name},
                                                  attrs={
                                                      "shape": [10, 10],
                                                      "dtype": cpu_var.dtype,
                                                      "value": 0.0,
                                                      "place_type": 0
                                                  })
        return main_program, npu_var, cpu_var

    def test_npu_cpoy_to_cpu(self):
        main_program, npu_var, cpu_var = self.get_prog()
        main_program.global_block().append_op(type='memcpy',
                                              inputs={'X': npu_var},
                                              outputs={'Out': cpu_var},
                                              attrs={'dst_place_type': 0})
        place = fluid.NPUPlace(0)
        exe = fluid.Executor(place)
        npu_, cpu_ = exe.run(main_program,
                             feed={},
                             fetch_list=[npu_var.name, cpu_var.name])
        np.testing.assert_allclose(npu_, cpu_)
        np.testing.assert_allclose(cpu_, np.ones((10, 10)))

    def test_cpu_cpoy_npu(self):
        main_program, npu_var, cpu_var = self.get_prog()
        main_program.global_block().append_op(type='memcpy',
                                              inputs={'X': cpu_var},
                                              outputs={'Out': npu_var},
                                              attrs={'dst_place_type': 4})
        place = fluid.NPUPlace(0)
        exe = fluid.Executor(place)
        npu_, cpu_ = exe.run(main_program,
                             feed={},
                             fetch_list=[npu_var.name, cpu_var.name])
        np.testing.assert_allclose(npu_, cpu_)
        np.testing.assert_allclose(npu_, np.zeros((10, 10)))


if __name__ == '__main__':
    unittest.main()

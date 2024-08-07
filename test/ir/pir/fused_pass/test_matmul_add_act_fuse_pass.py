# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.base import core

paddle.enable_static()


class TestMatmulAddPattern(PassTest):
    r'''
    Matmul     Y
       \     /
         Add
    both FcOp and GemmEpilogueOp
    both 1D-elementwiseAdd and 2D-elementwiseAdd.
    '''

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        '''
        matmul_add_act_fuse_pass's unittest have been tested locally, and the pass works well on LLMs.
        However, since the GemmEpilogueOp relies on users manually installing libCutlassGemmEpilogue.so,
        this test code has been temporarily shut down(i.e. use_cutlass = False).
        You can easily run this unittest by manually installing libCutlassGemmEpilogue.so and set use_cutlass True.
        (See details: paddle/phi/kernels/fusion/cutlass/gemm_epilogue/README.md)
        '''
        use_cutlass = False  # option
        if use_cutlass:
            fused_op_name = "pd_op.gemm_epilogue"
        else:
            fused_op_name = "pd_op.fc"

        for M in [32, 128]:
            for N in [32, 128]:
                for K in [128]:
                    x_shape = [M, K]
                    w_shape = [K, N]
                    for y_shape in [[N], [M, N]]:
                        if fused_op_name == "pd_op.fc" and len(y_shape) == 2:
                            continue
                        with paddle.pir_utils.IrGuard():
                            start_prog = paddle.static.Program()
                            main_prog = paddle.static.Program()
                            with paddle.pir.core.program_guard(
                                main_prog, start_prog
                            ):
                                x = paddle.static.data(
                                    name='x', shape=x_shape, dtype='float16'
                                )
                                w = paddle.static.data(
                                    name='w', shape=w_shape, dtype='float16'
                                )
                                y = paddle.static.data(
                                    name='y', shape=y_shape, dtype='float16'
                                )
                                out = paddle.add(paddle.matmul(x, w), y)
                                out = paddle.assign(out)
                                self.pass_attr_list = [
                                    {
                                        'matmul_add_act_fuse_pass': {
                                            "use_cutlass": use_cutlass
                                        }
                                    }
                                ]
                                self.feeds = {
                                    "x": np.random.random(x_shape).astype(
                                        "float16"
                                    ),
                                    "w": np.random.random(w_shape).astype(
                                        "float16"
                                    ),
                                    "y": np.random.random(y_shape).astype(
                                        "float16"
                                    ),
                                }
                                self.fetch_list = [out]
                                self.valid_op_map = {
                                    "pd_op.add": 0,
                                    "pd_op.matmul": 0,
                                    fused_op_name: 1,
                                }

                                yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=2e-3, rtol=2e-3)


r"""
'''
matmul_add_act_fuse_pass's unittest have been tested locally, and the pass works well on LLMs.
However, since the GemmEpilogueOp relies on users manually installing libCutlassGemmEpilogue.so,
this test code has been temporarily shut down.
You can easily run this unittest by manually installing libCutlassGemmEpilogue.so and set use_cutlass True.
(See details: paddle/phi/kernels/fusion/cutlass/gemm_epilogue/README.md)
'''
class TestMatmulAddPatternReverseAdd(PassTest):
    r'''
      Y     Matmul
       \     /
         Add
    only GemmEpilogueOp
    both 1D-elementwiseAdd and 2D-elementwiseAdd.
    '''

    def is_program_valid(self, program=None):
        return False

    def sample_program(self):
        for x_shape in [[48, 13696], [2, 13696], [34, 13696]]:
            M = x_shape[0]
            for w_shape in [[13696, 4096]]:
                N = w_shape[1]
                for y_shape in [[N], [M, N]]:
                    with paddle.pir_utils.IrGuard():
                        start_prog = paddle.static.Program()
                        main_prog = paddle.static.Program()
                        with paddle.pir.core.program_guard(
                            main_prog, start_prog
                        ):
                            x = paddle.static.data(
                                name='x', shape=x_shape, dtype='float16'
                            )
                            w = paddle.static.data(
                                name='w', shape=w_shape, dtype='float16'
                            )
                            y = paddle.static.data(
                                name='y', shape=y_shape, dtype='float16'
                            )
                            out = paddle.add(paddle.matmul(x, w), y)
                            out = paddle.assign(out)
                            self.pass_attr_list = [
                                {
                                    'matmul_add_act_fuse_pass': {
                                        "use_cutlass": True
                                    }
                                }
                            ]
                            self.feeds = {
                                "x": np.random.random(x_shape).astype(
                                    "float16"
                                ),
                                "w": np.random.random(w_shape).astype(
                                    "float16"
                                ),
                                "y": np.random.random(y_shape).astype(
                                    "float16"
                                ),
                            }
                            self.fetch_list = [out]
                            self.valid_op_map = {
                                "pd_op.add": 0,
                                "pd_op.relu": 0,
                                "pd_op.matmul": 0,
                                "pd_op.gemm_epilogue": 1,
                            }

                            yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=2e-3, rtol=2e-3)
"""


class TestMatmulAddActPattern(PassTest):
    r'''
    Matmul     Y
       \     /
         Add
          |
         Act
    both FcOp and GemmEpilogueOp
    both 1D-elementwiseAdd and 2D-elementwiseAdd.
    '''

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        '''
        matmul_add_act_fuse_pass's unittest have been tested locally, and the pass works well on LLMs.
        However, since the GemmEpilogueOp relies on users manually installing libCutlassGemmEpilogue.so,
        this test code has been temporarily shut down(i.e. use_cutlass = False).
        You can easily run this unittest by manually installing libCutlassGemmEpilogue.so and set use_cutlass True.
        (See details: paddle/phi/kernels/fusion/cutlass/gemm_epilogue/README.md)
        '''
        use_cutlass = False  # option
        if use_cutlass:
            fused_op_name = "pd_op.gemm_epilogue"
            acts = ["pd_op.relu", "pd_op.gelu"]
            acts_map = {
                "pd_op.relu": paddle.nn.functional.relu,
                "pd_op.gelu": paddle.nn.functional.gelu,
            }
        else:
            fused_op_name = "pd_op.fc"
            acts = ["pd_op.relu"]
            acts_map = {"pd_op.relu": paddle.nn.functional.relu}

        for act in acts:
            for x_shape in [[2, 128]]:
                M = x_shape[0]
                for w_shape in [[128, 128], [128, 64]]:
                    N = w_shape[1]
                    for y_shape in [[N], [M, N]]:
                        if fused_op_name == "pd_op.fc" and len(y_shape) == 2:
                            continue
                        with paddle.pir_utils.IrGuard():
                            start_prog = paddle.static.Program()
                            main_prog = paddle.static.Program()
                            with paddle.pir.core.program_guard(
                                main_prog, start_prog
                            ):
                                x = paddle.static.data(
                                    name='x', shape=x_shape, dtype='float16'
                                )
                                w = paddle.static.data(
                                    name='w', shape=w_shape, dtype='float16'
                                )
                                y = paddle.static.data(
                                    name='y', shape=y_shape, dtype='float16'
                                )

                                matmul_add_out = paddle.add(
                                    paddle.matmul(x, w), y
                                )
                                out = acts_map[act](matmul_add_out)
                                out = paddle.assign(out)

                                self.pass_attr_list = [
                                    {
                                        'matmul_add_act_fuse_pass': {
                                            "use_cutlass": use_cutlass
                                        }
                                    }
                                ]
                                self.feeds = {
                                    "x": np.random.random(x_shape).astype(
                                        "float16"
                                    ),
                                    "w": np.random.random(w_shape).astype(
                                        "float16"
                                    ),
                                    "y": np.random.random(y_shape).astype(
                                        "float16"
                                    ),
                                }
                                self.fetch_list = [out]
                                self.valid_op_map = {
                                    "pd_op.matmul": 0,
                                    "pd_op.add": 0,
                                    act: 0,
                                    fused_op_name: 1,
                                }

                                yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=2e-3, rtol=2e-3)


r"""
'''
matmul_add_act_fuse_pass's unittest have been tested locally, and the pass works well on LLMs.
However, since the GemmEpilogueOp relies on users manually installing libCutlassGemmEpilogue.so,
this test code has been temporarily shut down.
You can easily run this unittest by manually installing libCutlassGemmEpilogue.so and set use_cutlass True.
(See details: paddle/phi/kernels/fusion/cutlass/gemm_epilogue/README.md)
'''
class TestMatmulAddActPatternReverseAdd(PassTest):
    r'''
      Y     Matmul
       \     /
         Add
          |
         Act
    onlyGemmEpilogueOp
    both 1D-elementwiseAdd and 2D-elementwiseAdd.
    '''

    def is_program_valid(self, program=None):
        return False

    def sample_program(self):
        acts = ["pd_op.relu", "pd_op.gelu"]
        acts_map = {
            "pd_op.relu": paddle.nn.functional.relu,
            "pd_op.gelu": paddle.nn.functional.gelu,
        }
        for act in acts:
            for M in [32, 512, 4096]:
                for N in [32, 512, 4096]:
                    for K in [32, 512, 4096]:
                        x_shape = [M, K]
                        w_shape = [K, N]
                        for y_shape in [[N], [M, N]]:
                            with paddle.pir_utils.IrGuard():
                                start_prog = paddle.static.Program()
                                main_prog = paddle.static.Program()
                                with paddle.pir.core.program_guard(
                                    main_prog, start_prog
                                ):
                                    x = paddle.static.data(
                                        name='x', shape=x_shape, dtype='float16'
                                    )
                                    w = paddle.static.data(
                                        name='w', shape=w_shape, dtype='float16'
                                    )
                                    y = paddle.static.data(
                                        name='y', shape=y_shape, dtype='float16'
                                    )

                                    matmul_add_out = paddle.add(
                                        y, paddle.matmul(x, w)
                                    )
                                    out = acts_map[act](matmul_add_out)
                                    out = paddle.assign(out)

                                    self.pass_attr_list = [
                                        {
                                            'matmul_add_act_fuse_pass': {
                                                "use_cutlass": True
                                            }
                                        }
                                    ]
                                    self.feeds = {
                                        "x": np.random.random(x_shape).astype(
                                            "float16"
                                        ),
                                        "w": np.random.random(w_shape).astype(
                                            "float16"
                                        ),
                                        "y": np.random.random(y_shape).astype(
                                            "float16"
                                        ),
                                    }
                                    self.fetch_list = [out]
                                    self.valid_op_map = {
                                        "pd_op.matmul": 0,
                                        "pd_op.add": 0,
                                        act: 0,
                                        "pd_op.gemm_epilogue": 1,
                                    }

                                    yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=2e-3, rtol=2e-3)
"""

if __name__ == "__main__":
    unittest.main()

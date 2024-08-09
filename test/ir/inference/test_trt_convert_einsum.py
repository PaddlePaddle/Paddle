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

from __future__ import annotations

import unittest
from functools import partial

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertEinsumTest_SingleOperand(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8200:
            return False
        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_input1(dims, batch):
            if dims == 1:
                return np.ones(shape=[batch]).astype(np.float32)
            elif dims == 3:
                return np.ones((batch, 2, 3)).astype(np.float32)

        def generate_equation1(dims):
            if dims == 1:
                return ["i->"]
            elif dims == 3:
                # "ijk->","ijk->j","ijk->k"
                # error: The current implementation of Einsum doesn't support mask dimensions on multiple contracting/free dimensions
                return [
                    "ijk->ikj",
                    "ijk->i",
                    "ijk->ij",
                    "ijk->ik",
                    "ijk->ijk",
                    "ijk->jk",
                ]

        # Single operand: transpose, sum
        for dims in [1, 3]:
            for batch in [2]:
                equation_list = generate_equation1(dims)
                for equation in equation_list:
                    self.equation = equation
                    self.dims = dims
                    dics = [
                        {
                            "equation": equation,
                        }
                    ]
                    ops_config = [
                        {
                            "op_type": "einsum",
                            "op_inputs": {"Operands": ["operands_data0"]},
                            "op_outputs": {"Out": ["einsum_output_data"]},
                            "op_attrs": dics[0],
                        }
                    ]
                    ops = self.generate_op_config(ops_config)

                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "operands_data0": TensorConfig(
                                data_gen=partial(generate_input1, dims, batch)
                            )
                        },
                        outputs=["einsum_output_data"],
                    )

                    yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {
                    "operands_data0": [1],
                }
                self.dynamic_shape.max_input_shape = {
                    "operands_data0": [3],
                }
                self.dynamic_shape.opt_input_shape = {
                    "operands_data0": [2],
                }

            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "operands_data0": [1, 2, 3],
                }
                self.dynamic_shape.max_input_shape = {
                    "operands_data0": [4, 2, 3],
                }
                self.dynamic_shape.opt_input_shape = {
                    "operands_data0": [2, 2, 3],
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if (not dynamic_shape) or ("..." in self.equation):
                return 0, 3
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5

    def test(self):
        self.run_test()


class TrtConvertEinsumTest_DoubuleOperand_Vector_Matrix(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8200:
            return False
        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_input_matrix(dims, batch):
            if dims == 1:
                return np.ones(shape=[batch]).astype(np.float32)
            elif dims == 2:
                return np.ones(shape=[batch, 3]).astype(np.float32)
            elif dims == 3:
                return np.ones((batch, 2, 3)).astype(np.float32)

        """
        genertate_vector
        """

        def generate_input_vector(vec_shape):
            return np.ones(vec_shape).astype(np.float32)

        def generate_equation_matrix_vector(dims, vec_shape):
            if dims == 1:
                return ["i,i->", "i,i->i", "i,j->ij"]
            elif dims == 2 and vec_shape == [3]:
                return ["ij,j->i", "ij,j->j", "ij,j->ij", "ij,j", "ij,j->"]
            elif dims == 3 and vec_shape == [3]:
                return [
                    "ijk,k->i",
                    "ijk,k->j",
                    "ijk,k->k",
                    "ijk,k->ij",
                    "ijk,k->ik",
                    "ijk,k->jk",
                    "ijk,k->ijk",
                    "ijk,k",
                    "ijk,k->",
                ]

        # Doubule operands vector
        for dims in [1]:
            self.dims = dims
            for vec_shape in [[2]]:
                for batch in [2]:
                    equation_list = generate_equation_matrix_vector(
                        dims, vec_shape
                    )
                    for equation in equation_list:
                        self.equation = equation
                        self.dims = dims
                        dics = [{"equation": equation}, {}]
                        ops_config = [
                            {
                                "op_type": "einsum",
                                "op_inputs": {
                                    "Operands": [
                                        "operands_data0",
                                        "operands_data1",
                                    ]
                                },
                                "op_outputs": {"Out": ["einsum_output_data"]},
                                "op_attrs": dics[0],
                            }
                        ]
                        ops = self.generate_op_config(ops_config)

                        program_config = ProgramConfig(
                            ops=ops,
                            weights={},
                            inputs={
                                "operands_data0": TensorConfig(
                                    data_gen=partial(
                                        generate_input_matrix, dims, batch
                                    )
                                ),
                                "operands_data1": TensorConfig(
                                    data_gen=partial(
                                        generate_input_vector, vec_shape
                                    )
                                ),
                            },
                            outputs=["einsum_output_data"],
                        )

                        yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {
                    "operands_data0": [1],
                    "operands_data1": [1],
                }
                self.dynamic_shape.max_input_shape = {
                    "operands_data0": [4],
                    "operands_data1": [4],
                }
                self.dynamic_shape.opt_input_shape = {
                    "operands_data0": [2],
                    "operands_data1": [2],
                }
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "operands_data0": [1, 2, 3],
                    "operands_data1": [1],
                }
                self.dynamic_shape.max_input_shape = {
                    "operands_data0": [4, 2, 3],
                    "operands_data1": [4],
                }
                self.dynamic_shape.opt_input_shape = {
                    "operands_data0": [2, 2, 3],
                    "operands_data1": [3],
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if (not dynamic_shape) or ("..." in self.equation):
                return 0, 4
            return 1, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5

    def test(self):
        self.run_test()


class TrtConvertEinsumTest_DoubuleOperand_Matrix_Matrix(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8200:
            return False
        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_input_matrix(input_shape):
            return np.ones(shape=input_shape).astype(np.float32)

        # Doubule operands vector
        for item in [
            [[4, 5], [4, 5], "ij,ij->ij"],  # MatrixEleMul
            [[4, 5], [2, 5], "ij,kj->ik"],  # MatrixMul
            [[4, 5], [3, 7], "ij,kl->ijkl"],  # MatrixOuter
            [[3, 4, 5], [3, 5, 2], "bij,bjk->bik"],
            [[3, 4, 5], [4, 5], "ijk,jk->i"],
            [[3, 4, 5], [2, 5], "ijk,lk->ijl"],
            [[2, 4, 5, 3], [3, 4, 5], "ijkl,lmn->ijkmn"],
        ]:
            self.x_shape = item[0]
            self.y_shape = item[1]
            equation = item[2]
            self.equation = equation

            dics = [{"equation": equation}, {}]
            ops_config = [
                {
                    "op_type": "einsum",
                    "op_inputs": {
                        "Operands": ["operands_data0", "operands_data1"]
                    },
                    "op_outputs": {"Out": ["einsum_output_data"]},
                    "op_attrs": dics[0],
                }
            ]
            ops = self.generate_op_config(ops_config)

            program_config = ProgramConfig(
                ops=ops,
                weights={},
                inputs={
                    "operands_data0": TensorConfig(
                        data_gen=partial(generate_input_matrix, self.x_shape)
                    ),
                    "operands_data1": TensorConfig(
                        data_gen=partial(generate_input_matrix, self.y_shape)
                    ),
                },
                outputs=["einsum_output_data"],
            )

            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            min_xshape = self.x_shape[:]
            max_xshape = self.x_shape[:]
            min_yshape = self.y_shape[:]
            max_yshape = self.y_shape[:]
            if "b" in self.equation:
                min_xshape[0] = 1
                max_xshape[0] = 4
                min_yshape[0] = 1
                max_yshape[0] = 4
            self.dynamic_shape.min_input_shape = {
                "operands_data0": min_xshape,
                "operands_data1": min_yshape,
            }
            self.dynamic_shape.max_input_shape = {
                "operands_data0": max_xshape,
                "operands_data1": max_yshape,
            }
            self.dynamic_shape.opt_input_shape = {
                "operands_data0": self.x_shape,
                "operands_data1": self.y_shape,
            }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if (not dynamic_shape) or ("..." in self.equation):
                return 0, 4
            return 1, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()

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

from __future__ import annotations

import os
import unittest
from functools import partial

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import SkipReasons, TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertGatherNdTest_dim_4_1(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        # The output has diff between gpu and trt in CI windows
        # if ( and self.trt_param.precision == paddle_infer.PrecisionType.Half):
        #     return False
        return True

    def sample_program_configs(self):
        def generate_input1():
            return np.random.random([2, 32, 64, 64]).astype(np.float32)

        def generate_input2():
            return np.ones([1]).astype(np.int32)

        ops_config = [
            {
                "op_type": "gather_nd",
                "op_inputs": {"X": ["input_data"], "Index": ["index_data"]},
                "op_outputs": {"Out": ["output_data"]},
                "op_attrs": {},
            }
        ]
        ops = self.generate_op_config(ops_config)
        for i in range(10):
            program_config = ProgramConfig(
                ops=ops,
                weights={},
                inputs={
                    "input_data": TensorConfig(
                        data_gen=partial(generate_input1)
                    ),
                    "index_data": TensorConfig(
                        data_gen=partial(generate_input2)
                    ),
                },
                outputs=["output_data"],
                no_cast_list=["index_data"],
            )

            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [2, 32, 64, 64],
                "index_data": [1],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [2, 32, 64, 64],
                "index_data": [1],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [2, 32, 64, 64],
                "index_data": [1],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), (0, 4), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), (0, 4), 1e-3

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), (1, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), (1, 3), 1e-3

    def add_skip_trt_case(self):
        def teller1(program_config, predictor_config):
            if len(self.dynamic_shape.min_input_shape) != 0 and os.name == 'nt':
                return True
            return False

        self.add_skip_case(
            teller1,
            SkipReasons.TRT_NOT_SUPPORT,
            "Under Windows Ci, this case will sporadically fail.",
        )

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


class TrtConvertGatherNdTest_dim_4_1_2(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1():
            return np.random.random([2, 32, 64, 64]).astype(np.float32)

        def generate_input2():
            return np.array([1, 2]).astype(np.int32)

        ops_config = [
            {
                "op_type": "gather_nd",
                "op_inputs": {"X": ["input_data"], "Index": ["index_data"]},
                "op_outputs": {"Out": ["output_data"]},
                "op_attrs": {},
            }
        ]
        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input1)),
                "index_data": TensorConfig(data_gen=partial(generate_input2)),
            },
            outputs=["output_data"],
            no_cast_list=["index_data"],
        )

        yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [2, 32, 64, 64],
                "index_data": [2],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [2, 32, 64, 64],
                "index_data": [2],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [2, 32, 64, 64],
                "index_data": [2],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), (0, 4), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), (0, 4), 1e-3

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), (1, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), (1, 3), 1e-3

    def add_skip_trt_case(self):
        def teller1(program_config, predictor_config):
            if len(self.dynamic_shape.min_input_shape) != 0 and os.name == 'nt':
                return True
            return False

        self.add_skip_case(
            teller1,
            SkipReasons.TRT_NOT_SUPPORT,
            "Under Windows Ci, this case will sporadically fail.",
        )

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


class TrtConvertGatherNdTest_dim_4_2(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1():
            return np.random.random([2, 32, 64, 64]).astype(np.float32)

        def generate_input2():
            return np.ones([2, 2]).astype(np.int32)

        ops_config = [
            {
                "op_type": "gather_nd",
                "op_inputs": {"X": ["input_data"], "Index": ["index_data"]},
                "op_outputs": {"Out": ["output_data"]},
                "op_attrs": {},
            }
        ]
        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input1)),
                "index_data": TensorConfig(data_gen=partial(generate_input2)),
            },
            outputs=["output_data"],
            no_cast_list=["index_data"],
        )

        yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [2, 32, 64, 64],
                "index_data": [2, 2],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [2, 32, 64, 64],
                "index_data": [2, 2],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [2, 32, 64, 64],
                "index_data": [2, 2],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), (0, 4), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), (0, 4), 1e-3

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), (1, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), (1, 3), 1e-3

    def add_skip_trt_case(self):
        def teller1(program_config, predictor_config):
            if len(self.dynamic_shape.min_input_shape) != 0 and os.name == 'nt':
                return True
            return False

        self.add_skip_case(
            teller1,
            SkipReasons.TRT_NOT_SUPPORT,
            "Under Windows Ci, this case will sporadically fail.",
        )

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


class TrtConvertGatherNdTest_dim_4_3(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1():
            return np.random.random([2, 32, 64, 64]).astype(np.float32)

        def generate_input2():
            return np.ones([2, 2, 4]).astype(np.int32)

        ops_config = [
            {
                "op_type": "gather_nd",
                "op_inputs": {"X": ["input_data"], "Index": ["index_data"]},
                "op_outputs": {"Out": ["output_data"]},
                "op_attrs": {},
            }
        ]
        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input1)),
                "index_data": TensorConfig(data_gen=partial(generate_input2)),
            },
            outputs=["output_data"],
            no_cast_list=["index_data"],
        )

        yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [2, 32, 64, 64],
                "index_data": [2, 2, 4],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [2, 32, 64, 64],
                "index_data": [2, 2, 4],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [2, 32, 64, 64],
                "index_data": [2, 2, 4],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), (0, 4), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), (0, 4), 1e-3

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), (1, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), (1, 3), 1e-3

    def add_skip_trt_case(self):
        def teller1(program_config, predictor_config):
            if len(self.dynamic_shape.min_input_shape) != 0 and os.name == 'nt':
                return True
            return False

        self.add_skip_case(
            teller1,
            SkipReasons.TRT_NOT_SUPPORT,
            "Under Windows Ci, this case will sporadically fail.",
        )

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


class TrtConvertGatherNdTest_dim_2_2(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1():
            return np.random.random([2, 32]).astype(np.float32)

        def generate_input2():
            return np.array([[0, 3], [1, 9]]).astype(np.int32)

        ops_config = [
            {
                "op_type": "gather_nd",
                "op_inputs": {"X": ["input_data"], "Index": ["index_data"]},
                "op_outputs": {"Out": ["output_data"]},
                "op_attrs": {},
            }
        ]
        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input1)),
                "index_data": TensorConfig(data_gen=partial(generate_input2)),
            },
            outputs=["output_data"],
            no_cast_list=["index_data"],
        )

        yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [2, 32],
                "index_data": [2, 2],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [2, 32],
                "index_data": [2, 2],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [2, 32],
                "index_data": [2, 2],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), (0, 4), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), (0, 4), 1e-3

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), (1, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), (1, 3), 1e-3

    def add_skip_trt_case(self):
        def teller1(program_config, predictor_config):
            if len(self.dynamic_shape.min_input_shape) != 0 and os.name == 'nt':
                return True
            return False

        self.add_skip_case(
            teller1,
            SkipReasons.TRT_NOT_SUPPORT,
            "Under Windows Ci, this case will sporadically fail.",
        )

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


class TrtConvertGatherNdTest_dim_3_3(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1():
            return np.random.random([16, 32, 256]).astype(np.float32)

        def generate_input2():
            return np.array([[[2, 5], [3, 8]], [[0, 2], [0, 3]]]).astype(
                np.int32
            )

        ops_config = [
            {
                "op_type": "gather_nd",
                "op_inputs": {"X": ["input_data"], "Index": ["index_data"]},
                "op_outputs": {"Out": ["output_data"]},
                "op_attrs": {},
            }
        ]
        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input1)),
                "index_data": TensorConfig(data_gen=partial(generate_input2)),
            },
            outputs=["output_data"],
            no_cast_list=["index_data"],
        )

        yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [16, 32, 256],
                "index_data": [2, 2, 2],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [16, 32, 256],
                "index_data": [2, 2, 2],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [16, 32, 256],
                "index_data": [2, 2, 2],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), (0, 4), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), (0, 4), 1e-3

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), (1, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), (1, 3), 1e-3

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()

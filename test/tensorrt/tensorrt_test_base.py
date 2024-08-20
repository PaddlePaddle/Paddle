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

import copy
import unittest

import numpy as np

import paddle
from paddle.base import core
from paddle.tensorrt.converter import PaddleToTensorRTConverter
from paddle.tensorrt.util import (
    run_pir_pass,
    warmup_shape_infer,
)


class TensorRTBaseTest(unittest.TestCase):
    def setUp(self):
        self.python_api = None
        self.api_args = None
        self.program_config = None
        self.min_shape = None
        self.max_shape = None

    def create_fake_program(self):
        if self.python_api is None:
            raise ValueError(
                "The unittest must specify a python api that will be used for building pir program."
            )
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            api_args = copy.deepcopy(self.api_args)
            for feed_name in self.program_config["feed_list"]:
                input_shape_without_dynamic_dim = self.api_args[
                    feed_name
                ].shape[1:]
                input_dynamic_shape = [-1]
                input_dynamic_shape.extend(input_shape_without_dynamic_dim)
                input_dtype = self.api_args[feed_name].dtype
                input_data = paddle.static.data(
                    name=feed_name, shape=input_dynamic_shape, dtype=input_dtype
                )
                api_args[feed_name] = input_data
            actual_args = []
            for name, value in api_args.items():
                actual_args.append(value)
            output = self.python_api(*actual_args)
            fetch_list = []
            if isinstance(output, tuple):
                fetch_list = list(output)
            else:
                fetch_list.append(output)
        return main_program, startup_program, fetch_list

    def run_program(self, main_program, startup_program, fetch_list):
        place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        exe = paddle.static.Executor(place)
        exe.run(startup_program)
        feed_data = dict()  # noqa: C408
        for feed_name in self.program_config["feed_list"]:
            feed_data[feed_name] = self.api_args[feed_name]
        ret = exe.run(main_program, feed=feed_data, fetch_list=fetch_list)
        return ret

    def check_trt_result(self):
        paddle.framework.set_flags({"FLAGS_trt_min_group_size": 1})
        with paddle.pir_utils.IrGuard():
            main_program, startup_program, fetch_list = (
                self.create_fake_program()
            )
            fetch_num = len(fetch_list)
            output_expected = self.run_program(
                main_program, startup_program, fetch_list
            )
            min_shape_data = dict()  # noqa: C408
            max_shape_data = dict()  # noqa: C408
            for feed_name in self.program_config["feed_list"]:
                min_shape_data[feed_name] = np.random.randn(
                    *self.min_shape[feed_name]
                ).astype(self.api_args[feed_name].dtype)
                max_shape_data[feed_name] = np.random.randn(
                    *self.max_shape[feed_name]
                ).astype(self.api_args[feed_name].dtype)
            warmup_shape_infer(
                main_program,
                min_shape_feed=min_shape_data,
                max_shape_feed=max_shape_data,
            )

            # run pir pass(including some fusion pass and trt_op_marker_pass)
            program = run_pir_pass(main_program, partition_mode=False)

            # run trt_sub_graph_extract_pass()
            program_with_pir = run_pir_pass(program, partition_mode=True)

            # run TRTConverter(would lower group_op into tensorrt_engine_op)
            scope = paddle.static.global_scope()
            converter = PaddleToTensorRTConverter(program_with_pir, scope)
            converter.convert_program_to_trt()

            for op in program_with_pir.global_block().ops[::-1]:
                # Remove all invalid fetch op
                if op.name() == "pd_op.fetch":
                    program_with_pir.global_block().remove_op(op)

            # check whether has trt op
            has_trt_op = False
            for op in program_with_pir.global_block().ops:
                if op.name() == "pd_op.tensorrt_engine":
                    has_trt_op = True
            self.assertEqual(has_trt_op, True)

            trt_fetch_list = program_with_pir.list_vars()[-1 * fetch_num :]
            output_trt = self.run_program(
                program_with_pir, startup_program, trt_fetch_list
            )

        # Check that the results are close to each other within a tolerance of 1e-3
        for i in range(fetch_num):
            np.testing.assert_allclose(
                output_expected[i],
                output_trt[i],
                rtol=1e-3,
                atol=1e-3,
            )

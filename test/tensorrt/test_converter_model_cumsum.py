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
import os
import tempfile
import unittest

import numpy as np
from get_program import (
    get_program,
)

import paddle
import paddle.inference as paddle_infer
from paddle.framework import use_pir_api
from paddle.tensorrt.converter import PaddleToTensorRTConverter
from paddle.tensorrt.util import (
    predict_program,
    run_pir_pass,
    warmup_shape_infer_v2,
)


class TestConverterCumsumOp(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_path = os.path.join(self.temp_dir.name, 'tensor_axis_cumsum')
        self.trt_save_path = os.path.join(self.temp_dir.name, 'trt')
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_paddle_to_tensorrt_conversion_cumsum(self):
        # Step1: get program and init fake inputs
        paddle.enable_static()
        np_x = np.random.randn(9, 10, 11).astype('float32')

        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(
                    shape=np_x.shape, name='x', dtype=np_x.dtype
                )
                linear = paddle.nn.Linear(np_x.shape[-1], np_x.shape[-1])
                linear_out = linear(x)
                relu_out = paddle.nn.functional.relu(linear_out)
                axis = paddle.full([1], 2, dtype='int64')
                out = paddle.cumsum(relu_out, axis=axis)
                loss = paddle.mean(out)
                sgd = paddle.optimizer.SGD(learning_rate=0.0)
                sgd.minimize(paddle.mean(out))

                exe = paddle.static.Executor(self.place)
                exe.run(startup_prog)
                static_out = exe.run(feed={'x': np_x}, fetch_list=[out])

                # run infer
                paddle.static.save_inference_model(
                    self.save_path, [x], [out], exe
                )

                if use_pir_api():
                    config = paddle_infer.Config(
                        self.save_path + '.json', self.save_path + '.pdiparams'
                    )
                    config.enable_new_ir()
                    config.enable_new_executor()
                    config.use_optimized_model(True)
                else:
                    config = paddle_infer.Config(
                        self.save_path + '.pdmodel',
                        self.save_path + '.pdiparams',
                    )

        program, scope, feed_target_names, fetch_targets = get_program(
            self.temp_dir.name, "tensor_axis_cumsum", True
        )

        output_var = []
        feed_name = []
        for op in program.global_block().ops:
            if op.name() == "pd_op.fetch":
                for operand in op.operands():
                    source = operand.source()
                    output_var.append(source)
            if op.name() == "pd_op.data" or op.name() == "pd_op.feed":
                param_name = op.attrs()["name"]
                feed_name.append(param_name)

        with paddle.pir_utils.IrGuard():
            input_data_min_shape = np.random.randn(9, 10, 11).astype('float32')
            input_data_max_shape = np.random.randn(9, 10, 11).astype('float32')

            # Step1.1: get original results(for tests only)
            pred_res = predict_program(
                program,
                {feed_name[0]: input_data_min_shape},
                fetch_var_list=output_var,
            )

            # Step2: run warmup for collecting shape
            warmup_shape_infer_v2(
                program,
                min_shape_feed={feed_name[0]: input_data_min_shape},
                max_shape_feed={feed_name[0]: input_data_max_shape},
                fetch_var_list=output_var,
            )

            # Step3: run pir pass(including some fusion pass and trt_op_marker_pass)
            # program_with_pir = run_pir_pass(program, partition_mode=False)

            program_with_pir = run_pir_pass(program, partition_mode=True)

            # output_var = program_with_pir.list_vars()[-2]

            trt_output_var = []

            for op in program_with_pir.global_block().ops:
                if op.name() == "pd_op.fetch":
                    for operand in op.operands():
                        source = operand.source()
                        trt_output_var.append(source)

            # Step5: run TRTConverter(would lower group_op into tensorrt_engine_op)
            converter = PaddleToTensorRTConverter(program_with_pir, scope)
            converter.convert_program_to_trt()

            # #executor.run
            # output_converted=predict_program(
            #     program_with_pir,
            #     {feed_target_names[0]: input_data_min_shape},
            #     trt_output_var
            # )

            # predictor run
            save_path = "/home/zexuli/Paddle-2/test/tensorrt/inference/tensor_axis_cumsum"
            input_values = []
            input_values.extend(
                result
                for op in program_with_pir.global_block().ops
                if op.name() == "pd_op.data" or op.name() == "pd_op.feed"
                for result in op.results()
            )
            output_var = program_with_pir.list_vars()[-2]
            place = paddle.CUDAPlace(0)
            exe = paddle.static.Executor(place)
            paddle.static.save_inference_model(
                self.trt_save_path,
                input_values,
                trt_output_var,
                exe,
                program=program_with_pir,
            )

            config = paddle_infer.Config(
                self.trt_save_path + '.json', self.trt_save_path + '.pdiparams'
            )

            config.switch_ir_debug()

            if paddle.is_compiled_with_cuda():
                config.enable_use_gpu(100, 0)
            else:
                config.disable_gpu()
            predictor = paddle_infer.create_predictor(config)
            input_names = predictor.get_input_names()

            input_handle = predictor.get_input_handle(input_names[0])
            fake_input = input_data_min_shape
            input_handle.reshape(input_data_min_shape.shape)
            input_handle.copy_from_cpu(fake_input)
            predictor.run()
            output_names = predictor.get_output_names()
            output_handle = predictor.get_output_handle(output_names[0])
            infer_out = output_handle.copy_to_cpu()

            np.testing.assert_allclose(
                pred_res[0],
                infer_out,
                rtol=1e-3,
                atol=1e-3,
                err_msg="Outputs are not within the 0.2 tolerance",
            )


if __name__ == "__main__":
    unittest.main()

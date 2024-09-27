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

import os
import unittest

import numpy as np

os.environ['FLAGS_all_blocks_convert_trt'] = '1'

import paddle
import paddle.inference as paddle_infer


def check_output_allclose(out, pd_out, name, rtol=5e-5, atol=1e-2):
    if out is None and pd_out is None:
        return
    assert out is not None, "out value of " + name + " is None"
    assert pd_out is not None, "pd_out value of " + name + " is None"
    np.testing.assert_allclose(
        out,
        pd_out,
        rtol,
        atol,
        err_msg=f'custom op {name}: {out},\n paddle api {name}: {pd_out}',
    )


paddle.enable_static()


class TestWhileOP(unittest.TestCase):
    def setUp(self):
        def cond(tmp, out_0, step_idx_gpu, max_dec_len):
            return paddle.less_than(
                x=step_idx_gpu, y=max_dec_len, name="length_cond"
            )

        def body(tmp, out_0, step_idx_gpu, max_dec_len):
            paddle.increment(x=step_idx_gpu, value=1)

            param_attr = paddle.ParamAttr(
                name='conv2d.weight_1',
                initializer=paddle.nn.initializer.Constant(1.0),
            )
            res = paddle.static.nn.conv2d(
                input=tmp,
                num_filters=2,
                filter_size=3,
                act="relu",
                param_attr=param_attr,
            )

            out_0 = paddle.add(res, step_idx_gpu)

            return [tmp, out_0, step_idx_gpu, max_dec_len]

        main_program = paddle.static.default_main_program()
        startup_program = paddle.static.default_startup_program()
        with paddle.static.program_guard(main_program, startup_program):
            max_dec_len = paddle.full(
                shape=[1], fill_value=12, dtype='float32'
            )  # loop length
            step_idx_gpu = paddle.full(shape=[1], fill_value=0, dtype='float32')

            tmp = paddle.static.data(
                name='x', shape=[32, 3, 224, 224], dtype='float32'
            )

            param_attr = paddle.ParamAttr(
                name='conv2d.weight_0',
                initializer=paddle.nn.initializer.Constant(1.0),
            )
            out_1 = paddle.static.nn.conv2d(
                input=tmp,
                num_filters=2,
                filter_size=3,
                act="relu",
                param_attr=param_attr,
            )

            out_0 = paddle.full(
                shape=[32, 2, 222, 222], dtype='float32', fill_value=0
            )

            _, out_0, _, _ = paddle.static.nn.while_loop(
                cond, body, [tmp, out_0, step_idx_gpu, max_dec_len]
            )

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup_program)

            model_path = "./model"
            paddle.static.save_inference_model(
                model_path, [tmp], [out_0, out_1], exe
            )

    def test_all(self):
        compile_version = paddle_infer.get_trt_compile_version()
        runtime_version = paddle_infer.get_trt_runtime_version()
        if (
            compile_version[0] * 1000
            + compile_version[1] * 100
            + compile_version[2] * 10
            < 8400
        ):
            return True
        if (
            runtime_version[0] * 1000
            + runtime_version[1] * 100
            + runtime_version[2] * 10
            < 8400
        ):
            return True

        from paddle.inference import Config, create_predictor

        np_data = np.ones((32, 3, 224, 224)).astype("float32")

        # load inference model
        model_path = "./model"

        config_trt = Config(model_path + ".pdmodel", model_path + ".pdiparams")
        config_trt.enable_use_gpu(100, 0)
        config_trt.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=0,
            precision_mode=paddle.inference.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False,
        )
        config_trt.set_trt_dynamic_shape_info(
            {
                "x": [32, 3, 224, 224],
                "fill_constant_3.tmp_0": [1],
                "fill_constant_1.tmp_0": [1],
                "fill_constant_5.tmp_0": [32, 2, 222, 222],
            },
            {
                "x": [32, 3, 224, 224],
                "fill_constant_3.tmp_0": [1],
                "fill_constant_1.tmp_0": [1],
                "fill_constant_5.tmp_0": [32, 2, 222, 222],
            },
            {
                "x": [32, 3, 224, 224],
                "fill_constant_3.tmp_0": [1],
                "fill_constant_1.tmp_0": [1],
                "fill_constant_5.tmp_0": [32, 2, 222, 222],
            },
        )
        predictor_trt = create_predictor(config_trt)
        input_tensor_trt = predictor_trt.get_input_handle(
            predictor_trt.get_input_names()[0]
        )
        input_tensor_trt.reshape(np_data.shape)
        input_tensor_trt.copy_from_cpu(np_data.copy())
        predictor_trt.run()
        predict_trt = predictor_trt.get_output_handle(
            predictor_trt.get_output_names()[0]
        ).copy_to_cpu()

        config_gpu = Config(model_path + ".pdmodel", model_path + ".pdiparams")
        config_gpu.enable_use_gpu(100, 0)
        predictor_gpu = create_predictor(config_gpu)
        input_tensor_gpu = predictor_gpu.get_input_handle(
            predictor_gpu.get_input_names()[0]
        )
        input_tensor_gpu.reshape(np_data.shape)
        input_tensor_gpu.copy_from_cpu(np_data.copy())
        predictor_gpu.run()
        predict_gpu = predictor_gpu.get_output_handle(
            predictor_gpu.get_output_names()[0]
        ).copy_to_cpu()

        check_output_allclose(
            np.array(predict_trt).flatten(),
            np.array(predict_gpu).flatten(),
            "predict",
        )


if __name__ == '__main__':
    unittest.main()

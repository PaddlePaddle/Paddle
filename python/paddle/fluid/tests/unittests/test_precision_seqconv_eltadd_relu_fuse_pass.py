# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import paddle.fluid as fluid
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import PaddleDType
from paddle.fluid.core import create_paddle_predictor


class TestSeqconvEltaddReluPass(unittest.TestCase):
    def test_seqconv_eltadd_relu_pass_cpu_precision(self):
        x = fluid.data(name='x', shape=[None, 1], lod_level=1)
        w_param = fluid.ParamAttr(
            name='seq_weight',
            initializer=fluid.initializer.ConstantInitializer(value=2.0))
        b_param = fluid.ParamAttr(
            name='seq_bias',
            initializer=fluid.initializer.ConstantInitializer(value=2.0))
        seqconv_res = fluid.layers.sequence_conv(
            x,
            num_filters=2,
            filter_size=2,
            param_attr=w_param,
            bias_attr=b_param,
            act='relu')
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        np_x = np.array([i for i in range(10)]).reshape(
            [10, 1]).astype('float32')
        in_data = fluid.create_lod_tensor(np_x, [[2, 3, 5]], fluid.CPUPlace())
        fw_output = exe.run(feed={"x": in_data},
                            fetch_list=[seqconv_res],
                            return_numpy=False)
        path = "./tmp/inference_model"
        fluid.io.save_inference_model(
            dirname=path,
            feeded_var_names=['x'],
            target_vars=[seqconv_res],
            executor=exe)
        config = AnalysisConfig(path)
        config.disable_gpu()
        predictor = create_paddle_predictor(config)
        data = np.array([i for i in range(10)]).reshape(
            [10, 1]).astype('float32')
        inputs = PaddleTensor(data)
        inputs.name = 'x'
        inputs.dtype = PaddleDType.FLOAT32
        inputs.lod = [[0, 2, 5, 10]]
        outputs = predictor.run([inputs])
        output = outputs[0]
        output_data = output.as_ndarray()
        self.assertEqual(output_data.shape, np.array(fw_output[0]).shape)
        self.assertTrue(
            np.allclose(
                np.array(fw_output[0]).ravel(), output_data.ravel(),
                rtol=1e-05))

    def test_seqconv_eltadd_relu_pass_gpu_precision(self):
        x = fluid.data(name='x', shape=[None, 1], lod_level=1)
        w_param = fluid.ParamAttr(
            name='seq_weight',
            initializer=fluid.initializer.ConstantInitializer(value=2.0))
        b_param = fluid.ParamAttr(
            name='seq_bias',
            initializer=fluid.initializer.ConstantInitializer(value=2.0))
        seqconv_res = fluid.layers.sequence_conv(
            x,
            num_filters=2,
            filter_size=2,
            param_attr=w_param,
            bias_attr=b_param,
            act='relu')
        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        np_x = np.array([i for i in range(10)]).reshape(
            [10, 1]).astype('float32')
        in_data = fluid.create_lod_tensor(np_x, [[2, 3, 5]], fluid.CPUPlace())
        fw_output = exe.run(feed={"x": in_data},
                            fetch_list=[seqconv_res],
                            return_numpy=False)
        path = "./tmp/inference_model"
        fluid.io.save_inference_model(
            dirname=path,
            feeded_var_names=['x'],
            target_vars=[seqconv_res],
            executor=exe)
        config = AnalysisConfig(path)
        config.enable_use_gpu(100, 0)
        predictor = create_paddle_predictor(config)
        data = np.array([i for i in range(10)]).reshape(
            [10, 1]).astype('float32')
        inputs = PaddleTensor(data)
        inputs.name = 'x'
        inputs.dtype = PaddleDType.FLOAT32
        inputs.lod = [[0, 2, 5, 10]]
        outputs = predictor.run([inputs])
        output = outputs[0]
        output_data = output.as_ndarray()
        self.assertEqual(output_data.shape, np.array(fw_output[0]).shape)
        self.assertTrue(
            np.allclose(
                np.array(fw_output[0]).ravel(), output_data.ravel(),
                rtol=1e-05))


if __name__ == '__main__':
    unittest.main()

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
import os
import unittest
import paddle.fluid as fluid
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import PaddleDType
from paddle.fluid.core import create_paddle_predictor


class TestConvEltaddActFusePass(unittest.TestCase):
    '''This pass only enabled on gpu now. 
    '''

    def test_conv_eltadd_fuse_pass_gpu_precision(self):
        x = fluid.data(name='x', shape=[-1, 3, 100, 100])
        conv_res = fluid.layers.conv2d(
            x, num_filters=3, filter_size=3, act='relu')
        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        np_x = np.array([i % 255 for i in range(30000)]).reshape(
            [1, 3, 100, 100]).astype('float32')
        fw_output = exe.run(feed={"x": np_x}, fetch_list=[conv_res])
        print(fw_output[0])
        path = "./tmp/inference_model"
        fluid.io.save_inference_model(
            dirname=path,
            feeded_var_names=['x'],
            target_vars=[conv_res],
            executor=exe)
        config = AnalysisConfig(path)
        config.enable_use_gpu(100, 0)
        predictor = create_paddle_predictor(config)
        data = np.array([i % 255 for i in range(30000)]).reshape(
            [1, 3, 100, 100]).astype('float32')
        inputs = PaddleTensor(data)
        outputs = predictor.run([inputs])
        output = outputs[0]
        output_data = output.as_ndarray()
        print(output_data)
        self.assertEqual(output_data.shape, np.array(fw_output[0]).shape)
        self.assertTrue(
            np.allclose(
                np.array(fw_output[0]).ravel(), output_data.ravel(),
                rtol=1e-05))
        os.removedirs("./tmp/")


if __name__ == '__main__':
    unittest.main()

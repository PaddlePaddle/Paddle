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

import unittest
import os
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor


class TestConvBnFusePrecision(unittest.TestCase):
    def test_conv_bn_fuse_cpu_precision(self):
        # forward process
        x = fluid.data(name="x", shape=[-1, 3, 100, 100], dtype='float32')
        conv_res = fluid.layers.conv2d(
            input=x, num_filters=3, filter_size=3, act=None, bias_attr=False)
        bn_res = fluid.layers.batch_norm(input=conv_res, is_test=True)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        np_x = np.array([i % 255 for i in range(1 * 3 * 100 * 100)]).reshape(
            [1, 3, 100, 100]).astype('float32')
        fw_output = exe.run(feed={"x": np_x}, fetch_list=[bn_res])
        # save the model
        path = "./tmp/inference_model"
        fluid.io.save_inference_model(
            dirname=path,
            feeded_var_names=['x'],
            target_vars=[bn_res],
            executor=exe)
        # predictor with conv_bn_fusion
        config = AnalysisConfig(path)
        config.disable_gpu()
        predictor = create_paddle_predictor(config)
        # prepare fake data
        data = np.array([i % 255 for i in range(1 * 3 * 100 * 100)]).reshape(
            [1, 3, 100, 100]).astype('float32')
        inputs = PaddleTensor(data)
        outputs = predictor.run([inputs])
        output = outputs[0]
        print(type(output))
        output_data = output.as_ndarray()
        # compare the precision
        self.assertEqual(output_data.shape, np.array(fw_output[0]).shape)
        self.assertTrue(
            np.allclose(
                np.array(fw_output[0]).ravel(), output_data.ravel(),
                atol=1e-04))
        files = os.listdir(path)
        for item in files:
            f_path = os.path.join(path, item)
            os.remove(f_path)
        os.removedirs("./tmp/")


if __name__ == '__main__':
    unittest.main()

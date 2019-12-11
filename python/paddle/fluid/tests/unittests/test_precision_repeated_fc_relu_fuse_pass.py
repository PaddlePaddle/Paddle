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
    '''This pass only enabled on cpu
    '''

    def test_conv_eltadd_pass_cpu_precision(self):
        x = fluid.data(name='x', shape=[10, 10])

        weight_param1 = fluid.ParamAttr(
            name='fc_weight1',
            initializer=fluid.initializer.NormalInitializer())
        fc1_res = fluid.layers.fc(x,
                                  size=100,
                                  act='relu',
                                  param_attr=weight_param1,
                                  bias_attr=None)

        weight_param2 = fluid.ParamAttr(
            name='fc_weight2',
            initializer=fluid.initializer.NormalInitializer())
        fc2_res = fluid.layers.fc(fc1_res,
                                  size=100,
                                  act='relu',
                                  param_attr=weight_param2,
                                  bias_attr=None)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        np_x = np.array([float(i / 255) for i in range(100)]).reshape(
            [10, 10]).astype('float32')
        fw_output = exe.run(feed={"x": np_x}, fetch_list=[fc2_res])
        print(fw_output[0])
        path = "./tmp/inference_model"
        fluid.io.save_inference_model(
            dirname=path,
            feeded_var_names=['x'],
            target_vars=[fc2_res],
            executor=exe)
        config = AnalysisConfig(path)
        config.disable_gpu()
        predictor = create_paddle_predictor(config)
        data = np.array([float(i / 255) for i in range(100)]).reshape(
            [10, 10]).astype('float32')
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


if __name__ == '__main__':
    unittest.main()

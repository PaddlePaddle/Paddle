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

import unittest
import paddle
import numpy as np
import os
from paddle.fluid import core

paddle.enable_static()


class TestDistModelRun(unittest.TestCase):
    def test_dist_model_run(self):
        # step 0: declare folder to save the model and params
        folder = './dist_model_run_test/'
        file = 'inf'
        path_prefix = folder + file

        # step 1: saving the inference model and params
        x = paddle.static.data(name='x', shape=[28, 28], dtype='float32')
        y = paddle.static.data(name='y', shape=[28, 1], dtype='int64')
        predict = paddle.static.nn.fc(x, 10, activation='softmax')
        loss = paddle.nn.functional.cross_entropy(predict, y)
        avg_loss = paddle.tensor.stat.mean(loss)
        exe = paddle.static.Executor(paddle.CUDAPlace(0))
        exe.run(paddle.static.default_startup_program())
        x_data = np.random.randn(28, 28).astype('float32')
        y_data = np.random.randint(0, 9, size=[28, 1]).astype('int64')
        exe.run(paddle.static.default_main_program(),
                feed={'x': x_data,
                      'y': y_data},
                fetch_list=[avg_loss])
        paddle.static.save_inference_model(path_prefix, [x, y], [avg_loss], exe)
        print('save model to', path_prefix)

        # step 2: prepare fake data for the inference
        x_tensor = np.random.randn(28, 28).astype('float32')
        y_tensor = np.random.randint(0, 9, size=[28, 1]).astype('int64')

        # step 3: init the dist model to inference with fake data
        config = core.DistModelConfig()
        config.model_dir = path_prefix
        config.place = 'GPU'
        dist = core.DistModel(config)
        dist.init()
        dist_x = core.DistModelTensor(x_tensor, 'x')
        dist_y = core.DistModelTensor(y_tensor, 'y')
        input_data = [dist_x, dist_y]
        output_rst = dist.run(input_data)
        dist_model_rst = output_rst[0].as_ndarray().ravel().tolist()
        print("dist model rst:", dist_model_rst)

        # step 4: use framework's api to inference with fake data
        [inference_program, feed_target_names, fetch_targets] = (
            paddle.static.load_inference_model(path_prefix, exe))
        results = exe.run(inference_program,
                          feed={'x': x_tensor,
                                'y': y_tensor},
                          fetch_list=fetch_targets)
        load_inference_model_rst = results[0]
        print("load inference model api rst:", load_inference_model_rst)

        # step 5: compare two results
        self.assertTrue(np.allclose(dist_model_rst, load_inference_model_rst))

        # step 6: clean up the env, delete the saved model and params
        os.remove(path_prefix + '.pdiparams')
        os.remove(path_prefix + '.pdmodel')
        os.rmdir(folder)
        print('cleaned up the env')


if __name__ == '__main__':
    unittest.main()

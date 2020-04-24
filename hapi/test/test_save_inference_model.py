# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import print_function

import unittest
import os
import numpy as np
import shutil
import tempfile

import paddle
from paddle import fluid

from hapi.model import Model, Input
from hapi.vision.models import resnet18


class TestSaveInferenceModel(unittest.TestCase):
    def tearDown(self):
        shutil.rmtree(self.save_dir)

    def export_deploy_model(self):
        model = resnet18()

        inputs = [Input([None, 3, 224, 224], 'float32', name='image')]

        model.prepare(inputs=inputs)

        self.save_dir = tempfile.mkdtemp()
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        model.save_inference_model(self.save_dir)

        place = fluid.CPUPlace() if not fluid.is_compiled_with_cuda(
        ) else fluid.CUDAPlace(0)
        exe = fluid.Executor(place)

        [inference_program, feed_target_names, fetch_targets] = (
            fluid.io.load_inference_model(
                dirname=self.save_dir, executor=exe))
        tensor_img = np.array(
            np.random.random((1, 3, 224, 224)), dtype=np.float32)
        ori_results = model.test_batch(tensor_img)
        results = exe.run(inference_program,
                          feed={feed_target_names[0]: tensor_img},
                          fetch_list=fetch_targets)

        np.testing.assert_allclose(results, ori_results)

    def test_save_inference_model(self):
        self.export_deploy_model()


if __name__ == '__main__':
    unittest.main()

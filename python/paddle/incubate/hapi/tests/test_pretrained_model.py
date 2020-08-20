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

import unittest
import numpy as np

import paddle.fluid as fluid
import paddle.incubate.hapi.vision.models as models
from paddle.incubate.hapi import Model, Input


# test the predicted resutls of static graph and dynamic graph are equal
# when used pretrained model
class TestPretrainedModel(unittest.TestCase):
    def infer(self, x, arch, dygraph=True):
        if dygraph:
            fluid.enable_dygraph()

        net = models.__dict__[arch](pretrained=True, classifier_activation=None)
        inputs = [Input([None, 3, 224, 224], 'float32', 'image')]
        model = Model(network=net, inputs=inputs)
        model.prepare()
        res = model.test_batch(x)

        if dygraph:
            fluid.disable_dygraph()
        return res

    def test_models(self):
        arches = ['mobilenet_v1', 'mobilenet_v2', 'resnet18']
        for arch in arches:
            x = np.array(np.random.random((2, 3, 224, 224)), dtype=np.float32)
            y_dygraph = self.infer(x, arch)
            y_static = self.infer(x, arch, dygraph=False)
            np.testing.assert_allclose(y_dygraph, y_static)


if __name__ == '__main__':
    unittest.main()

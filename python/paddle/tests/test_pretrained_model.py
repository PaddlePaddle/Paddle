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

import os
import unittest
import tempfile
import shutil
import numpy as np

import paddle
from paddle.static import InputSpec
import paddle.vision.models as models

from paddle.fluid.framework import (  # noqa: F401
    convert_np_dtype_to_dtype_,
    _non_static_mode,
    _varbase_creator,
    in_dygraph_mode,
    _in_legacy_dygraph,
)
from paddle.fluid import core

# test the predicted resutls of static graph and dynamic graph are equal
# when used pretrained model


class TestPretrainedModel(unittest.TestCase):
    def infer(self, arch):
        path = os.path.join(tempfile.mkdtemp(), '.cache_test_pretrained_model')
        if not os.path.exists(path):
            os.makedirs(path)
        x = np.array(np.random.random((2, 3, 224, 224)), dtype=np.float32)

        if _in_legacy_dygraph():
            print("legacy")
            x_without_holder = core.VarBase()
        else:
            print("dygraph")
            x_without_holder = core.eager.Tensor()
            x_without_holder._set_use_cudnn(False)

        print("is dygraph", in_dygraph_mode())
        test_x = paddle.to_tensor(x)
        print("is dense", test_x.is_dense())
        test_x._set_use_cudnn(False)

        res = {}
        for dygraph in [True, False]:
            if not dygraph:
                paddle.enable_static()

            net = models.__dict__[arch](pretrained=True)
            inputs = [InputSpec([None, 3, 224, 224], 'float32', 'image')]
            model = paddle.Model(network=net, inputs=inputs)
            model.prepare()

            if dygraph:
                model.save(path)
                res['dygraph'] = model.predict_batch(x)
            else:
                model.load(path)
                res['static'] = model.predict_batch(x)

            if not dygraph:
                paddle.disable_static()

        shutil.rmtree(path)
        np.testing.assert_allclose(res['dygraph'], res['static'])

    def test_models(self):
        # TODO (LielinJiang): when model file cache is ok. add following test back
        # 'resnet18', 'vgg16', 'alexnet', 'resnext50_32x4d', 'inception_v3',
        # 'densenet121', 'googlenet', 'wide_resnet50_2', 'wide_resnet101_2'
        arches = [
            'shufflenet_v2_x0_25',  # bug model
        ]
        for arch in arches:
            self.infer(arch)


if __name__ == '__main__':
    unittest.main()

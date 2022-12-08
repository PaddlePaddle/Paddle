# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved
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

import paddle
from paddle.inference import (
    PlaceType,
    PrecisionType,
    convert_to_mixed_precision,
)
from paddle.jit import to_static
from paddle.static import InputSpec
from paddle.vision.models import resnet50


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or paddle.get_cudnn_version() < 8000,
    'should compile with cuda.',
)
class TestConvertToMixedPrecision(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        model = resnet50(True)
        net = to_static(
            model, input_spec=[InputSpec(shape=[None, 3, 224, 224], name='x')]
        )
        paddle.jit.save(
            net, os.path.join(self.temp_dir.name, 'resnet50/inference')
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def convert_to_fp16(self):
        convert_to_mixed_precision(
            os.path.join(self.temp_dir.name, 'resnet50/inference.pdmodel'),
            os.path.join(self.temp_dir.name, 'resnet50/inference.pdiparams'),
            os.path.join(self.temp_dir.name, 'mixed/inference.pdmodel'),
            os.path.join(self.temp_dir.name, 'mixed/inference.pdiparams'),
            PrecisionType.Half,
            PlaceType.GPU,
            True,
        )

    def convert_to_fp16_with_fp16_input(self):
        convert_to_mixed_precision(
            os.path.join(self.temp_dir.name, 'resnet50/inference.pdmodel'),
            os.path.join(self.temp_dir.name, 'resnet50/inference.pdiparams'),
            os.path.join(self.temp_dir.name, 'mixed1/inference.pdmodel'),
            os.path.join(self.temp_dir.name, 'mixed1/inference.pdiparams'),
            PrecisionType.Half,
            PlaceType.GPU,
            False,
        )

    def convert_to_fp16_with_blacklist(self):
        convert_to_mixed_precision(
            os.path.join(self.temp_dir.name, 'resnet50/inference.pdmodel'),
            os.path.join(self.temp_dir.name, 'resnet50/inference.pdiparams'),
            os.path.join(self.temp_dir.name, 'mixed2/inference.pdmodel'),
            os.path.join(self.temp_dir.name, 'mixed2/inference.pdiparams'),
            PrecisionType.Half,
            PlaceType.GPU,
            False,
            set('conv2d'),
        )

    def convert_to_bf16(self):
        convert_to_mixed_precision(
            os.path.join(self.temp_dir.name, 'resnet50/inference.pdmodel'),
            os.path.join(self.temp_dir.name, 'resnet50/inference.pdiparams'),
            os.path.join(self.temp_dir.name, 'mixed3/inference.pdmodel'),
            os.path.join(self.temp_dir.name, 'mixed3/inference.pdiparams'),
            PrecisionType.Bfloat16,
            PlaceType.GPU,
            True,
        )

    def test_convert(self):
        self.convert_to_fp16()
        self.convert_to_fp16_with_fp16_input()
        self.convert_to_fp16_with_blacklist()
        self.convert_to_bf16()


if __name__ == '__main__':
    unittest.main()

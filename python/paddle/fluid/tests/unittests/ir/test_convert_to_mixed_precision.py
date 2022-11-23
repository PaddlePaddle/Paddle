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

import unittest
<<<<<<< HEAD
import numpy as np
=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
import paddle

from paddle.vision.models import resnet50
from paddle.jit import to_static
from paddle.static import InputSpec

<<<<<<< HEAD
from paddle.inference import PrecisionType, BackendType
from paddle.inference import convert_to_mixed_precision


@unittest.skipIf(not paddle.is_compiled_with_cuda()
                 or paddle.get_cudnn_version() < 8000,
                 'should compile with cuda.')
class TestConvertToMixedPrecision(unittest.TestCase):

    def test_convert_to_fp16(self):
        model = resnet50(True)
        net = to_static(
            model, input_spec=[InputSpec(shape=[None, 3, 224, 224], name='x')])
        paddle.jit.save(net, 'resnet50/inference')
        convert_to_mixed_precision('resnet50/inference.pdmodel',
                                   'resnet50/inference.pdiparams',
                                   'mixed/inference.pdmodel',
                                   'mixed/inference.pdiparams',
                                   PrecisionType.Half, BackendType.GPU, True)
=======
from paddle.inference import PrecisionType, PlaceType
from paddle.inference import convert_to_mixed_precision


@unittest.skipIf(
    not paddle.is_compiled_with_cuda() or paddle.get_cudnn_version() < 8000,
    'should compile with cuda.',
)
class TestConvertToMixedPrecision(unittest.TestCase):
    def test_convert_to_fp16(self):
        model = resnet50(True)
        net = to_static(
            model, input_spec=[InputSpec(shape=[None, 3, 224, 224], name='x')]
        )
        paddle.jit.save(net, 'resnet50/inference')
        convert_to_mixed_precision(
            'resnet50/inference.pdmodel',
            'resnet50/inference.pdiparams',
            'mixed/inference.pdmodel',
            'mixed/inference.pdiparams',
            PrecisionType.Half,
            PlaceType.GPU,
            True,
        )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

    def test_convert_to_fp16_with_fp16_input(self):
        model = resnet50(True)
        net = to_static(
<<<<<<< HEAD
            model, input_spec=[InputSpec(shape=[None, 3, 224, 224], name='x')])
        paddle.jit.save(net, 'resnet50/inference')
        convert_to_mixed_precision('resnet50/inference.pdmodel',
                                   'resnet50/inference.pdiparams',
                                   'mixed1/inference.pdmodel',
                                   'mixed1/inference.pdiparams',
                                   PrecisionType.Half, BackendType.GPU, False)
=======
            model, input_spec=[InputSpec(shape=[None, 3, 224, 224], name='x')]
        )
        paddle.jit.save(net, 'resnet50/inference')
        convert_to_mixed_precision(
            'resnet50/inference.pdmodel',
            'resnet50/inference.pdiparams',
            'mixed1/inference.pdmodel',
            'mixed1/inference.pdiparams',
            PrecisionType.Half,
            PlaceType.GPU,
            False,
        )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

    def test_convert_to_fp16_with_blacklist(self):
        model = resnet50(True)
        net = to_static(
<<<<<<< HEAD
            model, input_spec=[InputSpec(shape=[None, 3, 224, 224], name='x')])
        paddle.jit.save(net, 'resnet50/inference')
        convert_to_mixed_precision('resnet50/inference.pdmodel',
                                   'resnet50/inference.pdiparams',
                                   'mixed2/inference.pdmodel',
                                   'mixed2/inference.pdiparams',
                                   PrecisionType.Half, BackendType.GPU, False,
                                   set('conv2d'))
=======
            model, input_spec=[InputSpec(shape=[None, 3, 224, 224], name='x')]
        )
        paddle.jit.save(net, 'resnet50/inference')
        convert_to_mixed_precision(
            'resnet50/inference.pdmodel',
            'resnet50/inference.pdiparams',
            'mixed2/inference.pdmodel',
            'mixed2/inference.pdiparams',
            PrecisionType.Half,
            PlaceType.GPU,
            False,
            set('conv2d'),
        )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

    def test_convert_to_bf16(self):
        model = resnet50(True)
        net = to_static(
<<<<<<< HEAD
            model, input_spec=[InputSpec(shape=[None, 3, 224, 224], name='x')])
        paddle.jit.save(net, 'resnet50/inference')
        convert_to_mixed_precision('resnet50/inference.pdmodel',
                                   'resnet50/inference.pdiparams',
                                   'mixed3/inference.pdmodel',
                                   'mixed3/inference.pdiparams',
                                   PrecisionType.Bfloat16, BackendType.GPU,
                                   True)
=======
            model, input_spec=[InputSpec(shape=[None, 3, 224, 224], name='x')]
        )
        paddle.jit.save(net, 'resnet50/inference')
        convert_to_mixed_precision(
            'resnet50/inference.pdmodel',
            'resnet50/inference.pdiparams',
            'mixed3/inference.pdmodel',
            'mixed3/inference.pdiparams',
            PrecisionType.Bfloat16,
            PlaceType.GPU,
            True,
        )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91


if __name__ == '__main__':
    unittest.main()

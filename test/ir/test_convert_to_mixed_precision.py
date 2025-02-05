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
            model,
            input_spec=[InputSpec(shape=[None, 3, 224, 224], name='x')],
            full_graph=True,
        )
        with paddle.pir_utils.OldIrGuard():
            paddle.jit.save(
                net, os.path.join(self.temp_dir.name, 'resnet50/inference')
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_convert_to_mixed_precision(self):
        mixed_precision_options = [
            PrecisionType.Half,
            PrecisionType.Half,
            PrecisionType.Half,
            PrecisionType.Bfloat16,
        ]
        keep_io_types_options = [True, False, False, True]
        black_list_options = [set(), set(), {'conv2d'}, set()]

        test_configs = zip(
            mixed_precision_options, keep_io_types_options, black_list_options
        )
        for mixed_precision, keep_io_types, black_list in test_configs:
            config = f'mixed_precision={mixed_precision}-keep_io_types={keep_io_types}-black_list={black_list}'
            with self.subTest(
                mixed_precision=mixed_precision,
                keep_io_types=keep_io_types,
                black_list=black_list,
            ):
                convert_to_mixed_precision(
                    os.path.join(
                        self.temp_dir.name, 'resnet50/inference.pdmodel'
                    ),
                    os.path.join(
                        self.temp_dir.name, 'resnet50/inference.pdiparams'
                    ),
                    os.path.join(
                        self.temp_dir.name, f'{config}/inference.pdmodel'
                    ),
                    os.path.join(
                        self.temp_dir.name, f'{config}/inference.pdiparams'
                    ),
                    backend=PlaceType.GPU,
                    mixed_precision=mixed_precision,
                    keep_io_types=keep_io_types,
                    black_list=black_list,
                )


if __name__ == '__main__':
    unittest.main()

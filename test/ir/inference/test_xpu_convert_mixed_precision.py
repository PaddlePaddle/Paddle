# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class ConvertMixedPrecision(unittest.TestCase):
    def test(self):
        with paddle.pir_utils.OldIrGuard():
            paddle.disable_static()
            self.temp_dir = tempfile.TemporaryDirectory()
            model = resnet50(True)
            net = to_static(
                model,
                input_spec=[InputSpec(shape=[None, 3, 224, 224], name='x')],
                full_graph=True,
            )
            paddle.jit.save(
                net, os.path.join(self.temp_dir.name, 'resnet50/inference')
            )
            convert_to_mixed_precision(
                os.path.join(self.temp_dir.name, 'resnet50/inference.pdmodel'),
                os.path.join(
                    self.temp_dir.name, 'resnet50/inference.pdiparams'
                ),
                os.path.join(
                    self.temp_dir.name, 'mixed_precision/inference.pdmodel'
                ),
                os.path.join(
                    self.temp_dir.name, 'mixed_precision/inference.pdiparams'
                ),
                backend=PlaceType.XPU,
                mixed_precision=PrecisionType.Half,
            )
            self.temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()

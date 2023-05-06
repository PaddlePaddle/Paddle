# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved
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
from paddle.inference import PlaceType, save_optimized_model
from paddle.jit import to_static
from paddle.static import InputSpec
from paddle.vision.models import resnet50


@unittest.skipIf(
    not paddle.is_compiled_with_xpu(),
    'should compile with xpu.',
)
class TestSaveOptimizedModel(unittest.TestCase):
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

    def test_save_optimized_model(self):
        backend_options = [PlaceType.XPU, PlaceType.CPU]
        black_list_options = [set(), {"fc_fuse_pass"}]

        test_configs = zip(backend_options, black_list_options)

        for backend, black_list in test_configs:
            config = f'backend={backend}-black_list={black_list}'
            with self.subTest(
                backend=backend,
                black_list=black_list,
            ):
                save_optimized_model(
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
                    backend=backend,
                    black_list=black_list,
                )


if __name__ == '__main__':
    unittest.main()

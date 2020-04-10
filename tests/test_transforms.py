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

# when test, you should add hapi root path to the PYTHONPATH,
# export PYTHONPATH=PATH_TO_HAPI:$PYTHONPATH
import unittest

from datasets.folder import DatasetFolder
from transform import transforms


class TestTransforms(unittest.TestCase):
    def do_transform(self, trans):
        dataset_folder = DatasetFolder('test_data', transform=trans)

        for _ in dataset_folder:
            pass

    def test_trans0(self):
        normalize = transforms.Normalize(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375])
        trans = transforms.Compose([
            transforms.RandomResizedCrop(224), transforms.GaussianNoise(),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4,
                hue=0.4), transforms.RandomHorizontalFlip(),
            transforms.Permute(mode='CHW'), normalize
        ])

        self.do_transform(trans)

    def test_trans1(self):
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        self.do_transform(trans)

    def test_trans2(self):
        trans = transforms.Compose([transforms.CenterCropResize(224)])
        self.do_transform(trans)


if __name__ == '__main__':
    unittest.main()

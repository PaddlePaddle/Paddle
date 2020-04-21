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
import cv2
import math
import random
import numpy as np

from hapi.datasets import DatasetFolder
from hapi.vision.transforms import transforms
from paddle import fluid


class ImageNetDataset(DatasetFolder):
    def __init__(self,
                 path,
                 mode='train',
                 image_size=224,
                 resize_short_size=256):
        super(ImageNetDataset, self).__init__(path)
        self.mode = mode

        normalize = transforms.Normalize(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375])
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.Permute(mode='CHW'), normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(resize_short_size),
                transforms.CenterCrop(image_size),
                transforms.Permute(mode='CHW'), normalize
            ])

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path).astype(np.float32)
        label = np.array([label])
        return self.transform(img, label)

    def __len__(self):
        return len(self.samples)

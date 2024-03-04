# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# repo: PaddleDetection
# model: configs^sparse_rcnn^sparse_rcnn_r50_fpn_3x_pro100_coco_single_dy2st_train
# method:flatten||api:paddle.tensor.ops.sigmoid||method:flatten||api:paddle.tensor.manipulation.concat||method:__gt__||method:all
import unittest
import random


class Test2(unittest.TestCase):
    def rand(self):
        return random.randint(1, 2)

    def test_add(self):
        assert self.rand() == 1


if __name__ == '__main__':
    unittest.main()

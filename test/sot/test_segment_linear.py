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

import unittest

from test_case_base import TestCaseBase

import paddle
from paddle import nn
from paddle.jit import sot
from paddle.jit.sot.utils import strict_mode_guard


class Head(nn.Layer):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(10, 150)

    def forward(self, x, patch_embed_size):
        masks = self.head(x)
        # [b, (h w), c] -> [b, c, h, w]
        h, w = patch_embed_size[0], patch_embed_size[1]
        masks = masks.reshape((1, h, w, paddle.shape(masks)[-1]))
        masks = masks.transpose((0, 3, 1, 2))
        return masks


class SimpleNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.tmp = nn.Linear(1, 1024 * 10)
        self.tmp2 = nn.Linear(1, 1 * 10 * 32 * 32)
        self.head = Head()

    def getshape(self, x):
        x = self.tmp2(x.mean().reshape([1])).reshape([1, 10, 32, 32])
        x = paddle.shape(x)
        return x

    def forward(self, x):
        sot.psdb.fallback()
        shape = self.getshape(x)
        feat = self.tmp(x.mean().reshape([1])).reshape([1, 1024, 10])
        logits = self.head(feat, shape[2:])
        return logits


class TestSegmentLinear(TestCaseBase):
    @strict_mode_guard(False)
    def test_simple(self):
        x = paddle.randn((1, 8, 8))
        net = SimpleNet()
        net = paddle.jit.to_static(
            net, full_graph=False
        )  # dont make effect. we need fetch sot PR in paddle.
        loss = net(x)
        loss = loss.sum()
        loss.backward()


if __name__ == "__main__":
    unittest.main()

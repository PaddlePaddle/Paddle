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

from __future__ import division
from __future__ import print_function

import unittest

import os
import time
import copy
import subprocess

import paddle
import paddle.fluid as fluid


class TestHub(unittest.TestCase):
    def testLoad(self, ):
        github_model = paddle.hub.load(
            'lyuwenyu/PaddleClas:hub_L',
            model='ResNet18',
            source='github',
            force_reload=True)
        local_model = paddle.hub.load(
            '~/.cache/paddle/hub/lyuwenyu_PaddleClas_hub_L',
            model='ResNet18',
            source='github',
            force_reload=False)
        assert type(github_model) == type(local_model), 'hub.load'

    def testHelp(self, ):
        github_docs = paddle.hub.help(
            'lyuwenyu/PaddleClas:hub_L',
            model='ResNet18',
            source='github',
            force_reload=True)
        local_docs = paddle.hub.list(
            '~/.cache/paddle/hub/lyuwenyu_PaddleClas_hub_L', source='local')
        assert github_docs == local_docs, 'hub.help'

    def testList(self, ):
        github_entries = paddle.hub.list(
            'lyuwenyu/PaddleClas:hub_L', source='github', force_reload=True)
        local_entries = paddle.hub.list(
            '~/.cache/paddle/hub/lyuwenyu_PaddleClas_hub_L', source='local')
        assert github_entries == local_entries, 'hub.list'


if __name__ == '__main__':
    unittest.main()

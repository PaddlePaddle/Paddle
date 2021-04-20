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
import paddle.hapi.hub as hub


class TestHub(unittest.TestCase):
    def testLoad(self, ):
        model = hub.load(
            'lyuwenyu/paddlehub_demo:main',
            model='MM',
            source='github',
            force_reload=True)

        model = hub.load(
            'lyuwenyu/paddlehub_demo:main',
            model='MM',
            source='github',
            force_reload=True,
            pretrained=True)

        model = hub.load(
            'lyuwenyu/paddlehub_demo',
            model='MM',
            source='github',
            force_reload=True,
            pretrained=False)

    def testHelp(self, ):
        docs = hub.help(
            'lyuwenyu/paddlehub_demo:main',
            model='MM',
            source='github',
            force_reload=True)

        docs = hub.load(
            'lyuwenyu/paddlehub_demo',
            model='MM',
            source='github',
            force_reload=False)

    def testList(self, ):
        models = hub.list(
            'lyuwenyu/paddlehub_demo:main',
            source='github',
            force_reload=True, )

        models = hub.list(
            'lyuwenyu/paddlehub_demo', source='github', force_reload=True)


if __name__ == '__main__':
    unittest.main()

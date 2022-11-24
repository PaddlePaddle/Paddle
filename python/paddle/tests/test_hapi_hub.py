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

import unittest
import os

import paddle
from paddle.hapi import hub

import numpy as np


class TestHub(unittest.TestCase):
<<<<<<< HEAD

    def setUp(self, ):
        self.local_repo = os.path.dirname(os.path.abspath(__file__))
        self.github_repo = 'lyuwenyu/paddlehub_demo:main'

    def testLoad(self, ):
        model = hub.load(self.local_repo,
                         model='MM',
                         source='local',
                         out_channels=8)
=======
    def setUp(
        self,
    ):
        self.local_repo = os.path.dirname(os.path.abspath(__file__))
        self.github_repo = 'lyuwenyu/paddlehub_demo:main'

    def testLoad(
        self,
    ):
        model = hub.load(
            self.local_repo, model='MM', source='local', out_channels=8
        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

        data = paddle.rand((1, 3, 100, 100))
        out = model(data)
        np.testing.assert_equal(out.shape, [1, 8, 50, 50])

<<<<<<< HEAD
        model = hub.load(self.github_repo,
                         model='MM',
                         source='github',
                         force_reload=True)

        model = hub.load(self.github_repo,
                         model='MM',
                         source='github',
                         force_reload=False,
                         pretrained=False)

        model = hub.load(self.github_repo.split(':')[0],
                         model='MM',
                         source='github',
                         force_reload=False,
                         pretrained=False)

        model = hub.load(self.github_repo,
                         model='MM',
                         source='github',
                         force_reload=False,
                         pretrained=True,
                         out_channels=8)
=======
        model = hub.load(
            self.github_repo, model='MM', source='github', force_reload=True
        )

        model = hub.load(
            self.github_repo,
            model='MM',
            source='github',
            force_reload=False,
            pretrained=False,
        )

        model = hub.load(
            self.github_repo.split(':')[0],
            model='MM',
            source='github',
            force_reload=False,
            pretrained=False,
        )

        model = hub.load(
            self.github_repo,
            model='MM',
            source='github',
            force_reload=False,
            pretrained=True,
            out_channels=8,
        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

        data = paddle.ones((1, 3, 2, 2))
        out = model(data)

        gt = np.array(
            [
                1.53965068,
                0.0,
                0.0,
                1.39455748,
                0.72066200,
                0.19773030,
                2.09201908,
                0.37345418,
            ]
        )
        np.testing.assert_equal(out.shape, [1, 8, 1, 1])
<<<<<<< HEAD
        np.testing.assert_almost_equal(out.numpy(),
                                       gt.reshape(1, 8, 1, 1),
                                       decimal=5)
=======
        np.testing.assert_almost_equal(
            out.numpy(), gt.reshape(1, 8, 1, 1), decimal=5
        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

    def testHelp(
        self,
    ):
        docs1 = hub.help(
            self.local_repo,
            model='MM',
            source='local',
        )

<<<<<<< HEAD
        docs2 = hub.help(self.github_repo,
                         model='MM',
                         source='github',
                         force_reload=False)
=======
        docs2 = hub.help(
            self.github_repo, model='MM', source='github', force_reload=False
        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

        assert docs1 == docs2 == 'This is a test demo for paddle hub\n    ', ''

    def testList(
        self,
    ):
        models1 = hub.list(
            self.local_repo,
            source='local',
            force_reload=False,
        )

        models2 = hub.list(
            self.github_repo,
            source='github',
            force_reload=False,
        )

        assert models1 == models2 == ['MM'], ''

    def testExcept(
        self,
    ):
        with self.assertRaises(ValueError):
<<<<<<< HEAD
            _ = hub.help(self.github_repo,
                         model='MM',
                         source='github-test',
                         force_reload=False)

        with self.assertRaises(ValueError):
            _ = hub.load(self.github_repo,
                         model='MM',
                         source='github-test',
                         force_reload=False)

        with self.assertRaises(ValueError):
            _ = hub.list(self.github_repo,
                         source='github-test',
                         force_reload=False)

        with self.assertRaises(ValueError):
            _ = hub.load(self.local_repo,
                         model=123,
                         source='local',
                         force_reload=False)

        with self.assertRaises(RuntimeError):
            _ = hub.load(self.local_repo,
                         model='123',
                         source='local',
                         force_reload=False)
=======
            _ = hub.help(
                self.github_repo,
                model='MM',
                source='github-test',
                force_reload=False,
            )

        with self.assertRaises(ValueError):
            _ = hub.load(
                self.github_repo,
                model='MM',
                source='github-test',
                force_reload=False,
            )

        with self.assertRaises(ValueError):
            _ = hub.list(
                self.github_repo, source='github-test', force_reload=False
            )

        with self.assertRaises(ValueError):
            _ = hub.load(
                self.local_repo, model=123, source='local', force_reload=False
            )

        with self.assertRaises(RuntimeError):
            _ = hub.load(
                self.local_repo, model='123', source='local', force_reload=False
            )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f


if __name__ == '__main__':
    unittest.main()

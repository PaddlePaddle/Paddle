# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle.vision.transforms import transforms

SEED = 2022


class TestTransformUnitTestBase(unittest.TestCase):
    def setUp(self):
        self.img = (np.random.rand(*self.get_shape()) * 255.0).astype(
            np.float32
        )
        self.set_trans_api()

    def get_shape(self):
        return (64, 64, 3)

    def set_trans_api(self):
        self.api = transforms.Resize(size=16)

    def dynamic_transform(self):
        paddle.seed(SEED)

        img_t = paddle.to_tensor(self.img)
        return self.api(img_t)

    def static_transform(self):
        paddle.enable_static()
        paddle.seed(SEED)

        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data(
                shape=self.get_shape(), dtype=paddle.float32, name='img'
            )
            out = self.api(x)

        exe = paddle.static.Executor()
        res = exe.run(main_program, fetch_list=[out], feed={'img': self.img})

        paddle.disable_static()
        return res[0]

    def test_transform(self):
        dy_res = self.dynamic_transform()
        st_res = self.static_transform()

        np.testing.assert_almost_equal(dy_res, st_res)


class TestResize(TestTransformUnitTestBase):
    def set_trans_api(self):
        self.api = transforms.Resize(size=(16, 16))


class TestResizeError(TestTransformUnitTestBase):
    def test_transform(self):
        pass

    def test_error(self):
        paddle.enable_static()
        # Not support while w<=0 or h<=0, but received w=-1, h=-1
        with self.assertRaises(NotImplementedError):
            main_program = paddle.static.Program()
            with paddle.static.program_guard(main_program):
                x = paddle.static.data(
                    shape=[-1, -1, -1], dtype=paddle.float32, name='img'
                )
                self.api(x)

        paddle.disable_static()


class TestRandomVerticalFlip0(TestTransformUnitTestBase):
    def set_trans_api(self):
        self.api = transforms.RandomVerticalFlip(prob=0)


class TestRandomVerticalFlip1(TestTransformUnitTestBase):
    def set_trans_api(self):
        self.api = transforms.RandomVerticalFlip(prob=1)


if __name__ == "__main__":
    unittest.main()

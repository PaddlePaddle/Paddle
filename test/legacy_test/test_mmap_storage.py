# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import unittest

import numpy as np

import paddle


class TestMmapStorageBase(unittest.TestCase):
    def setUp(self):
        self.init_cfg()
        np.random.seed(2025)
        paddle.seed(2025)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pp') as tmpfile:
            self.file_name = tmpfile.name
        self.data = paddle.randn(self.shape, dtype=self.dtype)
        self.nbytes = self.data.size * self.data.element_size()

    def init_cfg(self):
        self.shape = [400, 50, 20]
        self.dtype = 'float64'

    def test_mmap_storage(self):
        self.data.numpy().tofile(self.file_name)
        tmp = paddle.MmapStorage(self.file_name, self.nbytes)
        res = tmp.get_slice(self.dtype, 0, self.data.size).reshape(self.shape)
        np.testing.assert_allclose(res.numpy(), self.data.numpy())


class TestMmapStorage1(TestMmapStorageBase):
    def init_cfg(self):
        self.shape = [300, 40, 10]
        self.dtype = 'float32'


class TestMmapStorage2(TestMmapStorageBase):
    def init_cfg(self):
        self.shape = [300, 40, 10]
        self.dtype = 'float16'


class TestMmapStorage3(TestMmapStorageBase):
    def init_cfg(self):
        self.shape = [300, 40, 10]
        self.dtype = 'bfloat16'


class TestMmapStorage4(TestMmapStorageBase):
    def setUp(self):
        self.init_cfg()
        np.random.seed(2025)
        paddle.seed(2025)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pp') as tmpfile:
            self.file_name = tmpfile.name
        self.data = paddle.randint(0, 128, self.shape).astype(self.dtype)
        self.nbytes = self.data.size * self.data.element_size()

    def init_cfg(self):
        self.shape = [300, 40, 10]
        self.dtype = 'int64'


class TestMmapStorage5(TestMmapStorage4):
    def init_cfg(self):
        self.shape = [300, 40, 10]
        self.dtype = 'int32'


class TestMmapStorage6(TestMmapStorage4):
    def init_cfg(self):
        self.shape = [300, 40, 10]
        self.dtype = 'int16'


class TestMmapStorage7(TestMmapStorage4):
    def init_cfg(self):
        self.shape = [300, 40, 10]
        self.dtype = 'int8'


class TestMmapStorage8(TestMmapStorage4):
    def setUp(self):
        self.init_cfg()
        np.random.seed(2025)
        paddle.seed(2025)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pp') as tmpfile:
            self.file_name = tmpfile.name
        self.data = paddle.randint(0, 2, self.shape).astype(self.dtype)
        self.nbytes = self.data.size * self.data.element_size()

    def init_cfg(self):
        self.shape = [300, 40, 10]
        self.dtype = 'bool'

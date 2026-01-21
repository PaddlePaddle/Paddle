# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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


class TestAlias(unittest.TestCase):
    def setUp(self):
        from paddle.io import Dataset as IoDataset
        from paddle.utils.data import Dataset as UtilsDataset

        self.ioObject = IoDataset
        self.utilsObject = UtilsDataset

    def test_compatibility(self):
        self.assertTrue(type(self.ioObject) == type(self.utilsObject))


class TestChainDatasetAlias(TestAlias):
    def setUp(self):
        from paddle.io import ChainDataset as IoChainDataset
        from paddle.utils.data import ChainDataset as UtilsChainDataset

        self.ioObject = IoChainDataset
        self.utilsObject = UtilsChainDataset


class TestConcatDatasetAlias(TestAlias):
    def setUp(self):
        from paddle.io import ConcatDataset as IoConcatDataset
        from paddle.utils.data import ConcatDataset as UtilsConcatDataset

        self.ioObject = IoConcatDataset
        self.utilsObject = UtilsConcatDataset


class TestIterableDatasetAlias(TestAlias):
    def setUp(self):
        from paddle.io import IterableDataset as IoIterableDataset
        from paddle.utils.data import IterableDataset as UtilsIterableDataset

        self.ioObject = IoIterableDataset
        self.utilsObject = UtilsIterableDataset


if __name__ == "__main__":
    unittest.main()

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


class TestFolderDatasets(unittest.TestCase):
    def test_dataset(self):
        dataset_folder = DatasetFolder('test_data')

        for _ in dataset_folder:
            pass

        assert len(dataset_folder) == 3
        assert len(dataset_folder.classes) == 2


if __name__ == '__main__':
    unittest.main()

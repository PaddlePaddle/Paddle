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

import numpy as np
import pytest
from api_base import ApiBase

import paddle


class TestScatter:
    def __init__(self):
        self.init()
        test = ApiBase(
            func=paddle.scatter,
            feed_names=['X', 'index', 'updates'],
            feed_shapes=[
                self.input_data.shape,
                self.index_data.shape,
                self.updates_data.shape,
            ],
            is_train=False,
            feed_dtypes=['float32', 'int64', 'float32'],
        )
        print(self.input_data, self.index_data, self.updates_data)
        test.run(
            feed=[self.input_data, self.index_data, self.updates_data],
            overwrite=self.overwrite,
        )

    def init(self):
        pass


class TestScatter0(TestScatter):
    def init(self):
        self.input_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float32)
        self.index_data = np.array([2, 1, 0, 1]).astype(np.int64)
        self.updates_data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).astype(
            np.float32
        )
        self.overwrite = True


class TestScatter1(TestScatter):
    def init(self):
        self.input_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float32)
        self.index_data = np.array([2, 1, 0, 1]).astype(np.int64)
        self.updates_data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).astype(
            np.float32
        )
        self.overwrite = False


class TestScatter2(TestScatter):
    def init(self):
        self.input_data = np.ones((3, 3)).astype("float32")
        self.index_data = np.array([1, 2]).astype("int64")
        self.updates_data = np.random.random((2, 3)).astype("float32")
        self.overwrite = True


class TestScatter3(TestScatter):
    def init(self):
        self.input_data = np.ones((3, 3)).astype("float32")
        self.index_data = np.array([1, 1]).astype("int64")
        self.updates_data = np.random.random((2, 3)).astype("float32")
        self.overwrite = False


@pytest.mark.stack
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_scatter():

    TestScatter0()
    # TestScatter3()
    TestScatter2()


if __name__ == '__main__':
    test_scatter()

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


class TestTileOpRank1:
    def __init__(self):
        support_types = ['int32', 'int64', 'float32', 'float64']
        for ty in support_types:
            self.init_data()
            self.attrs = {'repeat_times': self.repeat_times}
            self.test = ApiBase(
                func=paddle.tile,
                feed_names=['X'],
                feed_shapes=[self.ori_shape],
                is_train=False,
                feed_dtypes=[ty],
            )
            data = np.random.rand(*(self.ori_shape)).astype(ty)
            print(data.shape)
            self.test.run(feed=[data], **self.attrs)

    def init_data(self):
        self.ori_shape = [100, 1]
        self.repeat_times = [2, 1]


class TestTileOpRank2Expanding(TestTileOpRank1):
    def init_data(self):
        self.ori_shape = [32, 120]
        self.repeat_times = [2, 2]


class TestTileOpRank2(TestTileOpRank1):
    def init_data(self):
        self.ori_shape = [12, 14]
        self.repeat_times = [2, 3]


class TestTileOpRank3_Corner(TestTileOpRank1):
    def init_data(self):
        self.ori_shape = (2, 10, 5)
        self.repeat_times = (1, 1, 1)


class TestTileOpRank3(TestTileOpRank1):
    def init_data(self):
        self.ori_shape = (2, 4, 15)
        self.repeat_times = (2, 1, 4)


class TestTileOpRank4(TestTileOpRank1):
    def init_data(self):
        self.ori_shape = (2, 4, 5, 7)
        self.repeat_times = (3, 2, 1, 2)


@pytest.mark.tile
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_tile():
    TestTileOpRank1()
    TestTileOpRank2Expanding()

    TestTileOpRank2()

    TestTileOpRank3_Corner()
    TestTileOpRank3()
    TestTileOpRank4()


if __name__ == '__main__':
    test_tile()

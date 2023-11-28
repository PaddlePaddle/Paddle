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

test = ApiBase(
    func=paddle.metric.accuracy,
    feed_names=['data', 'label'],
    is_train=False,
    feed_shapes=[[64, 5], [64, 1]],
    feed_dtypes=['float32', 'int64'],
)


@pytest.mark.accuracy
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_accuracy():
    np.random.seed(1)
    # data = np.array([[0.2, 0.1, 0.4, 0.1, 0.1], [0.2, 0.3, 0.1, 0.15, 0.25]], dtype=np.float32)
    # label = np.array([[2], [0]], dtype=np.int64)
    data = np.random.random((64, 5)).astype("float32")
    data = data / np.sum(data, axis=1, keepdims=True)
    label = np.random.randint(0, 4, (64, 1)).astype("int64")
    test.run(feed=[data, label], k=2)


test_accuracy()

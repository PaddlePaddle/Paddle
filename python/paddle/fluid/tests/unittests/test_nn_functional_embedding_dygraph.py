#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest


class EmbeddingDygraph(unittest.TestCase):
    def test_1(self):
        import paddle
        import paddle.nn as nn
        import numpy as np
        paddle.disable_static()

        # example 1
        inp_word = np.array([[2, 3, 5], [4, 2, 1]]).astype('int64')
        inp_word.shape  # [2, 3]
        dict_size = 20

        emb = nn.Embedding(dict_size, 32, weight_attr='emb.w', sparse=False)


if __name__ == '__main__':
    unittest.main()

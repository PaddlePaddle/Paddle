#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

paddle.enable_static()


class TestSeqConvApi(unittest.TestCase):
    def test_api(self):
        from paddle import base

        x = paddle.static.data('x', shape=[-1, 32], lod_level=1)
        y = paddle.static.nn.sequence_lod.sequence_conv(
            input=x, num_filters=2, filter_size=3, padding_start=None
        )

        place = base.CPUPlace()
        x_tensor = base.create_lod_tensor(
            np.random.rand(10, 32).astype("float32"), [[2, 3, 1, 4]], place
        )
        exe = base.Executor(place)
        exe.run(base.default_startup_program())
        ret = exe.run(feed={'x': x_tensor}, fetch_list=[y], return_numpy=False)


if __name__ == '__main__':
    unittest.main()

# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import base
from paddle.incubate.layers.nn import search_pyramid_hash


class TestPyramidHashOpApi(unittest.TestCase):
    def test_api(self):
        num_voc = 128
        embed_dim = 64
        x_shape, x_lod = [16, 10], [[3, 5, 2, 6]]
        x = paddle.static.data(name='x', shape=x_shape, dtype='int32')
        hash_embed = search_pyramid_hash(
            input=x,
            num_emb=embed_dim,
            space_len=num_voc * embed_dim,
            pyramid_layer=4,
            rand_len=16,
            drop_out_percent=0.5,
            is_training=True,
            use_filter=False,
            white_list_len=6400,
            black_list_len=2800,
            seed=3,
            lr=0.002,
            param_attr=base.ParamAttr(
                name="PyramidHash_emb_0",
                learning_rate=0,
            ),
            param_attr_wl=base.ParamAttr(
                name="Filter",
                learning_rate=0,
            ),
            param_attr_bl=None,
            distribute_update_vars=["PyramidHash_emb_0"],
            name=None,
        )

        place = base.CPUPlace()
        x_tensor = base.create_lod_tensor(
            np.random.randint(0, num_voc, x_shape).astype('int32'), x_lod, place
        )

        exe = base.Executor(place)
        exe.run(base.default_startup_program())
        ret = exe.run(
            feed={'x': x_tensor}, fetch_list=[hash_embed], return_numpy=False
        )


if __name__ == "__main__":
    unittest.main()

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

import paddle
from paddle import base
from paddle.incubate.distributed.fleet import role_maker
from paddle.incubate.distributed.fleet.parameter_server.distribute_transpiler import (
    fleet,
)
from paddle.incubate.distributed.fleet.parameter_server.distribute_transpiler.distributed_strategy import (
    StrategyFactory,
)
from paddle.incubate.layers.nn import search_pyramid_hash


class TestPyramidHashOpApi(unittest.TestCase):
    def test_dist_geo_server_transpiler(self):
        num_voc = 128
        embed_dim = 64
        x_shape, x_lod = [16, 10], [[3, 5, 2, 6]]
        x = paddle.static.data(name='x', shape=x_shape, dtype='int32')
        hash_embd = search_pyramid_hash(
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

        cost = paddle.sum(hash_embd)

        role = role_maker.UserDefinedRoleMaker(
            current_id=0,
            role=role_maker.Role.SERVER,
            worker_num=2,
            server_endpoints=["127.0.0.1:36011", "127.0.0.1:36012"],
        )

        fleet.init(role)

        strategy = StrategyFactory.create_geo_strategy(5)
        optimizer = paddle.optimizer.SGD(0.1)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(cost)

        pserver_startup_program = fleet.startup_program
        pserver_mian_program = fleet.main_program


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()

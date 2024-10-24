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

from legacy_test.test_collective_api_base import (
    TestCollectiveAPIRunnerBase,
    runtime_main,
)

import paddle
from paddle import base, framework
from paddle.base import data_feeder

paddle.enable_static()


def concat_new(tensor, group=None):
    op_type = 'dist_concat'
    data_feeder.check_variable_and_dtype(
        tensor,
        'tensor',
        [
            'float16',
            'float32',
            'float64',
            'int32',
            'int64',
            'int8',
            'uint8',
            'bool',
            'uint16',
        ],
        op_type,
    )

    helper = framework.LayerHelper(op_type, **locals())
    ring_id = 0 if group is None else group.id
    nranks = 2

    out = helper.create_variable_for_type_inference(dtype=tensor.dtype)
    helper.append_op(
        type=op_type,
        inputs={'x': [tensor]},
        outputs={'out': [out]},
        attrs={
            'ring_id': ring_id,
            'nranks': nranks,
        },
    )
    return out


def concat_new_comm(tensor, group=None, rank=0):
    op_type = 'c_concat'
    data_feeder.check_variable_and_dtype(
        tensor,
        'tensor',
        [
            'float16',
            'float32',
            'float64',
            'int32',
            'int64',
        ],
        op_type,
    )

    helper = framework.LayerHelper(op_type, **locals())
    ring_id = 0 if group is None else group.id
    nranks = 2

    out = helper.create_variable_for_type_inference(dtype=tensor.dtype)
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [out]},
        attrs={
            'ring_id': ring_id,
            'nranks': nranks,
            'rank': rank,
        },
    )
    return out


class TestCollectiveConcatAPI(TestCollectiveAPIRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, dtype="float32"):
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(
                name="tindata", shape=[10, 1000], dtype=dtype
            )
            tindata.desc.set_need_check_feed(False)
            toutdata = concat_new_comm(tindata, rank=rank)

            return [toutdata]

    def get_model_new(
        self, main_prog, startup_program, rank, dtype=None, reduce_type=None
    ):
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(
                name="tindata", shape=[10, 1000], dtype=dtype
            )
            tindata.desc.set_need_check_feed(False)
            toutdata = concat_new(tindata)
            return [toutdata]


if __name__ == "__main__":
    runtime_main(TestCollectiveConcatAPI, "concat")

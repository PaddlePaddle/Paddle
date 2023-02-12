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

import paddle
import paddle.fluid as fluid
import numpy as np

paddle.enable_static()

MAX_SIZE_QUERY = 6
MAX_SIZE_RESPONSE = 1000

USE_IDS = True
RES_TYPE = 'float'

# network
in_query = fluid.data(name='X', shape=[MAX_SIZE_QUERY], dtype='int32')
in_url_id = fluid.data(name='url_id', shape=[1], dtype='int32')
req_id = fluid.data(name='rid', shape=[1], dtype='int32')
out_succeed = fluid.data(name='succeed', shape=[1], dtype='bool')
out_data = fluid.data(name='Out', shape=[MAX_SIZE_RESPONSE], dtype='float32')

default_prog = fluid.default_main_program()
cur_block = default_prog.current_block()
cur_block.append_op(
    type='rpc_call',
    inputs={
        'X': in_query,
        'url_id': in_url_id,
    },
    outputs={'Out': req_id},
    attrs={
        'url_list': [
            'http://10.127.2.19:8082/run/predict',
            'http://10.174.140.91:2001/wenxin/inference',
        ],
        'vocab_path': '/work/Paddle/vocab.txt',
        'use_ids': USE_IDS,
    },
)
cur_block.append_op(
    type='rpc_result',
    inputs={
        'X': req_id,
    },
    outputs={
        'succeed': out_succeed,
        'Out': out_data,
    },
    attrs={
        'res_dtype': RES_TYPE,
    },
)

# data
# 货物很好
query_tensor = np.array([29989, 29981, 2264, 1708, 1672, 1598], dtype='int32')
# 货物很差
# query_tensor = np.array([29989, 29981, 2264, 1708, 1672, 2212], dtype='int32')

url_id_tensor = np.array([1], dtype='int32')

# run
exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())
succeed, data, = exe.run(
    fluid.default_main_program(),
    feed={
        'X': query_tensor,
        'url_id': url_id_tensor,
    },
    fetch_list=[out_succeed, out_data],
)
if succeed:
    if RES_TYPE == 'str':
        print('output:', data.tobytes().decode('utf-8'))
    else:
        print('output:', data)
else:
    print('request failed')

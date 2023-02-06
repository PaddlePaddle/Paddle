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

MAX_SIZE_SERVICE = 20
MAX_SIZE_PAYLOAD = 100
MAX_SIZE_RESPONSE = 1000

# data
def pad_or_truncate(s, limit):
    """
    pad_or_truncate

    Args:
        s (str)
        limit (int)

    Returns:
        np.array
    """
    buf = bytearray(s, 'utf-8')
    if len(buf) <= limit:
        lack_size = limit - len(buf)
        buf += bytearray(lack_size)
    else:
        buf = buf[:MAX_SIZE_PAYLOAD]
    return np.frombuffer(buf, dtype='uint8')


service_str = "test"
service_tensor = pad_or_truncate(service_str, MAX_SIZE_SERVICE)

query_str = "paddle是什么"
query_tensor = pad_or_truncate(query_str, MAX_SIZE_PAYLOAD)

# network
in_service = fluid.data(name='service', shape=[MAX_SIZE_SERVICE], dtype='uint8')
in_query = fluid.data(name='X', shape=[MAX_SIZE_PAYLOAD], dtype='uint8')
req_id = fluid.data(name='rid', shape=[1], dtype='int')
out_data = fluid.data(name='Out', shape=[MAX_SIZE_RESPONSE], dtype='uint8')

default_prog = fluid.default_main_program()
cur_block = default_prog.current_block()
cur_block.append_op(
    type='rpc_call',
    inputs={
        'X': in_query,
        'service': in_service,
    },
    outputs={'Out': req_id},
)
cur_block.append_op(
    type='rpc_result',
    inputs={
        'X': req_id,
    },
    outputs={'Out': out_data},
)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())
(out,) = exe.run(
    fluid.default_main_program(),
    feed={
        'X': query_tensor,
        'service': service_tensor,
    },
    fetch_list=[out_data],
)
print('output:', out.tobytes().decode('utf-8'))

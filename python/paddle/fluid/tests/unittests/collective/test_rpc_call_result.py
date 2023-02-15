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
import os

paddle.enable_static()

MAX_SIZE_QUERY = 6
MAX_SIZE_RESPONSE = 1000

USE_IDS = True
RES_TYPE = 'float'

# network
in_query = fluid.data(name='X', shape=[-1, MAX_SIZE_QUERY], dtype='int32')

req_ids = paddle.static.nn.rpc_call(
    in_query,
    os.environ.get("url_list"),
    "/code_lp/ernie-bot/post-train/ernie_3.0_100b_no_distill/config/ernie3.0_vocab_multi_prompt_v9.txt",
    True,
)
out_data, out_succeed = paddle.static.nn.rpc_result(req_ids, RES_TYPE, 1)
paddle.static.Print(in_query)
paddle.static.Print(req_ids)
paddle.static.Print(out_data.astype("float32"))

# data
# 货物很好
# 货物很差

query_tensor = np.array(
    [
        [29989, 29981, 2264, 1708, 1672, 1598],
        [29989, 29981, 2264, 1708, 1672, 2212],
    ]
).astype("int32")

# run
exe = fluid.Executor(fluid.CUDAPlace(0))
exe.run(fluid.default_startup_program())
import time

t1 = time.time()
for _ in range(1):
    succeed, data, = exe.run(
        fluid.default_main_program(),
        feed={
            'X': query_tensor,
        },
        fetch_list=[out_succeed, out_data],
    )
t2 = time.time()
print("speed: ", (t2 - t1) / 1, "s/step")

if succeed:
    if RES_TYPE == 'str':
        for d in data:
            print('output:', d.tobytes().decode('utf-8'))
    else:
        print('output:', data)
else:
    print('request failed')

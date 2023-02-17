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
import subprocess
import unittest
import os


def rpc_test(use_ids, out_type, url):
    paddle.enable_static()

    MAX_SIZE_QUERY = 18
    RES_TYPE = out_type

    with open("vocab.txt", "w") as voc:
        voc.write("ABC 0\n")
        voc.write("EFG 1\n")
        voc.write("HIG 2\n")
        voc.write("[<S>] 3\n")
        voc.write("[<N>] 4\n")
        voc.write("[<t>] 5\n")
        voc.write("[<T>] 6\n")
        voc.write("##good 7\n")
        voc.write("bad@@ 8\n")
        voc.write("@@badok 9\n")
        voc.write("你好 10\n")
        voc.write("haha 11\n")
        voc.write("##haha@@ 12\n")
        voc.write("[PAD] 13\n")
        voc.write("[gEnd] 14\n")

    # network
    in_query = fluid.data(name='X', shape=[-1, MAX_SIZE_QUERY], dtype='int32')

    req_ids = paddle.static.nn.rpc_call(
        in_query,
        url,
        "vocab.txt",
        use_ids,
    )

    out_data, out_succeed = paddle.static.nn.rpc_result(req_ids, RES_TYPE)
    paddle.static.Print(in_query)
    paddle.static.Print(req_ids)
    paddle.static.Print(out_data.astype("float32"))

    query_tensor = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 14],
        ]
    ).astype("int32")

    # run
    exe = fluid.Executor(fluid.CUDAPlace(0))
    exe.run(fluid.default_startup_program())

    for _ in range(1):
        succeed, data, = exe.run(
            fluid.default_main_program(),
            feed={
                'X': query_tensor,
            },
            fetch_list=[out_succeed, out_data],
        )
    if out_type == "str":
        print(data[0].tobytes().decode("utf-8", "ignore"))
    else:
        print(data[0])


class RPCCallTest(unittest.TestCase):
    def test_cases(self):
        ip = 'localhost'
        port = int(os.environ.get("PADDLE_DIST_UT_PORT"))

        server_cmd = f"python py_server_test.py --ip {ip} --port {port}"
        with open(f"server.{port}.log", "w") as output:
            process = subprocess.Popen(
                server_cmd.split(), stdout=output, stderr=output
            )

        for uid in [True, False]:
            for otype in ['str', 'float']:
                try:
                    rpc_test(uid, otype, f"http://{ip}:{port}/run/predict")
                except:
                    process.kill()
                    raise RuntimeError("rpc test error")


if __name__ == "__main__":
    unittest.main()

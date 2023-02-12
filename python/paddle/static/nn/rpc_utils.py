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

from paddle import fluid


def rpc_call(src_ids=None, url_id=None, url_list=[], voc_path=""):
    request_id = (
        fluid.default_main_program()
        .block(0)
        .create_var(
            name="request_id",
            dtype="int32",
            shape=[1],
            persistable=False,
            stop_gradient=True,
        )
    )

    fluid.default_main_program().block(0).append_op(
        type="rpc_call",
        inputs={
            'X': [src_ids],
            'url_id': [url_id],
        },
        outputs={"Out": [request_id]},
        attrs={"url_list": url_list, "vocab_path": voc_path},
    )
    return request_id


def rpc_result(request_id):
    res = (
        fluid.default_main_program()
        .block(0)
        .create_var(
            name="rpc_res",
            dtype="float32",
            shape=[1000],
            persistable=False,
            stop_gradient=True,
        )
    )
    success = (
        fluid.default_main_program()
        .block(0)
        .create_var(
            name="rpc_success",
            dtype="bool",
            shape=[1],
            persistable=False,
            stop_gradient=True,
        )
    )
    fluid.default_main_program().block(0).append_op(
        type="rpc_result",
        inputs={"X": [request_id]},
        outputs={"Out": [res], "succeed": [success]},
    )
    return res, success

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


class IDGen:
    def __init__(self) -> None:
        self.ids = {}

    def gen_name_with_idx(self, name):
        if name not in self.ids:
            self.ids[name] = -1
        self.ids[name] += 1
        return name + "_" + str(self.ids[name])

    def __call__(self, name) -> str:
        return self.gen_name_with_idx(name)


id_gen = IDGen()


def rpc_call(
    src_ids=None, url="", voc_path="", cvt2str=True, timeout=3000, retry=100
):
    request_id = (
        fluid.default_main_program()
        .block(0)
        .create_var(
            name=id_gen("rpc_request_id"),
            dtype="int32",
            shape=[src_ids.shape[0]],
            persistable=False,
            stop_gradient=True,
        )
    )
    src_ids = src_ids.astype("int32")

    fluid.default_main_program().block(0).append_op(
        type="rpc_call",
        inputs={
            'X': [src_ids],
        },
        outputs={"Out": [request_id]},
        attrs={
            "url": url,
            "vocab_path": voc_path,
            "use_ids": not cvt2str,
            "timeout": timeout,
            "retry": retry,
        },
    )
    return request_id


def rpc_result(request_ids, result_dtype):
    if result_dtype == "float":
        res = (
            fluid.default_main_program()
            .block(0)
            .create_var(
                name=id_gen("rpc_res"),
                dtype="float32",
                shape=[request_ids.shape[0]],
                persistable=False,
                stop_gradient=True,
            )
        )
    elif result_dtype == "str":
        res = (
            fluid.default_main_program()
            .block(0)
            .create_var(
                name=id_gen("rpc_res"),
                dtype="uint8",
                shape=[request_ids.shape[0]],
                persistable=False,
                stop_gradient=True,
            )
        )
    else:
        raise ValueError("result dtype must be one of str ot float")

    success = (
        fluid.default_main_program()
        .block(0)
        .create_var(
            name=id_gen("rpc_success"),
            dtype="bool",
            shape=[1],
            persistable=False,
            stop_gradient=True,
        )
    )
    fluid.default_main_program().block(0).append_op(
        type="rpc_result",
        inputs={"X": [request_ids]},
        outputs={"Out": [res], "succeed": [success]},
        attrs={"res_type": result_dtype},
    )
    return res, success

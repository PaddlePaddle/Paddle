#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib

import paddle

__all__ = [
    "enable_operator_stats_collection",
    "disable_operator_stats_collection",
    "collect_operator_stats",
]


def _get_operator_stats_flag():
    return paddle.get_flags(["FLAGS_low_precision_op_list"])


def _print_operator_stats():
    if not _get_operator_stats_flag():
        return

    # Print the stats of operators, mainly including the calls of dtypes such as different fp32, fp16, bf16 and others.
    print("<{:-^120}>".format(" op list "))
    op_list = paddle.fluid.core.get_low_precision_op_list()
    op_count = 0
    print(
        "<{:-^40}".format(" Op Name "),
        "|",
        "{:-^17}".format(" FP16 Calls "),
        "|",
        "{:-^17}".format(" BF16 Calls "),
        "|",
        "{:-^17}".format(" FP32 Calls"),
        "|",
        "{:-^17}>".format(" Other Calls "),
    )
    for x in op_list:
        # fp16, bf16, fp32, other
        called = op_list[x].split(",")
        print(
            "  %-40s|  %-17s|  %-17s|  %-17s|  %-17s"
            % (x, called[0], called[1], called[2], called[3])
        )
        op_count += 1
    print("<{:-^120}>".format(" op count: " + str(op_count) + " "))


def enable_operator_stats_collection():
    if _get_operator_stats_flag():
        # Clear the previous stats.
        pass
    else:
        paddle.set_flags({'FLAGS_low_precision_op_list': 1})


def disable_operator_stats_collection():
    _print_operator_stats()
    paddle.set_flags({'FLAGS_low_precision_op_list': 0})


@contextlib.contextmanager
def collect_operator_stats():
    enable_operator_stats_collection()
    yield
    disable_operator_stats_collection()

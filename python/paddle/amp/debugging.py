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
from paddle.fluid.framework import dygraph_only

__all__ = [
    "enable_operator_stats_collection",
    "disable_operator_stats_collection",
    "collect_operator_stats",
]


def _get_operator_stats_flag():
    flags = paddle.get_flags(["FLAGS_low_precision_op_list"])
    return flags["FLAGS_low_precision_op_list"]


def _print_operator_stats(op_count_dict):
    """
    Parse and print the stats of operators, mainly including the calls of
    dtypes such as different fp32, fp16, bf16 and others.

    Args:
        op_count_dict(dict): a dict to record the number of calls for different
            operator and dtype. An example is
            {'conv2d': '1,0,0,0', 'elementwise_add': '1,0,0,0'} or
            {'conv2d': [1, 0, 0, 0], 'elementwise_add': [1, 0, 0, 0]}.
    """
    print("<{:-^120}>".format(" op list "))
    total_ops = 0
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
    if op_count_dict is not None and isinstance(op_count_dict, dict):
        for op_type in op_count_dict:
            # fp16, bf16, fp32, other
            value = op_count_dict[op_type]
            if isinstance(value, list):
                called = value
            elif isinstance(value, str):
                called = value.split(",")
            else:
                raise ValueError(
                    "Input {} is expected to be a list of str, but recieved {}.".format(
                        value, type(value)
                    )
                )
            print(
                "  %-40s|  %-17s|  %-17s|  %-17s|  %-17s"
                % (op_type, called[0], called[1], called[2], called[3])
            )
            total_ops += 1
    print("<{:-^120}>\n".format(" op count: " + str(total_ops) + " "))


@dygraph_only
def enable_operator_stats_collection():
    """
    Enable to collect the number of operators for different data types.
    The statistical data are categorized according to four data types, namely
    float32, float16, bfloat16 and others. This funciton is used in pair with
    the corresponding disable function.

    Examples:

     .. code-block:: python

        import paddle

        conv = paddle.nn.Conv2D(3, 2, 3)
        x = paddle.rand([10, 3, 32, 32])

        paddle.amp.debugging.enable_operator_stats_collection()
        # AMP list including conv2d, elementwise_add, reshape2, cast (transfer_dtype)
        with paddle.amp.auto_cast(enable=True, level='O2'):
            out = conv(x)
        # Print to the standard output.
        paddle.amp.debugging.disable_operator_stats_collection()
        # <------------------------------------------------------- op list -------------------------------------------------------->
        # <--------------- Op Name ---------------- | -- FP16 Calls --- | -- BF16 Calls --- | --- FP32 Calls--- | -- Other Calls -->
        #   conv2d                                  |  1                |  0                |  0                |  0
        #   elementwise_add                         |  1                |  0                |  0                |  0
        #   reshape2                                |  1                |  0                |  0                |  0
        #   transfer_dtype                          |  0                |  0                |  3                |  0
        # <----------------------------------------------------- op count: 4 ------------------------------------------------------>

    """
    # Clear the previous stats.
    paddle.fluid.core.clear_low_precision_op_list()
    paddle.set_flags({'FLAGS_low_precision_op_list': 1})


@dygraph_only
def disable_operator_stats_collection():
    """
    Disable the collection the number of operators for different data types.
    This funciton is used in pair with the corresponding enable function.
    The statistical data are categorized according to four data types, namely
    float32, float16, bfloat16 and others, and will be printed after the
    function call.

    Examples:

     .. code-block:: python

        import paddle

        conv = paddle.nn.Conv2D(3, 2, 3)
        x = paddle.rand([10, 3, 32, 32])

        paddle.amp.debugging.enable_operator_stats_collection()
        # AMP list including conv2d, elementwise_add, reshape2, cast (transfer_dtype)
        with paddle.amp.auto_cast(enable=True, level='O2'):
            out = conv(x)
        # Print to the standard output.
        paddle.amp.debugging.disable_operator_stats_collection()
        # <------------------------------------------------------- op list -------------------------------------------------------->
        # <--------------- Op Name ---------------- | -- FP16 Calls --- | -- BF16 Calls --- | --- FP32 Calls--- | -- Other Calls -->
        #   conv2d                                  |  1                |  0                |  0                |  0
        #   elementwise_add                         |  1                |  0                |  0                |  0
        #   reshape2                                |  1                |  0                |  0                |  0
        #   transfer_dtype                          |  0                |  0                |  3                |  0
        # <----------------------------------------------------- op count: 4 ------------------------------------------------------>

    """
    if not _get_operator_stats_flag():
        return

    op_count_dict = paddle.fluid.core.get_low_precision_op_list()
    _print_operator_stats(op_count_dict)
    paddle.set_flags({'FLAGS_low_precision_op_list': 0})


@dygraph_only
@contextlib.contextmanager
def collect_operator_stats():
    """
    The context switcher to enable to collect the number of operators for
    different data types. The statistical data are categorized according
    to four data types, namely float32, float16, bfloat16 and others, and
    will be printed when exiting the context.

    Examples:

     .. code-block:: python

        import paddle

        conv = paddle.nn.Conv2D(3, 2, 3)
        x = paddle.rand([10, 3, 32, 32])

        with paddle.amp.debugging.collect_operator_stats():
            # AMP list including conv2d, elementwise_add, reshape2, cast (transfer_dtype)
            with paddle.amp.auto_cast(enable=True, level='O2'):
                out = conv(x)
        # Print to the standard output.
        # <------------------------------------------------------- op list -------------------------------------------------------->
        # <--------------- Op Name ---------------- | -- FP16 Calls --- | -- BF16 Calls --- | --- FP32 Calls--- | -- Other Calls -->
        #   conv2d                                  |  1                |  0                |  0                |  0
        #   elementwise_add                         |  1                |  0                |  0                |  0
        #   reshape2                                |  1                |  0                |  0                |  0
        #   transfer_dtype                          |  0                |  0                |  3                |  0
        # <----------------------------------------------------- op count: 4 ------------------------------------------------------>

    """
    enable_operator_stats_collection()
    yield
    disable_operator_stats_collection()

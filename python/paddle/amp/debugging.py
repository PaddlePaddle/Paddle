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
import os
import random
from enum import Enum

import numpy as np

import paddle
from paddle.fluid.framework import dygraph_only

__all__ = [
    "enable_operator_stats_collection",
    "disable_operator_stats_collection",
    "collect_operator_stats",
]


class DebugMode(Enum):
    CHECK_NAN_INF_AND_ABORT = 0
    CHECK_NAN_INF = 1
    CHECK_ALL_FOR_OVERFLOW = 2
    CHECK_ALL = 3
    CHECK_ALL_AND_ABORT = 4
    DUMP_ALL = 5


class TensorCheckerConfig:
    """
    Collect the config for checking nan and inf in module or op tensor.

    Args:
    * enable: Whether to enable Tensor's value detection function. The default value is False, which means that these tools will never be used.

    * debug_mode: Debug mode,There are 6 kinds of debug mode.
        CHECK_NAN_INF_AND_ABORT(default): Print or save Tensor key information with NaN/Inf and interrupt the program
        CHECK_NAN_INF: Print or save Tensor critical information with NaN/Inf, but continue to run
        CHECK_ALL_AND_ABORT: Print or save the output Tensor key information of all operators, and interrupt the program if NaN/Inf occurs
        CHECK_ALL_FOR_OVERFLOW: Check the output of the FP32 operator, print or save key Tensor information that exceeds the FP16 representation range (overflow, underflow)
        CHECK_ALL: Print or save output Tensor key information for all operators
        DUMP_ALL: Saves all Tensor data. This mode does not print on the terminal

    * dump_dir: The collection data storage path. If it is None, it will be directly printed to the terminal

    * checked_op_list: A list of operators you want to check

    * skipped_op_list: A list of operators to skip checking

    * debug_step: The iteration scope of debugging

    * stack_height_limit: The maximum depth of the call stack, and supports printing the call stack at the error location. The specific scheme needs to be investigated

    * enable_traceback_filtering: Whether to filter the traceback. The main purpose is to filter out the internal code call stack of the framework and only display the user code call stack

    Examples:
       .. code-block:: python
          import paddle

          checker_config = paddle.amp.debugging.TensorCheckerConfig(enable=True, debug_mode=DebugMode.CHECK_NAN_INF_AND_ABORT)
          paddle.amp.debugging.enable_tensor_checker(checker_config)

          x = paddle.to_tensor([1, 0, 3], place=paddle.CPUPlace(), dtype='float32', stop_gradient=False)
          y = paddle.to_tensor([0.2, 0, 0.5], place=paddle.CPUPlace(), dtype='float32')
          res = paddle.pow(x, y)

          paddle.autograd.backward(res, retain_graph=True)
          paddle.amp.debugging.disable_tensor_checker()

    """

    # For module debugging
    Current_step_id = 0

    def __init__(
        self,
        enable,
        debug_mode=DebugMode.CHECK_NAN_INF_AND_ABORT,
        dump_dir=None,
        checked_op_list=None,
        skipped_op_list=None,
        debug_step=None,
        stack_height_limit=3,
        enable_traceback_filtering=False,
    ):
        self.seed_ = 123
        self.debug_mode_ = debug_mode
        self.dump_dir_ = dump_dir
        self.checked_op_list_ = checked_op_list
        self.skipped_op_list_ = skipped_op_list
        self.debug_step_ = debug_step
        self.stack_height_limit_ = stack_height_limit
        self.start_step_ = None
        self.end_step_ = None
        self.initial_seed_ = 123
        self.enable_ = enable
        self.enable_traceback_filtering_ = enable_traceback_filtering

        # check debug_mode
        if self.debug_mode_.name not in DebugMode.__members__:
            raise ValueError(
                "debug_mode in DebugMode",
                self.debug_mode_,
                DebugMode.__members__,
            )

        # check checked_op_list
        if self.checked_op_list_ is not None:
            if isinstance(self.checked_op_list_, (list, tuple)):
                check_op_list = ",".join(
                    value for value in self.checked_op_list_
                )
                os.environ["Paddle_check_nan_inf_op_list"] = str(check_op_list)
            else:
                raise ValueError("checked_op_list must be list or tuple")

        # check skipped_op_list
        if self.skipped_op_list_ is not None:
            if isinstance(self.skipped_op_list_, (list, tuple)):
                skipped_op_list = ",".join(
                    value for value in self.skipped_op_list_
                )
                os.environ["Paddle_skip_nan_inf_op_list"] = str(skipped_op_list)
            else:
                raise ValueError("skipped_op_list must be list or tuple")

        # check debug_step_
        if self.debug_step_ is not None:
            if isinstance(self.debug_step_, (tuple, list)):
                assert (
                    len(self.debug_step_) == 2
                    and self.debug_step_[1] > self.debug_step_[0]
                )
                self.start_step_, self.end_step_ = self.debug_step_
                self.start_step_ = max(self.start_step_, 0)

        if self.enable_:
            self.__set_seed__(self.enable_)

    def __seed__(self, seed, flag):
        # get random seed
        self.seed_ = seed
        paddle.seed(self.seed_)

        np.random.seed(self.seed_)
        random.seed(self.seed_)

        # set cudnn and cpu
        paddle.set_flags({"FLAGS_cudnn_deterministic": flag})
        paddle.set_flags({"FLAGS_cpu_deterministic": flag})

        # info
        print("AMP Debugging TensorCheckerConfig: seed ", self.seed_)
        print(
            "AMP Debugging TensorCheckerConfig: FLAGS_cudnn_deterministic is ",
            flag,
        )
        print(
            "AMP Debugging TensorCheckerConfig: FLAGS_cpu_deterministic is ",
            flag,
        )

    def __set_seed__(self, enable):
        if enable:
            if self.initial_seed_ != 34342423252:
                self.seed_ = self.initial_seed_
            self.__seed__(self.seed_, True)
        else:
            self.__seed__(self.initial_seed_, False)

    def __env__(self):
        # set debug level
        paddle.set_flags({"FLAGS_check_nan_inf_level": self.debug_mode_.value})

        # set output_dir
        if self.dump_dir_ is not None:
            paddle.fluid.core.set_nan_inf_debug_path(self.dump_dir_)

        # set stack_height_limit
        if isinstance(self.stack_height_limit_, (int)):
            paddle.set_flags(
                {"FLAGS_call_stack_level": self.stack_height_limit_}
            )
        else:
            raise ValueError("stack_height_limit must be int")

    def __set_env__(self, check_flag):
        paddle.set_flags({"FLAGS_check_nan_inf": check_flag})
        if check_flag:
            self.__env__()

    def check(self):
        TensorCheckerConfig.Current_step_id += 1
        if self.enable_:
            if self.start_step_ is not None and self.end_step_ is not None:
                if (
                    self.start_step_ > TensorCheckerConfig.Current_step_id
                    or TensorCheckerConfig.Current_step_id >= self.end_step_
                ):
                    return False
            return True
        return False

    def run(self):
        self.__set_env__(self.enable_)

    def end(self):
        self.__set_env__(False)


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


@dygraph_only
def enable_tensor_checker(checker_config):
    """
    enable_tensor_checker(checker_config) is enables model level accuracy checking, which is used together with disables_tensor_checker() to achieve model level precision checking through the combination of these two APIs, checking the output Tensors of all operators within the specified range.

    Attention:

    * If disable is called before loss. backward()_tensor_checker(), the gradient operator is not checked;

    * If disable is called before optimizer.step() tensor_checker(), the optimizer and other weight update related operators will not be checked

    Examples:
       .. code-block:: python
           import paddle

           checker_config = paddle.amp.debugging.TensorCheckerConfig(enable=True, debug_mode=DebugMode.CHECK_NAN_INF_AND_ABORT)
           paddle.amp.debugging.enable_tensor_checker(checker_config)

           x = paddle.to_tensor([1, 0, 3], place=paddle.CPUPlace(), dtype='float32', stop_gradient=False)
           y = paddle.to_tensor([0.2, 0, 0.5], place=paddle.CPUPlace(), dtype='float32')
           res = paddle.pow(x, y)
           paddle.autograd.backward(res, retain_graph=True)

           paddle.amp.debugging.disable_tensor_checker()
    """
    if checker_config.check():
        checker_config.run()
    else:
        checker_config.end()


@dygraph_only
def disable_tensor_checker():
    """
    disable_tensor_checker() to disables the accuracy checking, which is used together with enables_tensor_checker(config) to achieve model level precision checking through the combination of these two APIs, checking the output Tensors of all operators within the specified range.

    Attention:

    * If disable_tensor_checker() is called before loss.backward(), the gradient operator is not checked;

    * If disable_tensor_checker() is called before optimizer.step(), the optimizer and other weight update related operators will not be checked

    Examples:
       .. code-block:: python
           import paddle

           checker_config = paddle.amp.debugging.TensorCheckerConfig(enable=True, debug_mode=DebugMode.CHECK_NAN_INF_AND_ABORT)
           paddle.amp.debugging.enable_tensor_checker(checker_config)

           x = paddle.to_tensor([1, 0, 3], place=paddle.CPUPlace(), dtype='float32', stop_gradient=False)
           y = paddle.to_tensor([0.2, 0, 0.5], place=paddle.CPUPlace(), dtype='float32')
           res = paddle.pow(x, y)
           paddle.autograd.backward(res, retain_graph=True)

           paddle.amp.debugging.disable_tensor_checker()

    """
    paddle.set_flags({"FLAGS_check_nan_inf": 0})


@dygraph_only
def check_numerics(x, checker_config):
    """
    check_numerics is the api to check the tensor whether has nan or inf.

    Args:
        x (Tensor) - Tensor or LoDTensor of any dimensions.
        checker_config (TensorCheckerConfig) - Configuration for precision checking.

    Examples:
       .. code-block:: python
           import paddle

           checker_config = paddle.amp.debugging.TensorCheckerConfig(enable=True, debug_mode=DebugMode.CHECK_NAN_INF_AND_ABORT)
           x = paddle.to_tensor([1, 0, 3], place=paddle.CPUPlace(), dtype='float32', stop_gradient=False)
           paddle.amp.debugging.checker_config(x, checker_config)
    """
    if checker_config.check():
        checker_config.run()
    else:
        checker_config.end()
    paddle.fluid.core.check_numerics("check_numerics", x)

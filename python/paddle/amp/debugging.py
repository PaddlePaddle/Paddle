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
import random
from enum import Enum

import numpy as np

import paddle
from paddle.fluid import core
from paddle.fluid.framework import dygraph_only

__all__ = [
    "DebugMode",
    "TensorCheckerConfig",
    "enable_operator_stats_collection",
    "disable_operator_stats_collection",
    "collect_operator_stats",
    "enable_tensor_checker",
    "disable_tensor_checker",
    "compare_accuracy",
]


class DebugMode(Enum):
    """
    The DebugMode is a feature that helps to present the state of the TensorCheckerConfig. Each DebugMode has a specific meaning, which is explained below:

    - DebugMode.CHECK_NAN_INF_AND_ABORT: This mode prints or saves information about Tensors that contain NaN/Inf and interrupts the program.

    - DebugMode.CHECK_NAN_INF: This mode prints or saves critical information about Tensors that contain NaN/Inf but allows the program to continue running.

    - DebugMode.CHECK_ALL_FOR_OVERFLOW: This mode checks the output of the FP32 operator and prints or saves information about key Tensors that exceed the FP16 representation range, such as overflow or underflow.

    - DebugMode.CHECK_ALL: This mode prints or saves output Tensor key information for all operators.

    """

    CHECK_NAN_INF_AND_ABORT = 0
    CHECK_NAN_INF = 1
    CHECK_ALL_FOR_OVERFLOW = 2
    CHECK_ALL = 3
    # CHECK_ALL_AND_ABORT = 4
    # DUMP_ALL = 5


def set_checked_op_list(checked_op_list):
    # check checked_op_list
    if checked_op_list is not None:
        if isinstance(checked_op_list, (list, tuple)):
            check_op_list = ",".join(value for value in checked_op_list)
            paddle.fluid.core.set_checked_op_list(check_op_list)
        else:
            raise ValueError("checked_op_list must be list or tuple")


def set_skipped_op_list(skipped_op_list):
    # check skipped_op_list
    if skipped_op_list is not None:
        if isinstance(skipped_op_list, (list, tuple)):
            skip_op_list = ",".join(value for value in skipped_op_list)
            paddle.fluid.core.set_skipped_op_list(skip_op_list)
        else:
            raise ValueError("skipped_op_list must be list or tuple")


class TensorCheckerConfig:
    """
    The purpose of this class is to collect the configuration for checking NaN and Inf values in the tensors of a module or operator. It takes the following arguments:

    Args:
        enable(bool): Indicating whether to enable the detection of NaN and Inf values in tensors. The default value is False, which means that these tools will not be used.

        debug_mode(DebugMode, optional): A parameter that determines the type of debugging to be used. Default is DebugMode.CHECK_NAN_INF_AND_ABORT.

        output_dir(string, optional): The path to store collected data. If this parameter is set to None, the data will be printed to the terminal. Default is None.

        checked_op_list(list|tuple, optional): Specifies a list of operators that need to be checked during program execution, for example, checked_op_list=['elementwise_add', 'conv2d'], indicating that the output results of elementwise_add and conv2d should be checked for nan/inf during program execution. Default is None.

        skipped_op_list(list|tuple, optional): Specifies a list of operators that do not need to be checked during program execution, for example, skipped_op_list=['elementwise_add', 'conv2d'], indicating that the output results of elementwise_add and conv2d should not be checked for nan/inf during program execution. None is None.

        debug_step(list|tuple, optional): A list or tuple used primarily for nan/inf checking during model training. For example, debug_step=[1,5] indicates that nan/inf checking should only be performed on model training iterations 1 to 5. Default is None.

        stack_height_limit(int, optional): An integer value specifying the maximum depth of the call stack. This feature supports printing the call stack at the error location. Currently, only enabling or disabling call stack printing is supported. If you want to print the corresponding C++ call stack when NaN is detected in GPU Kernel, set stack_height_limit to 1, otherwise set it to 0. Default is 1.

    Examples:

        ..  code-block:: python

            import paddle

            checker_config = paddle.amp.debugging.TensorCheckerConfig(enable=True, debug_mode=paddle.amp.debugging.DebugMode.CHECK_NAN_INF)
            paddle.amp.debugging.enable_tensor_checker(checker_config)

            x = paddle.to_tensor([1, 0, 3], place=paddle.CPUPlace(), dtype='float32', stop_gradient=False)
            y = paddle.to_tensor([0.2, 0, 0.5], place=paddle.CPUPlace(), dtype='float32')
            res = paddle.pow(x, y)
            paddle.autograd.backward(res, retain_graph=True)
            paddle.amp.debugging.disable_tensor_checker()

            #[PRECISION] [ERROR] in [device=cpu, op=elementwise_pow_grad, tensor=, dtype=fp32], numel=3, num_nan=1, num_inf=0, num_zero=0, max=2.886751e-01, min=2.000000e-01, mean=-nan

            # when DebugMode.CHECK_NAN_INF_AND_ABORT and stack_height_limit = 1
            #Traceback (most recent call last):
            #    res = paddle.pow(x, y)
            #  File "/usr/local/lib/python3.8/dist-packages/paddle/tensor/math.py", line 447, in pow
            #    return _C_ops.elementwise_pow(x, y)

    """

    # For module debugging
    current_step_id = 0

    def __init__(
        self,
        enable,
        debug_mode=DebugMode.CHECK_NAN_INF_AND_ABORT,
        output_dir=None,
        checked_op_list=None,
        skipped_op_list=None,
        debug_step=None,
        stack_height_limit=1,
    ):

        self.enable = enable
        self.debug_mode = debug_mode
        self.output_dir = output_dir

        self.checked_op_list = checked_op_list
        self.skipped_op_list = skipped_op_list

        self.debug_step = debug_step
        self.stack_height_limit = stack_height_limit

        self.start_step = None
        self.end_step = None

        self.seed = 123
        self.initial_seed = 123

        # check debug_step
        if debug_step is not None:
            if isinstance(debug_step, (tuple, list)):
                assert (
                    len(self.debug_step) == 2
                    and self.debug_step[1] > self.debug_step[0]
                )
                self.start_step, self.end_step = self.debug_step
                self.start_step = max(self.start_step, 0)
            else:
                raise ValueError("debug_step must be list or tuple")

        if core.is_compiled_with_cuda():
            for i in range(core.get_cuda_device_count()):
                self.initial_seed = core.default_cuda_generator(
                    i
                ).initial_seed()
        elif core.is_compiled_with_xpu():
            for i in range(core.get_xpu_device_count()):
                self.initial_seed = core.default_xpu_generator(i).initial_seed()

        self.initial_seed = core.default_cpu_generator().initial_seed()

        # check debug_mode
        if self.debug_mode.name not in DebugMode.__members__:
            raise ValueError(
                "debug_mode in DebugMode",
                self.debug_mode,
                DebugMode.__members__,
            )

        set_checked_op_list(self.checked_op_list)

        set_skipped_op_list(self.skipped_op_list)

        if self.enable:
            self._set_seed(self.enable)

    def _set_seed(self, flag):
        if self.initial_seed != self.seed:
            self.seed = self.initial_seed

        if self.seed > np.iinfo(np.uint32).max or self.seed < 0:
            print("[Warnning: Seed must be between 0 and 2**32 - 1")
            self.seed = 123

        # get random seed
        paddle.seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # info
        print("AMP Debugging TensorCheckerConfig: seed ", self.seed)

        # set cudnn and cpu
        if core.is_compiled_with_cuda():
            paddle.set_flags({"FLAGS_cudnn_deterministic": flag})
            print(
                "AMP Debugging TensorCheckerConfig: FLAGS_cudnn_deterministic is ",
                flag,
            )

        paddle.set_flags({"FLAGS_cpu_deterministic": flag})
        print(
            "AMP Debugging TensorCheckerConfig: FLAGS_cpu_deterministic is ",
            flag,
        )

    def _set_env(self, check_flag):
        paddle.set_flags({"FLAGS_check_nan_inf": check_flag})
        if check_flag:
            # set debug level
            paddle.set_flags(
                {"FLAGS_check_nan_inf_level": self.debug_mode.value}
            )

            # set output_dir
            if self.output_dir is not None:
                paddle.fluid.core.set_nan_inf_debug_path(self.output_dir)

            # set stack_height_limit
            if isinstance(self.stack_height_limit, (int)):
                paddle.fluid.core.set_nan_inf_stack_limit(
                    self.stack_height_limit
                )
            else:
                raise ValueError("stack_height_limit must be int")

    def update_and_check_step_id(self):
        if self.enable:
            if self.start_step is not None and self.end_step is not None:
                if (
                    self.start_step > TensorCheckerConfig.current_step_id
                    or TensorCheckerConfig.current_step_id >= self.end_step
                ):
                    return False
                else:
                    TensorCheckerConfig.current_step_id += 1
            return True
        return False

    def start_check_nan_inf(self):
        if self.enable:
            self._set_env(self.enable)

    def stop_check_nan_inf(self):
        self._set_env(False)


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
        for op_type in sorted(op_count_dict):
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
    float32, float16, bfloat16 and others. This function is used in pair with
    the corresponding disable function.

    Examples:

        ..  code-block:: python

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
    This function is used in pair with the corresponding enable function.
    The statistical data are categorized according to four data types, namely
    float32, float16, bfloat16 and others, and will be printed after the
    function call.

    Examples:

        ..  code-block:: python

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

        ..  code-block:: python

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


def compare_accuracy(
    dump_path,
    another_dump_path,
    output_filename,
    loss_scale=1,
    dump_all_tensors=False,
):
    r"""
    This is a precision comparison tool that can be used to compare log data of float16 and float32.

    Args:
        dump_path(str): The path of the running log, such as the log for execution using the float32 data type.
        another_dump_path(str): the path of another running log ,such as the log for execution using the float16 data type.
        output_filename(str): the excel file nmae of compare output.
        loss_scale(float, optional): the loss_scale during the training phase. Default is 1.
        dump_all_tensors(bool, optional): dump all tensor, It is currently not support. Default is False.

    Examples:

        ..  code-block:: python

            import paddle
            from paddle.fluid import core
            try:
                import xlsxwriter as xlw
            except ImportError:
                import subprocess

                subprocess.check_call(
                    ['python', '-m', 'pip', 'install', 'xlsxwriter==3.0.9']
                )
                import xlsxwriter as xlw

            if core.is_compiled_with_cuda():
                paddle.set_flags(
                    {"FLAGS_check_nan_inf": 1, "FLAGS_check_nan_inf_level": 3}
                )
                path = "workerlog_log_dir"
                paddle.fluid.core.set_nan_inf_debug_path(path)
                x = paddle.to_tensor(
                    [2, 3, 4, 0], dtype="float32"
                )
                y = paddle.to_tensor(
                    [1, 5, 2, 0], dtype="float32"
                )
                z1 = x + y
                out_excel = "compary_accuracy_out_excel.csv"
                paddle.amp.debugging.compare_accuracy(
                    path, path, out_excel, loss_scale=1, dump_all_tensors=False
                )
    """
    assert dump_all_tensors is False, "It is currently not supported."
    paddle.amp.accuracy_compare.compare_accuracy(
        dump_path,
        another_dump_path,
        output_filename,
        loss_scale,
        dump_all_tensors=False,
    )


def enable_tensor_checker(checker_config):
    """
    The enable_tensor_checker(checker_config) function enables model-level accuracy checking and is used in combination with disables_tensor_checker() to achieve model-level precision checking by checking the output Tensors of all operators within the specified range.

    Args:
        checker_config(TensorCheckerConfig): Checker_config is to collect the configuration for checking NaN and Inf values in the tensors of a module or operator.

    Note:
        If disable_tensor_checker() is called before backward(), the gradient operator will not be checked.
        If disable_tensor_checker() is called before optimizer.step(), the optimizer and other weight update related operators will not be checked.

    Examples:

        ..  code-block:: python

            import paddle

            checker_config = paddle.amp.debugging.TensorCheckerConfig(enable=True, debug_mode=paddle.amp.debugging.DebugMode.CHECK_NAN_INF)
            paddle.amp.debugging.enable_tensor_checker(checker_config)

            x = paddle.to_tensor([1, 0, 3], place=paddle.CPUPlace(), dtype='float32', stop_gradient=False)
            y = paddle.to_tensor([0.2, 0, 0.5], place=paddle.CPUPlace(), dtype='float32')
            res = paddle.pow(x, y)
            paddle.autograd.backward(res, retain_graph=True)
            paddle.amp.debugging.disable_tensor_checker()
            #[PRECISION] [ERROR] in [device=cpu, op=elementwise_pow_grad, tensor=, dtype=fp32], numel=3, num_nan=1, num_inf=0, num_zero=0, max=2.886751e-01, min=2.000000e-01, mean=-nan

            # when DebugMode.CHECK_NAN_INF_AND_ABORT and stack_height_limit = 1
            # Traceback (most recent call last):
            #   File "tp.py", line 8, in <module>
            #     res = paddle.pow(x, y)
            #   File "/usr/local/lib/python3.8/dist-packages/paddle/tensor/math.py", line 447, in pow
            #     return _C_ops.elementwise_pow(x, y)

    """
    if checker_config.update_and_check_step_id():
        checker_config.start_check_nan_inf()
    else:
        checker_config.stop_check_nan_inf()


def disable_tensor_checker():
    """
    disable_tensor_checker() is used to disable accuracy checking, and is used together with enable_tensor_checker(config) to achieve model-level precision checking by checking the output Tensors of all operators within the specified range.

    Note:
        If disable_tensor_checker() is called before backward(), the gradient operator will not be checked;
        If disable_tensor_checker() is called before optimizer.step(), the optimizer and other weight update related operators will not be checked.

    Examples:

        ..  code-block:: python

            import paddle

            checker_config = paddle.amp.debugging.TensorCheckerConfig(enable=True, debug_mode=paddle.amp.debugging.DebugMode.CHECK_NAN_INF)
            paddle.amp.debugging.enable_tensor_checker(checker_config)

            x = paddle.to_tensor([1, 0, 3], place=paddle.CPUPlace(), dtype='float32', stop_gradient=False)
            y = paddle.to_tensor([0.2, 0, 0.5], place=paddle.CPUPlace(), dtype='float32')
            res = paddle.pow(x, y)
            paddle.autograd.backward(res, retain_graph=True)
            paddle.amp.debugging.disable_tensor_checker()
            #[PRECISION] [ERROR] in [device=cpu, op=elementwise_pow_grad, tensor=, dtype=fp32], numel=3, num_nan=1, num_inf=0, num_zero=0, max=2.886751e-01, min=2.000000e-01, mean=-nan

            # when DebugMode.CHECK_NAN_INF_AND_ABORT and stack_height_limit = 1
            # Traceback (most recent call last):
            #     res = paddle.pow(x, y)
            #   File "/usr/local/lib/python3.8/dist-packages/paddle/tensor/math.py", line 447, in pow
            #     return _C_ops.elementwise_pow(x, y)

    """
    paddle.set_flags({"FLAGS_check_nan_inf": 0})

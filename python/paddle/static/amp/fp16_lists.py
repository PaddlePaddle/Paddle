#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import logging

import paddle
from paddle.amp.amp_lists import (
    EXTRA_BLACK_LIST,
    FP16_BLACK_LIST,
    FP16_WHITE_LIST,
)
from paddle.base import core
from paddle.base.log_helper import get_logger

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)

black_list = FP16_BLACK_LIST
_extra_black_list = EXTRA_BLACK_LIST
white_list = FP16_WHITE_LIST


def check_amp_dtype(dtype):
    """
    Check amp_dtype: float16 or bfloat16
    """
    if isinstance(dtype, str):
        dtype = dtype.lower()
    if dtype not in ['float16', 'bfloat16']:
        raise ValueError(
            "If enable AMP, dtype should be 'float16' or 'bfloat16'."
        )
    return dtype


def get_low_precision_vartype(dtype):
    if isinstance(dtype, (core.VarDesc.VarType, core.DataType)):
        return dtype
    elif isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "float16":
            var_type = core.VarDesc.VarType.FP16
        elif dtype == "bfloat16":
            var_type = core.VarDesc.VarType.BF16
        else:
            raise ValueError(
                "If enable AMP, dtype should be 'float16' or 'bfloat16'."
            )
        return var_type
    else:
        raise TypeError(
            f"The type of dtype is expected to be string or core.VarDesc.VarType, but received {type(dtype)}."
        )


def get_low_precision_dtypestr(dtype):
    if isinstance(dtype, str):
        return check_amp_dtype(dtype)
    elif isinstance(dtype, (core.VarDesc.VarType, core.DataType)):
        if dtype == paddle.float16:
            return "float16"
        elif dtype == paddle.bfloat16:
            return "bfloat16"
        else:
            raise ValueError(
                "If enable AMP, dtype should be core.VarDesc.VarType.FP16 or core.VarDesc.VarType.BF16."
            )
    else:
        raise TypeError(
            f"The type of dtype is expected to be string or core.VarDesc.VarType, but received {type(dtype)}."
        )


def _get_sys_unsupported_list(dtype):
    var_type = get_low_precision_vartype(dtype)

    # The set of ops that don't support fp16 calculation
    device = None
    if core.is_compiled_with_xpu():
        device = 'XPU'
    elif isinstance(
        paddle.framework._current_expected_place(), paddle.CustomPlace
    ):
        device = paddle.framework._current_expected_place().get_device_type()
    else:
        device = 'GPU'
    all_ops, _, sys_unsupported_list = core.op_supported_infos(device, var_type)

    # sys_unsupported_list will include the following ops.
    supported_fp16_list = {
        "conditional_block",
        "conditional_block_infer",
        "select_input",
        "while",
        "cast",
        "tensor_array_to_tensor",
        "lod_array_length",
        "write_to_array",
    }
    sys_unsupported_list -= supported_fp16_list

    return device, sys_unsupported_list, all_ops


def _get_unsupported_list(dtype):
    # The set of ops that don't support fp16 calculation
    _, _sys_unsupported_list, _sys_all_list = _get_sys_unsupported_list(dtype)
    return _sys_unsupported_list, _sys_all_list


# The three sets listed below are changed dynamically. They don't contain all
# paddle ops currently.

# The set of ops that support fp16 calculation and are considered numerically-
# safe and performance-critical. These ops are always converted to fp16.

_only_supported_fp16_list = {'resnet_unit', 'fused_bn_add_activation'}


def _get_white_list(dtype):
    white_list_for_dtype = copy.copy(FP16_WHITE_LIST)
    if dtype == 'float16':
        white_list_for_dtype = white_list_for_dtype | _only_supported_fp16_list
    return white_list_for_dtype


def _get_black_list():
    _black_list = copy.copy(FP16_BLACK_LIST)
    _black_list = _black_list | EXTRA_BLACK_LIST
    return _black_list


class AutoMixedPrecisionLists:
    """
    AutoMixedPrecisionLists is a class for black/white list. It can update
    pre-defined black list and white list according to users' custom black
    white lists. The lists are used for an algorithm which determines op's
    execution mode (fp32, fp16 or bf16).

    Args:
        custom_white_list (set): Users' custom white list.
        custom_black_list (set): Users' custom black list.
        custom_black_varnames (set): Users' custom black variables' names.
        dtype (str): the low precision dtype, which can be set to 'float16' or 'bfloat16'.
    """

    def __init__(
        self,
        custom_white_list=None,
        custom_black_list=None,
        custom_black_varnames=None,
        dtype="float16",
    ):
        self.amp_dtype = check_amp_dtype(dtype)
        self._custom_white_list = custom_white_list
        self._custom_black_list = custom_black_list
        self.white_list = copy.copy(_get_white_list(self.amp_dtype))
        self.black_list = copy.copy(_get_black_list())
        self.gray_list = copy.copy(gray_list)
        unsupported_list, sys_all_list = _get_unsupported_list(self.amp_dtype)
        self.unsupported_list = copy.copy(unsupported_list)
        self.all_list = copy.copy(sys_all_list)
        self.black_varnames = copy.copy(custom_black_varnames)
        self._update_list()

    def _update_list(self):
        """
        Update black and white list according to users' custom list.
        """
        _logger.debug(f"---- custom_white_list {self._custom_white_list} ---- ")
        _logger.debug(f"---- custom_black_list {self._custom_black_list} ---- ")
        _logger.debug(f"---- custom_black_varnames {self.black_varnames} ---- ")
        if self._custom_white_list and self._custom_black_list:
            for op_name in self._custom_white_list:
                if op_name in self._custom_black_list:
                    raise ValueError(
                        f"The given custom_white_list overlaps custom_black_list with < {op_name} >!"
                    )
        if self._custom_white_list:
            for op_name in self._custom_white_list:
                if op_name in self.black_list:
                    self.black_list.remove(op_name)
                elif op_name in self.gray_list:
                    self.gray_list.remove(op_name)
                self.white_list.add(op_name)
        if self._custom_black_list:
            for op_name in self._custom_black_list:
                if op_name in self.white_list:
                    self.white_list.remove(op_name)
                elif op_name in self.gray_list:
                    self.gray_list.remove(op_name)
                self.black_list.add(op_name)
                self.unsupported_list.add(op_name)
        device, sys_unsupported_list, _ = _get_sys_unsupported_list(
            self.amp_dtype
        )
        actual_unsupported_list = []
        for op_name in sys_unsupported_list:
            if op_name in self.white_list:
                actual_unsupported_list.append(op_name)
        if len(actual_unsupported_list) > 0:
            _logger.warning(
                f"On current {device}, {self.amp_dtype} is not supported for operators < {actual_unsupported_list} > in white_list!"
            )


# This set contains two types of ops. All ops supported fp16 calculation. One
# of two types is considered numerically-safe, but may be made unsafe by an
# upstream blacklist op. Another type do not have numerically-significant
# effects, like stack, flatten2.
gray_list = {
    'elementwise_add',
    'elementwise_sub',
    'elementwise_mul',
    'elementwise_div',
    'elementwise_max',
    'elementwise_min',
    'elementwise_pow',
    'elementwise_mod',
    'elementwise_floordiv',
    'batch_norm',
    'layer_norm',
    'tanh',
    'sigmoid',
    'top_k',
    'pool2d',
    'pool3d',
    'dropout',
    'relu',
    'relu6',
    'leaky_relu',
    'soft_relu',
    'flatten2',
    'stack',
    'unstack',
    'uniform_random',
    'uniform_random_batch_size_like',
    'gaussian_random',
    'slice',
    'rank',
    'scale',
    'transpose2',
    'reshape2',
    'gather',
    'fill_constant',
    'get_tensor_from_selected_rows',
    'sign',
    'cast',
    'fused_bn_add_activation',
    'c_identity',
    'c_concat',
    'c_allreduce_sum',
    'concat',
    'split',
    'fused_feedforward',
    'fused_attention',
    'fused_multi_transformer',
}

CustomOpLists = AutoMixedPrecisionLists

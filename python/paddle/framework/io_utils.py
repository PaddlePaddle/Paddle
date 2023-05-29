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

import logging
import math
import os
import pickle
import sys
from io import BytesIO

import numpy as np

import paddle
from paddle.fluid import core
from paddle.fluid.framework import Parameter, Variable, static_only
from paddle.fluid.log_helper import get_logger
from paddle.fluid.wrapped_decorator import signature_safe_contextmanager

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)

# This file contains various utility functions that are used in static.io(io related api that used in static graph)
# and framework.io(io related api that used in dygraph)


class _open_buffer:
    def __init__(self, buffer):
        self.buffer = buffer

    def __enter__(self):
        return self.buffer


class _buffer_reader(_open_buffer):
    def __init__(self, buffer):
        super().__init__(buffer)
        self.initial_tell = self.buffer.tell()

    def __exit__(self, *args):
        # `args[0]` is type of exception. When the `read` is abnormal, the file pointer returns to the initial position.
        if args[0] is not None:
            self.buffer.seek(self.initial_tell)


class _buffer_writer(_open_buffer):
    def __exit__(self, *args):
        self.buffer.flush()


def _is_file_path(path):
    return isinstance(path, str)


def _open_file_buffer(path_or_buffer, mode):

    if _is_file_path(path_or_buffer):
        return open(path_or_buffer, mode)
    else:
        if 'w' in mode:
            return _buffer_writer(path_or_buffer)
        elif 'r' in mode:
            return _buffer_reader(path_or_buffer)
        else:
            raise ValueError(f"Expected 'r' or 'w' in mode but got {mode}")


def _is_memory_buffer(buffer):
    return isinstance(buffer, BytesIO)


def is_persistable(var):
    """

    Check whether the given variable is persistable.

    Args:
        var(Variable): The variable to be checked.

    Returns:
        bool: True if the given `var` is persistable
        False if not.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid

            paddle.enable_static()
            param = fluid.default_main_program().global_block().var('fc.b')
            res = fluid.io.is_persistable(param)
    """
    if (
        var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH
        or var.desc.type() == core.VarDesc.VarType.FETCH_LIST
        or var.desc.type() == core.VarDesc.VarType.READER
    ):
        return False
    return var.persistable


def is_parameter(var):
    """
    Check whether the given variable is an instance of Parameter.

    Args:
        var(Variable): The variable to be checked.

    Returns:
        bool: True if the given `var` is an instance of Parameter,
        False if not.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid

            paddle.enable_static()
            param = fluid.default_main_program().global_block().var('fc.w')
            res = fluid.io.is_parameter(param)
    """
    return isinstance(var, Parameter)


def is_belong_to_optimizer(var):
    if not (isinstance(var, Parameter) or var.desc.need_check_feed()):
        return is_persistable(var)

    return False


def _clone_var_in_block_(block, var):
    assert isinstance(var, Variable)
    if var.desc.type() == core.VarDesc.VarType.LOD_TENSOR:
        return block.create_var(
            name=var.name,
            shape=var.shape,
            dtype=var.dtype,
            type=var.type,
            lod_level=var.lod_level,
            persistable=True,
        )
    else:
        return block.create_var(
            name=var.name,
            shape=var.shape,
            dtype=var.dtype,
            type=var.type,
            persistable=True,
        )


@signature_safe_contextmanager
def _load_program_scope(main=None, startup=None, scope=None):
    prog = main if main else paddle.fluid.Program()
    startup_prog = startup if startup else paddle.fluid.Program()
    scope = scope if scope else paddle.fluid.core.Scope()
    with paddle.fluid.scope_guard(scope):
        with paddle.fluid.program_guard(prog, startup_prog):
            with paddle.fluid.unique_name.guard():
                with paddle.fluid.framework._dygraph_guard(None):
                    yield


@static_only
def _legacy_static_save(param_dict, model_path, protocol=2):
    def get_tensor(var):
        if isinstance(var, core.eager.Tensor):
            return np.array(var)
        elif isinstance(var, core.LoDTensor):
            return np.array(var)
        return var

    param_dict = {name: get_tensor(param_dict[name]) for name in param_dict}

    # When value of dict is lager than 4GB ,there is a Bug on 'MAC python3'
    if (
        _is_file_path(model_path)
        and sys.platform == 'darwin'
        and sys.version_info.major == 3
    ):
        pickle_bytes = pickle.dumps(param_dict, protocol=protocol)
        with open(model_path, 'wb') as f:
            max_bytes = 2**30
            for i in range(0, len(pickle_bytes), max_bytes):
                f.write(pickle_bytes[i : i + max_bytes])
    else:
        with _open_file_buffer(model_path, 'wb') as f:
            pickle.dump(param_dict, f, protocol=protocol)


def _pickle_loads_mac(path, f):
    pickle_bytes = bytearray(0)
    file_size = os.path.getsize(path)
    max_bytes = 2**30
    for _ in range(0, file_size, max_bytes):
        pickle_bytes += f.read(max_bytes)
    load_result = pickle.loads(pickle_bytes, encoding='latin1')
    return load_result


def _pack_loaded_dict(load_obj):
    if isinstance(load_obj, dict):
        unpack_info = 'UnpackBigParamInfor@@'
        if unpack_info in load_obj:
            removes = []
            for key, value in load_obj[unpack_info].items():
                slices = [load_obj[part] for part in value["slices"]]
                load_obj[key] = np.concatenate(slices).reshape(
                    value["OriginShape"]
                )
                removes += value["slices"]
            for key in removes:
                load_obj.pop(key)
            load_obj.pop(unpack_info)

    return load_obj


def _unpack_saved_dict(saved_obj, protocol):
    temp_saved_obj = {}
    unpack_infor = {}
    # When pickle protocol=2 or protocol=3 the serialized object cannot be larger than 4G.
    if 1 < protocol < 4:
        if isinstance(saved_obj, dict):
            for key, value in saved_obj.items():
                if isinstance(value, np.ndarray):
                    MAX_NUMBER_OF_ELEMENT = int(
                        (2**30 - 1) / value.dtype.itemsize
                    )
                    num_element = np.prod(value.shape)
                    if num_element > MAX_NUMBER_OF_ELEMENT:
                        unpack_infor[key] = {}
                        unpack_infor[key]["OriginShape"] = value.shape
                        unpack_infor[key]["slices"] = []
                        value = value.flatten()
                        for i in range(
                            int(
                                math.ceil(
                                    num_element * 1.0 / MAX_NUMBER_OF_ELEMENT
                                )
                            )
                        ):
                            part_name = key + "@@." + str(i)
                            unpack_infor[key]["slices"].append(part_name)
                            temp_saved_obj[part_name] = value[
                                i
                                * MAX_NUMBER_OF_ELEMENT : MAX_NUMBER_OF_ELEMENT
                                * (i + 1)
                            ]

    if unpack_infor:
        for key, value in unpack_infor.items():
            if key in saved_obj:
                saved_obj.pop(key)
                for part in value['slices']:
                    saved_obj[part] = temp_saved_obj[part]
        saved_obj['UnpackBigParamInfor@@'] = unpack_infor
    return saved_obj
